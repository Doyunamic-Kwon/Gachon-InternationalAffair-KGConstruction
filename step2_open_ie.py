"""
Step 2. Open IE — 두 가지 방식 비교
  A) Pattern-based  : SpaCy 의존구문 파싱 → (S, 동사구, O) 자유 트리플
  B) LLM-based      : Claude API(Batches) → 자유형 트리플 + OIA 온톨로지 매핑

적용 대상: candidates.jsonl (1730개 무라벨 후보) + raw HTML 전체
"""
import json
import os
import re
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report

# ─── 경로 상수 ────────────────────────────────────────────────────────────────
CANDIDATES_PATH = Path("data/re_fixed_v6/candidates.jsonl")
TRAIN_PATH      = Path("data/re_fixed_v6/train.jsonl")
GOLD_DIR        = Path("data/re_fixed_v6/labeling_by_relation")
RAW_DIR         = Path("data/raw")
REPORT_DIR      = Path("reports/open_ie")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# OIA 도메인 온톨로지 (12개 관계 + NO_RELATION)
OIA_RELATIONS = [
    "HAS_CONTACT_EMAIL", "HAS_CONTACT_PHONE", "HAS_DEADLINE", "HAS_FEE",
    "MENTIONS_EXAM_LEVEL", "NO_RELATION", "REFERENCES_ATTACHMENT",
    "REFERENCES_EXTERNAL_RESOURCE", "announced_by", "mentions",
    "requires_document", "requires_qualification",
]

# ─── 유틸 ─────────────────────────────────────────────────────────────────────
def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def strip_markers(text: str) -> str:
    """[E1], [/E1], [E2], [/E2] 태그 제거"""
    return re.sub(r"\[/?E[12]\]", "", text).strip()

def load_gold_df() -> pd.DataFrame:
    dfs = []
    for f in GOLD_DIR.glob("*.csv"):
        df = pd.read_csv(f)
        df["final_relation"] = df["gold_relation"].fillna(df.get("suggested_relation", "NO_RELATION"))
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True).dropna(subset=["final_relation"])


# ══════════════════════════════════════════════════════════════════════════════
# A. Pattern-based Open IE (SpaCy 의존구문)
# ══════════════════════════════════════════════════════════════════════════════
def _extract_pattern_triple(text: str, nlp) -> dict | None:
    """SpaCy로 (S, V, O) 추출 + E1/E2 엔티티 포함 여부 확인"""
    raw = re.sub(r"\[/?E[12]\]", "", text)
    doc = nlp(raw)
    subj = verb = obj = ""
    for token in doc:
        if token.dep_ in ("nsubj", "csubj") and not subj:
            subj = token.text
        if token.pos_ == "VERB" and not verb:
            verb = token.text
        if token.dep_ in ("obj", "iobj", "pobj") and not obj:
            obj = token.text
    if subj and verb:
        return {"subject": subj, "predicate": verb, "object": obj or ""}
    return None

def run_pattern_open_ie(candidates: list[dict]) -> list[dict]:
    """전체 candidates에 SpaCy Open IE 적용"""
    try:
        import spacy
        nlp = spacy.load("ko_core_news_sm")
    except Exception:
        print("  ⚠ ko_core_news_sm 없음 → 영어 모델 시도")
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            print("  ✗ SpaCy 모델 없음. 패턴 기반 스킵.")
            return []

    results = []
    for row in candidates:
        triple = _extract_pattern_triple(row.get("marked_text", ""), nlp)
        if triple:
            triple["id"] = row["id"]
            triple["gold_relation"] = row.get("relation", "UNKNOWN")
            triple["head_text"] = row.get("head", {}).get("text", "")
            triple["tail_text"] = row.get("tail", {}).get("text", "")
            results.append(triple)

    print(f"  Pattern Open IE: {len(results)}/{len(candidates)}개 트리플 추출")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# B. LLM-based Open IE (Claude Batches API)
# ══════════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """당신은 한국어 정보 추출 전문가입니다.
주어진 문장에서 개체 간의 관계를 자유로운 형식으로 추출하세요.

출력 형식 (JSON):
{
  "triples": [
    {"head": "엔티티1", "relation": "관계 서술어", "tail": "엔티티2"}
  ],
  "oia_relation": "가장 적합한 OIA 관계 레이블 (아래 목록 중 하나)",
  "confidence": 0.0~1.0
}

OIA 관계 목록:
HAS_CONTACT_EMAIL, HAS_CONTACT_PHONE, HAS_DEADLINE, HAS_FEE,
MENTIONS_EXAM_LEVEL, NO_RELATION, REFERENCES_ATTACHMENT,
REFERENCES_EXTERNAL_RESOURCE, announced_by, mentions,
requires_document, requires_qualification"""

def build_user_prompt(row: dict) -> str:
    marked = row.get("marked_text", "")
    sentence = row.get("sentence", "")
    head_text = row.get("head", {}).get("text", "")
    tail_text = row.get("tail", {}).get("text", "")
    head_type = row.get("head", {}).get("type", "")
    tail_type = row.get("tail", {}).get("type", "")
    return (
        f"문장: {sentence}\n"
        f"마킹된 문장: {marked}\n"
        f"E1({head_type}): {head_text}\n"
        f"E2({tail_type}): {tail_text}\n\n"
        "위 문장에서 E1과 E2 사이의 관계를 추출하세요."
    )

def run_llm_open_ie_batch(
    candidates: list[dict],
    api_key: str | None = None,
    model: str = "claude-haiku-4-5",
    max_items: int = 500,
    output_cache: Path = REPORT_DIR / "llm_batch_results.json",
) -> list[dict]:
    """
    Anthropic Batches API로 LLM 기반 Open IE 수행.
    이미 결과가 캐시돼 있으면 재사용.
    """
    # 캐시 확인
    if output_cache.exists():
        print(f"  LLM Open IE 캐시 로드: {output_cache}")
        return json.loads(output_cache.read_text())

    try:
        import anthropic
    except ImportError:
        print("  ✗ anthropic 패키지 없음. pip install anthropic 실행 필요.")
        return []

    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        print("  ✗ ANTHROPIC_API_KEY 환경변수가 없습니다.")
        print("    export ANTHROPIC_API_KEY=sk-ant-... 후 재실행하세요.")
        return []

    client = anthropic.Anthropic(api_key=key)

    # 배치 요청 준비 (최대 max_items개)
    subset = candidates[:max_items]
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request as BatchRequest

    requests = []
    for row in subset:
        requests.append(BatchRequest(
            custom_id=row["id"],
            params=MessageCreateParamsNonStreaming(
                model=model,
                max_tokens=512,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": build_user_prompt(row)}],
            )
        ))

    print(f"  배치 생성 중 ({len(requests)}개 요청, 모델={model})...")
    batch = client.messages.batches.create(requests=requests)
    print(f"  배치 ID: {batch.id} | 상태: {batch.processing_status}")

    # 완료 대기 (polling)
    poll_interval = 15
    while batch.processing_status != "ended":
        time.sleep(poll_interval)
        batch = client.messages.batches.retrieve(batch.id)
        done = batch.request_counts.succeeded + batch.request_counts.errored
        total = len(requests)
        print(f"  처리 중... {done}/{total} | {batch.processing_status}")

    print(f"  배치 완료 ✓ | 성공={batch.request_counts.succeeded} 실패={batch.request_counts.errored}")

    # 결과 수집
    id_to_row = {r["id"]: r for r in candidates}
    results = []
    for result in client.messages.batches.results(batch.id):
        if result.result.type != "succeeded":
            continue
        msg = result.result.message
        text = next((b.text for b in msg.content if b.type == "text"), "")
        # JSON 파싱
        try:
            # 마크다운 코드블록 제거
            clean = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
            parsed = json.loads(clean)
        except json.JSONDecodeError:
            parsed = {"triples": [], "oia_relation": "NO_RELATION", "confidence": 0.0}

        row = id_to_row.get(result.custom_id, {})
        results.append({
            "id": result.custom_id,
            "gold_relation": row.get("relation", "UNKNOWN"),
            "llm_oia_relation": parsed.get("oia_relation", "NO_RELATION"),
            "llm_triples": parsed.get("triples", []),
            "llm_confidence": parsed.get("confidence", 0.0),
            "head_text": row.get("head", {}).get("text", ""),
            "tail_text": row.get("tail", {}).get("text", ""),
            "marked_text": row.get("marked_text", ""),
        })

    # 캐시 저장
    output_cache.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"  결과 저장: {output_cache}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 평가 및 비교
# ══════════════════════════════════════════════════════════════════════════════
def _map_pattern_to_oia(triple: dict, row: dict) -> str:
    """
    SpaCy 트리플의 predicate를 OIA 온톨로지에 규칙 기반 매핑.
    (간단한 키워드 매칭)
    """
    pred = triple.get("predicate", "").lower()
    tail_type = row.get("tail", {}).get("type", "")

    mapping = {
        ("email", "Email"): "HAS_CONTACT_EMAIL",
        ("phone", "Phone"): "HAS_CONTACT_PHONE",
        ("fee", "Fee"):     "HAS_FEE",
        ("deadline",):      "HAS_DEADLINE",
        ("topik", "level"): "MENTIONS_EXAM_LEVEL",
        ("document",):      "requires_document",
        ("qualification",): "requires_qualification",
    }
    for keywords, relation in mapping.items():
        if any(k in pred or k in tail_type.lower() for k in keywords):
            return relation

    return "NO_RELATION"

def evaluate_llm_results(llm_results: list[dict]) -> dict:
    """LLM Open IE의 OIA 매핑 F1 측정"""
    if not llm_results:
        return {}
    y_gold = [r["gold_relation"] for r in llm_results]
    y_pred = [r["llm_oia_relation"] for r in llm_results]
    macro_f1 = f1_score(y_gold, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_gold, y_pred, average="micro", zero_division=0)
    report = classification_report(y_gold, y_pred, zero_division=0)
    return {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "classification_report": report,
        "n_samples": len(llm_results),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 메인 실행
# ══════════════════════════════════════════════════════════════════════════════
def run_open_ie(api_key: str | None = None, max_llm: int = 500) -> tuple[float, float]:
    """
    두 방식 Open IE 실행 및 비교.
    Returns: (pattern_coverage, llm_macro_f1)
    """
    print("\n" + "="*60)
    print("  Step 2. Open IE — Pattern-based vs LLM-based")
    print("="*60)

    # 데이터 로드
    candidates = load_jsonl(CANDIDATES_PATH) if CANDIDATES_PATH.exists() else []
    train_rows  = load_jsonl(TRAIN_PATH)     if TRAIN_PATH.exists()     else []
    gold_df     = load_gold_df()
    corpus      = candidates + train_rows
    print(f"  Unlabeled pool: {len(candidates)}개 | Train: {len(train_rows)}개 | Gold: {len(gold_df)}개")

    # ── A. Pattern-based ──────────────────────────────────────────────────────
    print("\n▶ A. Pattern-based Open IE (SpaCy 의존구문)")
    pattern_triples = run_pattern_open_ie(corpus)
    pattern_coverage = len(pattern_triples) / max(len(corpus), 1)
    print(f"  추출 커버리지: {pattern_coverage:.1%}")

    # 대표 예시 출력
    if pattern_triples:
        print("  예시 트리플 (5개):")
        for t in pattern_triples[:5]:
            print(f"    ({t['head_text']}) --[{t['predicate']}]--> ({t['tail_text']}) | gold={t['gold_relation']}")

    # ── B. LLM-based ──────────────────────────────────────────────────────────
    print(f"\n▶ B. LLM-based Open IE (Claude API, max={max_llm}개)")
    llm_results = run_llm_open_ie_batch(
        corpus,
        api_key=api_key,
        max_items=max_llm,
    )
    llm_macro_f1 = 0.0
    if llm_results:
        metrics = evaluate_llm_results(llm_results)
        llm_macro_f1 = metrics.get("macro_f1", 0.0)
        print(f"\n  LLM Open IE → OIA 매핑 성능:")
        print(f"    Macro F1 : {metrics['macro_f1']:.4f}")
        print(f"    Micro F1 : {metrics['micro_f1']:.4f}")
        print(f"    샘플 수  : {metrics['n_samples']}")
        print("\n" + metrics["classification_report"])

        # 자유형 트리플 예시
        print("  자유형 트리플 예시 (5개):")
        for r in llm_results[:5]:
            triples_str = "; ".join(
                f"({t['head']}, {t['relation']}, {t['tail']})"
                for t in r.get("llm_triples", [])[:2]
            )
            print(f"    {triples_str}")
            print(f"      → LLM OIA: {r['llm_oia_relation']} | Gold: {r['gold_relation']}")

    # ── 비교 요약 ─────────────────────────────────────────────────────────────
    print("\n" + "-"*60)
    print("  [Open IE 방법론 비교]")
    print(f"  Pattern-based  추출 커버리지 : {pattern_coverage:.1%}")
    print(f"  LLM-based      OIA 매핑 F1   : {llm_macro_f1:.4f}")
    print("  주의: Pattern-based는 무라벨 추출 (F1 측정 불가)")
    print(f"        LLM-based는 자유 추출 후 온톨로지 매핑으로 F1 측정")

    # 결과 저장
    summary = {
        "pattern_coverage": pattern_coverage,
        "pattern_triple_count": len(pattern_triples),
        "llm_macro_f1": llm_macro_f1,
        "llm_sample_count": len(llm_results),
    }
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return pattern_coverage, llm_macro_f1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Open IE: Pattern-based vs LLM-based")
    parser.add_argument("--api-key", default=None,
                        help="Anthropic API key (없으면 ANTHROPIC_API_KEY env 사용)")
    parser.add_argument("--max-llm", type=int, default=500,
                        help="LLM 배치 처리 최대 개수 (기본 500)")
    parser.add_argument("--pattern-only", action="store_true",
                        help="LLM 없이 패턴 방식만 실행")
    args = parser.parse_args()

    if args.pattern_only:
        candidates = load_jsonl(CANDIDATES_PATH) if CANDIDATES_PATH.exists() else []
        train_rows = load_jsonl(TRAIN_PATH) if TRAIN_PATH.exists() else []
        corpus = candidates + train_rows
        run_pattern_open_ie(corpus)
    else:
        run_open_ie(api_key=args.api_key, max_llm=args.max_llm)
