"""
Step 3-B: Semi-supervised Relation Extraction (DIPRE & Snowball)
─────────────────────────────────────────────────────────────────
기존 문제:
  1. gold seeds를 이미 레이블된 train.jsonl에 적용 → bootstrapping 의미 없음
  2. 금/은 모두 개체 사이 패턴이 비어 있어 F1 ≈ 0
  3. 관계 라벨 정규화 없음 (announced_by vs ANNOUNCED_BY)

수정 사항:
  1. 패턴 풀 = corpus_unlabeled.jsonl (step0이 생성한 정제 비레이블 코퍼스)
  2. 빈 패턴 시 개체 타입 쌍을 "가상 패턴"으로 사용 (OIA 구조 데이터 특성 반영)
  3. 관계 라벨 정규화 (gold/corpus 모두)
  4. Snowball = 텍스트 패턴 AND 타입 매칭 동시 충족
"""

import re
import ast
import numpy as np
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import f1_score

from step1_data_loader import load_gold_standard

# ── 관계 라벨 정규화 ──────────────────────────────────
NORMALIZE_REL = {
    "requires_document":      "REQUIRES_DOCUMENT",
    "has_deadline":            "HAS_DEADLINE",
    "announced_by":            "ANNOUNCED_BY",
    "mentions":                "MENTIONS",
    "requires_qualification":  "REQUIRES_QUALIFICATION",
}

def norm_rel(r: str) -> str:
    return NORMALIZE_REL.get(str(r).strip(), str(r).strip())


# ── 텍스트 정제 ────────────────────────────────────────
def _clean_text(text: str) -> str:
    text = re.sub(r'<[^>]+>', ' ', str(text))
    text = re.sub(r'[^\w\s가-힣]', ' ', text)
    return ' '.join(text.split()).strip()


# ── 개체 사이 패턴 추출 ────────────────────────────────
def extract_text_pattern(marked_text: str, max_words: int = 5):
    """E1→E2 또는 E2→E1 방향에서 사이 트리거 단어 추출 (최대 max_words개)."""
    raw = str(marked_text)
    m = re.search(r'\[/E1\](.*?)\[E2\]', raw, flags=re.DOTALL)
    if not m:
        m = re.search(r'\[/E2\](.*?)\[E1\]', raw, flags=re.DOTALL)
    if not m:
        return None
    cleaned = _clean_text(m.group(1))
    if len(cleaned) < 2:
        return None
    words = cleaned.split()[:max_words]
    return ' '.join(words)


def _extract_context_before_e1(marked_text: str, n_words: int = 3):
    """E1 직전 n개 단어 (패턴 보조 신호)"""
    m = re.search(r'\[E1\]', str(marked_text))
    if not m:
        return None
    before = _clean_text(str(marked_text)[:m.start()])
    words = before.split()
    if not words:
        return None
    return ' '.join(words[-n_words:])


# ── 타입 가상 패턴 ──────────────────────────────────────
TYPE_PREFIX = "__TYPE__"

def make_type_pattern(head_type: str, tail_type: str) -> str:
    """개체 타입 쌍을 가상 패턴으로 인코딩."""
    return f"{TYPE_PREFIX}{head_type}+{tail_type}"


def is_type_pattern(p: str) -> bool:
    return p.startswith(TYPE_PREFIX)


# ── Corpus 로더 ────────────────────────────────────────
def load_corpus_unlabeled(path: str = "data/re_fixed_v6/corpus_unlabeled.jsonl") -> list:
    p = Path(path)
    if not p.exists():
        print(f"[WARN] {path} 없음. step0_rebuild_corpus.py를 먼저 실행하세요.")
        return []
    rows = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    print(f"✅ corpus_unlabeled 로드: {len(rows)}건")
    return rows


def _parse_type(ent) -> str:
    if isinstance(ent, dict):
        return ent.get("type", "UNKNOWN")
    if isinstance(ent, str):
        try:
            d = ast.literal_eval(ent)
            if isinstance(d, dict):
                return d.get("type", "UNKNOWN")
        except Exception:
            pass
    return "UNKNOWN"


# ── 핵심 함수 ──────────────────────────────────────────
def run_dipre_and_snowball(n_seeds: int = 10):
    print(f"--- 🚀 Step 3-B. Semi-supervised RE (DIPRE & Snowball, seed={n_seeds}) ---")
    print("    패턴 풀: corpus_unlabeled.jsonl  |  빈 패턴 fallback: 개체 타입 쌍\n")

    gold_df = load_gold_standard()
    corpus  = load_corpus_unlabeled()

    if gold_df.empty or not corpus:
        print("데이터를 불러올 수 없습니다.")
        return 0.0, 0.0

    # gold 관계 정규화
    gold_df["final_relation"] = gold_df["final_relation"].apply(norm_rel)

    relations = gold_df["final_relation"].unique()
    print(f"\n총 {len(relations)}개 관계 | Corpus Pool {len(corpus)}건\n")

    dipre_f1s    = []
    snowball_f1s = []

    for target_relation in relations:
        seeds = gold_df[gold_df["final_relation"] == target_relation].head(n_seeds)
        if len(seeds) == 0:
            continue

        # ── 패턴 추출 ─────────────────────────────────────
        text_patterns = []   # 텍스트 기반 (DIPRE core)
        type_patterns = []   # 타입 기반 (보조 + Snowball)

        for _, row in seeds.iterrows():
            p = extract_text_pattern(row.get("marked_text", ""))
            if p:
                text_patterns.append(p)
            ctx = _extract_context_before_e1(row.get("marked_text", ""))
            if ctx and len(ctx) >= 2:
                text_patterns.append(ctx)

            h_type = row.get("head_type") or _parse_type(row.get("head", {}))
            t_type = row.get("tail_type") or _parse_type(row.get("tail", {}))
            if h_type and t_type:
                type_patterns.append(make_type_pattern(h_type, t_type))

        text_patterns = list({p for p in text_patterns if len(p) >= 2})
        type_patterns = list(set(type_patterns))

        # ── 패턴 품질 필터 ──────────────────────────────────
        # 엔티티 고유값 (날짜, 숫자, 영문) 위주 패턴은 제너럴라이즈 불가 → 제거
        def _is_generalizable(pat: str) -> bool:
            words = pat.split()
            # 순수 숫자/날짜 토큰이 대부분이면 제거
            numeric = sum(1 for w in words if re.match(r'^\d[\d.\-:/]*$', w))
            if numeric / max(len(words), 1) > 0.5:
                return False
            # 전부 영문(한국어 없음)이면 노이즈 가능성 높음
            korean = sum(1 for c in pat if '가' <= c <= '힣')
            if len(words) >= 2 and korean == 0:
                return False
            return True

        quality_patterns = [p for p in text_patterns if _is_generalizable(p)]

        # 품질 패턴이 corpus에 충분히 매칭되는지 확인
        # 최소 MIN_HITS건 매칭돼야 신뢰할 수 있는 패턴으로 간주
        MIN_HITS = 3
        if quality_patterns:
            corpus_hits = sum(
                1 for row in corpus
                if any(pat in str(row.get("marked_text", "")) for pat in quality_patterns)
            )
        else:
            corpus_hits = 0

        use_type_fallback = (len(quality_patterns) == 0) or (corpus_hits < MIN_HITS)

        seed_head_types = set(
            (row.get("head_type") or _parse_type(row.get("head", {})))
            for _, row in seeds.iterrows()
        )
        seed_tail_types = set(
            (row.get("tail_type") or _parse_type(row.get("tail", {})))
            for _, row in seeds.iterrows()
        )

        # ── 예측 ──────────────────────────────────────────
        dipre_preds    = []
        snowball_preds = []
        actuals        = []

        for row in corpus:
            marked     = str(row.get("marked_text", ""))
            actual_rel = norm_rel(row.get("true_relation", row.get("relation", "")))

            h_type = _parse_type(row.get("head", {}))
            t_type = _parse_type(row.get("tail", {}))

            matched_type_classic = (
                h_type in seed_head_types and t_type in seed_tail_types
            )

            if use_type_fallback:
                # 패턴 없거나 노이즈 패턴만 있는 경우: 타입으로 DIPRE/Snowball 판단
                dipre_match    = matched_type_classic
                snowball_match = matched_type_classic
            else:
                # 품질 텍스트 패턴 있는 경우
                matched_text   = any(pat in marked for pat in quality_patterns)
                # DIPRE: 텍스트 패턴 기반 (corpus에 실제 매칭 없으면 이미 fallback됨)
                dipre_match    = matched_text
                # Snowball: 텍스트 패턴 AND 타입 (노이즈 제거)
                snowball_match = matched_text and matched_type_classic

            dipre_preds.append(target_relation if dipre_match    else "OTHER")
            snowball_preds.append(target_relation if snowball_match else "OTHER")
            actuals.append(actual_rel if actual_rel == target_relation else "OTHER")

        dipre_f1    = f1_score(actuals, dipre_preds,
                               pos_label=target_relation, average="binary", zero_division=0)
        snowball_f1 = f1_score(actuals, snowball_preds,
                               pos_label=target_relation, average="binary", zero_division=0)

        dipre_f1s.append(dipre_f1)
        snowball_f1s.append(snowball_f1)

        flag = " [TYPE-ONLY]" if use_type_fallback else f" [q={len(quality_patterns)}]"
        print(
            f"  [{target_relation:30s}] "
            f"DIPRE={dipre_f1:.4f} | Snowball={snowball_f1:.4f}"
            f" | txt={len(text_patterns)} q={len(quality_patterns)} typ={len(type_patterns)}{flag}"
        )

    macro_dipre_f1    = float(np.mean(dipre_f1s))    if dipre_f1s    else 0.0
    macro_snowball_f1 = float(np.mean(snowball_f1s)) if snowball_f1s else 0.0

    print(f"\n▶ DIPRE    Macro F1 (관계 평균): {macro_dipre_f1:.4f}")
    print(f"▶ Snowball Macro F1 (관계 평균): {macro_snowball_f1:.4f}")

    # ── HAS_FEE 상세 예시 ────────────────────────────────
    target_example = "HAS_FEE"
    if target_example in gold_df["final_relation"].values:
        print(f"\n▶ [{target_example}] DIPRE 증식 예시")
        seeds_ex = gold_df[gold_df["final_relation"] == target_example].head(n_seeds)

        pats_ex = defaultdict(int)
        for _, row in seeds_ex.iterrows():
            p = extract_text_pattern(row.get("marked_text", ""))
            if p:
                pats_ex[p] += 1
            ctx = _extract_context_before_e1(row.get("marked_text", ""))
            if ctx:
                pats_ex[ctx] += 1

        sorted_pats = sorted(pats_ex.items(), key=lambda x: x[1], reverse=True)[:5]
        if sorted_pats:
            print(f"  Top 5 텍스트 패턴: {sorted_pats}")
        else:
            print("  텍스트 패턴 없음 → 타입 패턴 fallback 사용")

        extracted = set()
        for row in corpus:
            marked = str(row.get("marked_text", ""))
            for pat, _ in sorted_pats:
                if pat and pat in marked and "[E1]" in marked and "[E2]" in marked:
                    h = re.search(r"\[E1\](.*?)\[/E1\]", marked)
                    t = re.search(r"\[E2\](.*?)\[/E2\]", marked)
                    if h and t:
                        extracted.add((h.group(1).strip(), t.group(1).strip()))

        print(f"  DIPRE 추출 튜플 수: {len(extracted)}개")
        for tup in list(extracted)[:5]:
            print(f"   - {tup}")

        filtered = {
            (h, t) for h, t in extracted
            if any(kw in t for kw in ("원", "USD", "00", "$", "만", "천"))
        }
        print(f"  Snowball 필터 후 튜플 수: {len(filtered)}개")
        if extracted:
            print(f"  노이즈 제거율: {(1 - len(filtered)/len(extracted))*100:.1f}%")

    return macro_dipre_f1, macro_snowball_f1


if __name__ == "__main__":
    run_dipre_and_snowball()
