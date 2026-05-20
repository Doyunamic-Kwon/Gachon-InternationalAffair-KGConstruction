"""
Step 0: Silver Corpus Rebuild
─────────────────────────────
문제: candidates.jsonl의 784건(45%)이 개체 사이 패턴 없음 (인접 개체),
     838건은 full-page blob에 마커가 박혀 있어 패턴이 너무 길고 노이즈가 많음.
     → DIPRE/Snowball이 패턴을 추출할 수가 없음.

해결:
  1. 리치(≥10자) 케이스: E1~E2 마커 주변 ±200자 윈도우 추출
  2. 빈/짧은 케이스: 관계별 한국어 템플릿 문장 생성
  3. (선택) OPEN_AI_KEY 환경변수가 있으면 템플릿 대신 OpenAI로 다양한 문장 생성
  4. 관계 라벨 정규화 (requires_document → REQUIRES_DOCUMENT 등)
  5. 출력: data/re_fixed_v6/corpus_unlabeled.jsonl  ← DIPRE pool
           data/re_fixed_v6/corpus_clean.jsonl      ← 라벨 포함 clean corpus
"""

import os
import re
import json
import time
import random
import argparse
from pathlib import Path
from collections import Counter


# ─────────────────────────────────────────────
# 1. Relation 정규화 매핑
# ─────────────────────────────────────────────
NORMALIZE = {
    "requires_document": "REQUIRES_DOCUMENT",
    "has_deadline":       "HAS_DEADLINE",
    "announced_by":       "ANNOUNCED_BY",
    "mentions":           "MENTIONS",
    "requires_qualification": "REQUIRES_QUALIFICATION",
}


def normalize_relation(rel: str) -> str:
    return NORMALIZE.get(rel, rel)


# ─────────────────────────────────────────────
# 2. 패턴 추출 (마커 사이 텍스트)
# ─────────────────────────────────────────────
def between_entities(marked_text: str) -> str:
    m = re.search(r'\[/E1\](.*?)\[E2\]', marked_text, re.DOTALL)
    if not m:
        m = re.search(r'\[/E2\](.*?)\[E1\]', marked_text, re.DOTALL)
    return m.group(1).strip() if m else ""


# ─────────────────────────────────────────────
# 3. 풍부한 케이스: 윈도우 추출
# ─────────────────────────────────────────────
def extract_window(marked_text: str, window: int = 200) -> str:
    """E1 시작 전 window자 ~ E2 끝 후 window자만 남김"""
    # 첫 번째 E1/E2 마커 위치 찾기
    e1_start = marked_text.find('[E1]')
    e2_end_m = re.search(r'\[/E2\]', marked_text)
    if e1_start == -1 or not e2_end_m:
        # E2→E1 순서
        e1_start = marked_text.find('[E2]')
        e2_end_m = re.search(r'\[/E1\]', marked_text)
    if e1_start == -1 or not e2_end_m:
        return marked_text  # fallback

    e2_end = e2_end_m.end()
    start = max(0, e1_start - window)
    end   = min(len(marked_text), e2_end + window)
    snippet = marked_text[start:end].strip()
    return snippet


# ─────────────────────────────────────────────
# 4. 빈 케이스: 템플릿 문장 생성
# ─────────────────────────────────────────────
TEMPLATES = {
    # 모든 템플릿은 {head}(E1)가 {tail}(E2)보다 먼저 등장하도록 작성
    "HAS_CONTACT_EMAIL": [
        "{head}의 문의 이메일 주소는 {tail}입니다.",
        "{head}에 대한 문의는 이메일 {tail}으로 연락 바랍니다.",
        "{head} 담당자 이메일 주소: {tail}",
    ],
    "HAS_CONTACT_PHONE": [
        "{head}의 전화번호는 {tail}입니다.",
        "{head}에 대한 문의 전화는 {tail}입니다.",
        "{head} 담당자 연락처: {tail}",
    ],
    "HAS_FEE": [
        "{head}의 수수료(전형료)는 {tail}입니다.",
        "{head} 신청 시 발생하는 비용은 {tail}입니다.",
        "{head} 등록금은 {tail}이 부과됩니다.",
    ],
    "HAS_DEADLINE": [
        "{head}의 신청 마감일은 {tail}입니다.",
        "{head} 접수 기간은 {tail}까지입니다.",
        "{head} 신청서 제출 기한은 {tail}입니다.",
    ],
    "REFERENCES_EXTERNAL_RESOURCE": [
        "{head}에서는 관련 정보를 {tail}에서 확인하실 수 있습니다.",
        "{head} 관련 자세한 내용은 외부 링크 {tail}을 참고하세요.",
        "{head}의 참고 자료는 {tail}입니다.",
    ],
    "REQUIRES_DOCUMENT": [
        "{head}에 지원하려면 필수 서류로 {tail}이 필요합니다.",
        "{head} 신청 시 제출해야 하는 서류는 {tail}입니다.",
        "{head} 접수에는 {tail} 등의 서류가 요구됩니다.",
    ],
    "MENTIONS": [
        "{head}에서는 {tail}에 대해 안내하고 있습니다.",
        "{head}는 {tail}에 관련된 내용을 포함하고 있습니다.",
        "{head} 공지에는 {tail}에 관한 정보가 있습니다.",
    ],
    "ANNOUNCED_BY": [
        "{head}는 {tail}에서 공지한 사항입니다.",
        "{head}의 공지 부서는 {tail}입니다.",
        "{head}에 대한 안내는 {tail}에서 발표하였습니다.",
    ],
    "MENTIONS_EXAM_LEVEL": [
        "{head}에서는 {tail} 시험을 요구합니다.",
        "{head} 지원을 위해 필요한 시험은 {tail}입니다.",
        "{head}에 명시된 시험 등급은 {tail}입니다.",
    ],
    "REFERENCES_ATTACHMENT": [
        "{head}의 첨부파일로 {tail}이 있습니다.",
        "{head}에 관련된 첨부 문서는 {tail}입니다.",
        "{head}에는 관련 자료로 {tail}이 첨부되어 있습니다.",
    ],
    "REQUIRES_QUALIFICATION": [
        "{head}의 지원 자격은 {tail}입니다.",
        "{head}에 지원하려면 {tail} 조건이 필요합니다.",
        "{head}에서 요구하는 자격 요건은 {tail}입니다.",
    ],
    "NO_RELATION": [
        "{head}에 관한 정보에서 {tail}이 언급되어 있습니다.",
        "{head} 페이지에 {tail} 관련 내용이 있습니다.",
    ],
}

DEFAULT_TEMPLATE = [
    "{head}와 {tail}의 관계에 대한 정보입니다.",
]


def apply_template(relation: str, head_text: str, tail_text: str) -> str:
    templates = TEMPLATES.get(relation, DEFAULT_TEMPLATE)
    tmpl = random.choice(templates)
    sentence = tmpl.format(head=head_text, tail=tail_text)

    # ── 마커 삽입: {head}와 {tail} 자리를 원본 템플릿에서 직접 찾기 ──
    # 1단계: 템플릿의 {head}, {tail} 자리 인덱스를 추적
    placeholder_head = "\x00HEAD\x00"
    placeholder_tail = "\x00TAIL\x00"
    marked_template = tmpl.replace("{head}", placeholder_head, 1)
    marked_template = marked_template.replace("{tail}", placeholder_tail, 1)
    marked_sentence = marked_template.format(**{}) if "{" not in marked_template else marked_template
    # format 없이 그대로 사용
    marked_sentence = marked_template
    marked_sentence = marked_sentence.replace(placeholder_head, f"[E1] {head_text} [/E1]")
    marked_sentence = marked_sentence.replace(placeholder_tail, f"[E2] {tail_text} [/E2]")

    return sentence, marked_sentence


# ─────────────────────────────────────────────
# 5. OpenAI batch 생성 (옵션)
# ─────────────────────────────────────────────
def generate_with_openai(empty_rows: list[dict], api_key: str, max_items: int = 500) -> dict:
    """
    OpenAI gpt-4o-mini로 빈 패턴 케이스에 자연스러운 한국어 문장 생성.
    반환: {row_id: {"sentence": ..., "marked_text": ...}}
    """
    try:
        from openai import OpenAI
    except ImportError:
        print("  [WARN] openai 미설치. pip install openai")
        return {}

    client = OpenAI(api_key=api_key)
    subset = empty_rows[:max_items]

    SYSTEM = (
        "당신은 한국 대학교 국제처(OIA) 웹사이트 문장 작성 전문가입니다. "
        "주어진 두 개체와 관계 정보를 바탕으로, "
        "OIA 웹사이트에 실제로 등장할 법한 자연스러운 한국어 문장을 1~2개 작성합니다. "
        "반드시 두 개체를 모두 포함하고, JSON 형식으로만 응답합니다: "
        '{"sentence": "...", "marked_text": "[E1] {head} [/E1] ... [E2] {tail} [/E2]"}'
    )

    results = {}
    total = len(subset)
    print(f"  OpenAI 문장 생성 시작: {total}건 (gpt-4o-mini)")

    for idx, row in enumerate(subset):
        if idx % 50 == 0:
            print(f"    진행 중: {idx}/{total}")

        head_text = row.get("head", {}).get("text", "")
        tail_text = row.get("tail", {}).get("text", "")
        head_type = row.get("head", {}).get("type", "")
        tail_type = row.get("tail", {}).get("type", "")
        relation  = normalize_relation(row.get("relation", ""))

        user_msg = (
            f"개체1: {head_text} (타입: {head_type})\n"
            f"개체2: {tail_text} (타입: {tail_type})\n"
            f"관계: {relation}\n\n"
            f"위 두 개체를 모두 포함하는 자연스러운 한국어 문장을 작성해주세요. "
            f"marked_text에서는 개체1을 [E1] ... [/E1], 개체2를 [E2] ... [/E2]로 감싸세요."
        )

        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0.7,
                max_tokens=200,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
            results[row["id"]] = {
                "sentence":    data.get("sentence", ""),
                "marked_text": data.get("marked_text", ""),
            }
        except Exception as e:
            print(f"    [WARN] row {row['id']}: {e}")
            results[row["id"]] = None

        # 간단한 rate-limit 방지
        if idx % 20 == 19:
            time.sleep(0.5)

    success = sum(1 for v in results.values() if v)
    print(f"  OpenAI 생성 완료: {success}/{total}건 성공")
    return results


# ─────────────────────────────────────────────
# 6. 메인 빌드 함수
# ─────────────────────────────────────────────
def rebuild_corpus(use_openai: bool = False, openai_max: int = 500):
    print("=== Step 0. Silver Corpus Rebuild ===\n")

    in_path  = Path("data/re_fixed_v6/candidates.jsonl")
    out_clean     = Path("data/re_fixed_v6/corpus_clean.jsonl")
    out_unlabeled = Path("data/re_fixed_v6/corpus_unlabeled.jsonl")

    with open(in_path, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]

    # ── 분류 ──────────────────────────────────
    empty_rows = []
    rich_rows  = []

    for r in rows:
        marked = r.get("marked_text", "")
        between = between_entities(marked)
        if len(between) < 2:
            empty_rows.append(r)
        else:
            rich_rows.append(r)

    print(f"전체: {len(rows)}건  |  빈 패턴: {len(empty_rows)}건  |  리치: {len(rich_rows)}건")

    # ── OpenAI 생성 (옵션) ─────────────────────
    openai_results = {}
    if use_openai:
        api_key = os.getenv("OPEN_AI_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("[WARN] OPEN_AI_KEY 환경변수 없음. 템플릿으로 대체합니다.")
        else:
            openai_results = generate_with_openai(empty_rows, api_key, max_items=openai_max)

    # ── 빌드 ──────────────────────────────────
    clean_rows    = []
    stat_empty    = 0
    stat_openai   = 0
    stat_template = 0
    stat_window   = 0

    for r in rows:
        new_r    = dict(r)
        relation = normalize_relation(r.get("relation", "NO_RELATION"))
        new_r["relation"] = relation

        head_text = (r.get("head") or {}).get("text", "")
        tail_text = (r.get("tail") or {}).get("text", "")

        marked  = r.get("marked_text", "")
        between = between_entities(marked)

        if len(between) < 2:
            stat_empty += 1
            # 1순위: OpenAI 생성 결과
            oi = openai_results.get(r["id"])
            if oi and oi.get("sentence") and oi.get("marked_text"):
                new_r["sentence"]    = oi["sentence"]
                new_r["marked_text"] = oi["marked_text"]
                new_r["context_src"] = "openai"
                stat_openai += 1
            else:
                # 2순위: 템플릿
                sent, new_marked = apply_template(relation, head_text, tail_text)
                new_r["sentence"]    = sent
                new_r["marked_text"] = new_marked
                new_r["context_src"] = "template"
                stat_template += 1
        else:
            # 풍부한 케이스: 윈도우 추출
            new_r["marked_text"] = extract_window(marked, window=200)
            # sentence도 마커를 제거한 clean 버전으로
            new_r["sentence"] = re.sub(r'\[/?E[12]\]', '', new_r["marked_text"]).strip()
            new_r["context_src"] = "window"
            stat_window += 1

        clean_rows.append(new_r)

    # ── 통계 ──────────────────────────────────
    print(f"\n  컨텍스트 소스:")
    print(f"    윈도우 추출: {stat_window}건")
    print(f"    OpenAI 생성: {stat_openai}건")
    print(f"    템플릿 생성: {stat_template}건")

    # 관계 분포 (정규화 후)
    rel_dist = Counter(r["relation"] for r in clean_rows)
    print(f"\n  정규화 후 관계 분포:")
    for rel, cnt in rel_dist.most_common():
        print(f"    {rel:35s}: {cnt}")

    # ── 저장 ──────────────────────────────────
    # corpus_clean: relation 포함
    with open(out_clean, "w", encoding="utf-8") as f:
        for r in clean_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n✅ corpus_clean.jsonl 저장: {len(clean_rows)}건 → {out_clean}")

    # corpus_unlabeled: relation을 'UNKNOWN'으로 마스킹 → DIPRE pool
    unlabeled_rows = []
    for r in clean_rows:
        ur = dict(r)
        ur["true_relation"] = r["relation"]   # 평가용 보관
        ur["relation"]      = "UNKNOWN"
        unlabeled_rows.append(ur)

    with open(out_unlabeled, "w", encoding="utf-8") as f:
        for r in unlabeled_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"✅ corpus_unlabeled.jsonl 저장: {len(unlabeled_rows)}건 → {out_unlabeled}")

    # ── 패턴 품질 검증 ──────────────────────────
    print("\n  [패턴 품질 검증]")
    between_lens = []
    for r in clean_rows:
        b = between_entities(r["marked_text"])
        between_lens.append(len(b))

    empty_after = sum(1 for l in between_lens if l < 2)
    short_after = sum(1 for l in between_lens if 2 <= l < 10)
    rich_after  = sum(1 for l in between_lens if l >= 10)
    print(f"  빈 패턴 (< 2자):   {empty_after}건 ({empty_after/len(clean_rows)*100:.1f}%)")
    print(f"  짧은 패턴 (2-9자): {short_after}건 ({short_after/len(clean_rows)*100:.1f}%)")
    print(f"  리치 패턴 (≥10자): {rich_after}건  ({rich_after/len(clean_rows)*100:.1f}%)")

    return clean_rows


# ─────────────────────────────────────────────
# 7. CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Silver corpus rebuild for DIPRE/Snowball")
    parser.add_argument("--use-openai",  action="store_true",
                        help="OpenAI API로 빈 패턴 문장 생성 (OPEN_AI_KEY 환경변수 필요)")
    parser.add_argument("--openai-max",  type=int, default=500,
                        help="OpenAI 생성 최대 건수 (기본 500)")
    args = parser.parse_args()

    random.seed(42)
    rebuild_corpus(use_openai=args.use_openai, openai_max=args.openai_max)
