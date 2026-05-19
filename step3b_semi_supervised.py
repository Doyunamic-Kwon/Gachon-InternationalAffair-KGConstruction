import re
import ast
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import f1_score
from step1_data_loader import load_gold_standard, load_silver_standard


def _clean_text(text):
    """HTML 태그 제거 + 공백 정규화"""
    text = re.sub(r'<[^>]+>', ' ', str(text))
    text = re.sub(r'[^\w\s가-힣]', ' ', text)
    return ' '.join(text.split()).strip()


def extract_pattern(marked_text):
    """
    E1/E2 사이 트리거 단어 추출 (HTML 정제 + 최대 5단어).
    E1-E2 인접(빈 패턴)이면 None 반환.
    """
    raw = str(marked_text)
    # E1→E2 방향
    m = re.search(r'\[/E1\](.*?)\[E2\]', raw, flags=re.DOTALL)
    if not m:
        # E2→E1 방향
        m = re.search(r'\[/E2\](.*?)\[E1\]', raw, flags=re.DOTALL)
    if not m:
        return None

    cleaned = _clean_text(m.group(1))
    if len(cleaned) < 2:
        return None

    # 최대 5개 단어만 유지 (긴 HTML 패턴 트리밍)
    words = cleaned.split()[:5]
    return ' '.join(words)


def _extract_context_before_e1(marked_text, n_words=3):
    """E1 직전 n개 단어 (패턴 보조 신호)"""
    raw = _clean_text(re.sub(r'\[/?E[12]\]', ' ', str(marked_text)))
    m = re.search(r'\[E1\]', str(marked_text))
    if not m:
        return None
    before = _clean_text(str(marked_text)[:m.start()])
    words = before.split()
    if not words:
        return None
    return ' '.join(words[-n_words:])


def _parse_silver_types(row):
    """silver_df head/tail 딕셔너리에서 타입 추출"""
    def parse(ent):
        if isinstance(ent, dict):
            return ent.get('type', 'UNKNOWN')
        if isinstance(ent, str):
            try:
                d = ast.literal_eval(ent)
                if isinstance(d, dict):
                    return d.get('type', 'UNKNOWN')
            except Exception:
                pass
        return 'UNKNOWN'
    return parse(row.get('head', {})), parse(row.get('tail', {}))


def run_dipre_and_snowball(n_seeds=10):
    print(f"--- 🚀 Step 3-B. Semi-supervised RE (DIPRE & Snowball, seed={n_seeds}) ---")

    gold_df   = load_gold_standard()
    silver_df = load_silver_standard()

    if gold_df.empty or silver_df.empty:
        print("데이터를 불러올 수 없습니다.")
        return 0.0, 0.0

    silver_valid = silver_df.dropna(subset=['relation'])
    silver_valid = silver_valid[
        silver_valid['relation'].apply(lambda x: bool(x) and x != 'NA')
    ].reset_index(drop=True)

    if silver_valid.empty:
        print("Silver 데이터에 유효한 relation이 없습니다.")
        return 0.0, 0.0

    relations = gold_df['final_relation'].unique()
    print(f"\n총 {len(relations)}개 관계 | Silver Pool {len(silver_valid)}건\n")

    dipre_f1s    = []
    snowball_f1s = []

    for target_relation in relations:
        seeds = gold_df[gold_df['final_relation'] == target_relation].head(n_seeds)
        if len(seeds) == 0:
            continue

        # 패턴 추출 (E1-E2 사이 트리거 + E1 직전 문맥)
        patterns = []
        for _, row in seeds.iterrows():
            p = extract_pattern(row.get('marked_text', ''))
            if p:
                patterns.append(p)
            ctx = _extract_context_before_e1(row.get('marked_text', ''))
            if ctx and len(ctx) >= 2:
                patterns.append(ctx)

        # 중복 제거 + 너무 짧은 패턴 제거
        patterns = list({p for p in patterns if len(p) >= 2})

        if not patterns:
            continue

        # 시드 개체 타입
        seed_head_types = set(seeds['head_type'].dropna().tolist())
        seed_tail_types = set(seeds['tail_type'].dropna().tolist())

        dipre_preds    = []
        snowball_preds = []
        actuals        = []

        for _, row in silver_valid.iterrows():
            marked     = str(row.get('marked_text', ''))
            actual_rel = row.get('relation', '')

            matched_pattern = any(pat in marked for pat in patterns)

            h_type, t_type = _parse_silver_types(row)
            matched_type   = (h_type in seed_head_types and t_type in seed_tail_types)

            dipre_preds.append(target_relation if matched_pattern else 'OTHER')
            snowball_preds.append(target_relation if (matched_pattern and matched_type) else 'OTHER')
            actuals.append(actual_rel if actual_rel == target_relation else 'OTHER')

        dipre_f1    = f1_score(actuals, dipre_preds,
                               pos_label=target_relation, average='binary', zero_division=0)
        snowball_f1 = f1_score(actuals, snowball_preds,
                               pos_label=target_relation, average='binary', zero_division=0)

        dipre_f1s.append(dipre_f1)
        snowball_f1s.append(snowball_f1)
        print(f"  [{target_relation}] DIPRE F1: {dipre_f1:.4f} | Snowball F1: {snowball_f1:.4f}"
              f" | patterns={len(patterns)}")

    macro_dipre_f1    = float(np.mean(dipre_f1s))    if dipre_f1s    else 0.0
    macro_snowball_f1 = float(np.mean(snowball_f1s)) if snowball_f1s else 0.0

    print(f"\n▶ DIPRE    Macro F1 (관계 평균): {macro_dipre_f1:.4f}")
    print(f"▶ Snowball Macro F1 (관계 평균): {macro_snowball_f1:.4f}")

    # ── HAS_FEE 관계 상세 예시 ────────────────────────
    target_example = 'HAS_FEE'
    if target_example in gold_df['final_relation'].values:
        print(f"\n▶ [{target_example}] DIPRE 증식 예시")
        seeds_ex = gold_df[gold_df['final_relation'] == target_example].head(n_seeds)
        pats_ex  = defaultdict(int)
        for _, row in seeds_ex.iterrows():
            p = extract_pattern(row.get('marked_text', ''))
            if p:
                pats_ex[p] += 1
            ctx = _extract_context_before_e1(row.get('marked_text', ''))
            if ctx:
                pats_ex[ctx] += 1

        sorted_pats = sorted(pats_ex.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"  Top 5 패턴: {sorted_pats}")

        extracted = set()
        for _, row in silver_valid.iterrows():
            marked = str(row.get('marked_text', ''))
            for pat, _ in sorted_pats:
                if pat and pat in marked and '[E1]' in marked and '[E2]' in marked:
                    h = re.search(r'\[E1\](.*?)\[/E1\]', marked)
                    t = re.search(r'\[E2\](.*?)\[/E2\]', marked)
                    if h and t:
                        extracted.add((h.group(1).strip(), t.group(1).strip()))
        print(f"  DIPRE 추출 튜플 수: {len(extracted)}개")
        for tup in list(extracted)[:5]:
            print(f"   - {tup}")

        filtered = {(h, t) for h, t in extracted
                    if any(kw in t for kw in ('원', 'USD', '00', '$', '만', '천'))}
        print(f"  Snowball 필터 후 튜플 수: {len(filtered)}개")
        if extracted:
            print(f"  노이즈 제거율: {(1 - len(filtered)/len(extracted))*100:.1f}%")

    return macro_dipre_f1, macro_snowball_f1


if __name__ == "__main__":
    run_dipre_and_snowball()
