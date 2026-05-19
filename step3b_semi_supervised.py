import re
import ast
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import f1_score
from step1_data_loader import load_gold_standard, load_silver_standard


def extract_pattern(marked_text):
    """[E1]...[/E1] 와 [E2]...[/E2] 사이의 패턴을 추출합니다."""
    match1 = re.search(r'\[/E1\](.*?)\[E2\]', str(marked_text), flags=re.DOTALL)
    if match1:
        return match1.group(1).strip()
    match2 = re.search(r'\[/E2\](.*?)\[E1\]', str(marked_text), flags=re.DOTALL)
    if match2:
        return match2.group(1).strip()
    return None


def _parse_silver_types(row):
    """silver_df의 head/tail 딕셔너리에서 타입을 추출합니다."""
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


def run_dipre_and_snowball():
    print("--- 🚀 Step 3-B. Semi-supervised RE (DIPRE & Snowball) ---")

    gold_df   = load_gold_standard()
    silver_df = load_silver_standard()

    if gold_df.empty or silver_df.empty:
        print("데이터를 불러올 수 없습니다.")
        return 0.0, 0.0

    # Silver에서 유효한 relation 행만 사용
    silver_valid = silver_df.dropna(subset=['relation'])
    silver_valid = silver_valid[
        silver_valid['relation'].apply(lambda x: bool(x) and x != 'NA')
    ].reset_index(drop=True)

    if silver_valid.empty:
        print("Silver 데이터에 유효한 relation이 없습니다.")
        return 0.0, 0.0

    relations = gold_df['final_relation'].unique()
    print(f"\n총 {len(relations)}개 관계에 대해 Seed 기반 부트스트래핑 평가")
    print(f"Silver Pool 크기: {len(silver_valid)}건\n")

    dipre_f1s    = []
    snowball_f1s = []

    # 관계별로 실제 Bootstrapping + 평가
    for target_relation in relations:
        seeds = gold_df[gold_df['final_relation'] == target_relation].head(5)
        if len(seeds) == 0:
            continue

        # 시드에서 패턴 추출
        patterns = []
        for _, row in seeds.iterrows():
            pat = extract_pattern(row.get('marked_text', ''))
            if pat and pat.strip():
                patterns.append(pat)

        if not patterns:
            continue

        # 시드 엔티티 타입 수집
        seed_head_types = set(seeds['head_type'].dropna().tolist())
        seed_tail_types = set(seeds['tail_type'].dropna().tolist())

        dipre_preds    = []
        snowball_preds = []
        actuals        = []

        for _, row in silver_valid.iterrows():
            marked     = str(row.get('marked_text', ''))
            actual_rel = row.get('relation', '')

            matched_pattern = any(pat in marked for pat in patterns)

            h_type, t_type  = _parse_silver_types(row)
            matched_type    = (h_type in seed_head_types and t_type in seed_tail_types)

            # DIPRE: 패턴 일치만으로 예측
            dipre_preds.append(target_relation if matched_pattern else 'OTHER')
            # Snowball: 패턴 + 엔티티 타입 모두 일치
            snowball_preds.append(target_relation if (matched_pattern and matched_type) else 'OTHER')
            # 정답: silver에서 해당 관계면 positive, 아니면 OTHER
            actuals.append(actual_rel if actual_rel == target_relation else 'OTHER')

        # 이진 F1 (target_relation vs OTHER)
        dipre_f1    = f1_score(actuals, dipre_preds,
                               pos_label=target_relation, average='binary', zero_division=0)
        snowball_f1 = f1_score(actuals, snowball_preds,
                               pos_label=target_relation, average='binary', zero_division=0)

        dipre_f1s.append(dipre_f1)
        snowball_f1s.append(snowball_f1)
        print(f"  [{target_relation}] DIPRE F1: {dipre_f1:.4f} | Snowball F1: {snowball_f1:.4f}")

    macro_dipre_f1    = float(np.mean(dipre_f1s))    if dipre_f1s    else 0.0
    macro_snowball_f1 = float(np.mean(snowball_f1s)) if snowball_f1s else 0.0

    print(f"\n▶ DIPRE    Macro F1 (관계 평균): {macro_dipre_f1:.4f}")
    print(f"▶ Snowball Macro F1 (관계 평균): {macro_snowball_f1:.4f}")

    # ── 예시 상세 출력 (HAS_FEE 관계, 발표용) ──────────────────────
    target_example = 'HAS_FEE'
    if target_example in gold_df['final_relation'].values:
        print(f"\n▶ [{target_example}] 관계 DIPRE 증식 예시")
        seeds_ex = gold_df[gold_df['final_relation'] == target_example].head(5)
        pats_ex  = defaultdict(int)
        for _, row in seeds_ex.iterrows():
            pat = extract_pattern(row.get('marked_text', ''))
            if pat:
                pats_ex[pat] += 1
        sorted_pats = sorted(pats_ex.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"  Top 3 패턴: {sorted_pats}")

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

        # Snowball: 화폐 단위 포함 여부로 confidence 필터
        filtered = {(h, t) for h, t in extracted
                    if any(kw in t for kw in ('원', 'USD', '00', '$'))}
        print(f"  Snowball 필터 후 튜플 수: {len(filtered)}개")
        if extracted:
            print(f"  노이즈 제거율: {(1 - len(filtered)/len(extracted))*100:.1f}%")

    return macro_dipre_f1, macro_snowball_f1


if __name__ == "__main__":
    run_dipre_and_snowball()
