import re
import ast
import pandas as pd
from collections import defaultdict
from step1_data_loader import load_gold_standard, load_silver_standard

def extract_pattern(marked_text):
    """
    [E1]...[/E1] 와 [E2]...[/E2] 사이의 패턴(단어들)을 추출합니다.
    """
    match1 = re.search(r'\[/E1\](.*?)\[E2\]', str(marked_text), flags=re.DOTALL)
    if match1:
        return match1.group(1).strip()
    match2 = re.search(r'\[/E2\](.*?)\[E1\]', str(marked_text), flags=re.DOTALL)
    if match2:
        return match2.group(1).strip()
    return None

def run_dipre_and_snowball():
    print("--- 🚀 Step 3-B. Semi-supervised RE (DIPRE & Snowball) ---")
    
    gold_df = load_gold_standard()
    silver_df = load_silver_standard()
    
    if gold_df.empty or silver_df.empty:
        print("데이터를 불러올 수 없습니다.")
        return
        
    target_relation = "HAS_FEE"
    print(f"\n[실험 대상 관계: {target_relation}]")
    
    # 1. DIPRE 단계
    print("\n▶ [DIPRE] Seed를 이용한 패턴 추출 및 튜플 무한 증식")
    seed_data = gold_df[gold_df['final_relation'] == target_relation].head(5)
    
    patterns = defaultdict(int)
    for text in seed_data['marked_text']:
        pat = extract_pattern(text)
        if pat:
            patterns[pat] += 1
            
    print(f"Seed 문장에서 추출된 핵심 패턴 수: {len(patterns)}개")
    sorted_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:3]
    print("Top 3 추출 패턴 (Seed 기준):", sorted_patterns)
    
    # Silver data에서 해당 패턴을 가진 문장 검색하여 튜플 증식
    extracted_tuples_dipre = set()
    for _, row in silver_df.iterrows():
        marked = row.get('marked_text', '')
        # 패턴 중 하나라도 포함되어 있으면 매칭되었다고 판단
        for pat, _ in sorted_patterns:
            if pat and pat in str(marked) and "[E1]" in str(marked) and "[E2]" in str(marked):
                try:
                    head_match = re.search(r'\[E1\](.*?)\[/E1\]', marked)
                    tail_match = re.search(r'\[E2\](.*?)\[/E2\]', marked)
                    if head_match and tail_match:
                        head_cand = head_match.group(1).strip()
                        tail_cand = tail_match.group(1).strip()
                        extracted_tuples_dipre.add((head_cand, tail_cand))
                except:
                    pass
                    
    print(f"DIPRE 증식을 통해 대규모 코퍼스(Silver)에서 새로 추출된 튜플 수: {len(extracted_tuples_dipre)}개")
    if len(extracted_tuples_dipre) > 0:
        print("DIPRE 추출 튜플 예시 (무한 증식/노이즈 포함):")
        for t in list(extracted_tuples_dipre)[:5]:
            print(f" - {t}")
    
    # 2. Snowball 단계
    print("\n▶ [Snowball] Confidence Score를 통한 노이즈 필터링 (Semantic Drift 제어)")
    snowball_filtered_tuples = set()
    
    for head, tail in extracted_tuples_dipre:
        score = 0.0
        # HAS_FEE 특성: tail에 '원', 'USD' 등 화폐 단위가 있으면 높은 점수
        if "원" in str(tail) or "00" in str(tail) or "USD" in str(tail):
            score += 0.9
        else:
            score += 0.2
            
        if score >= 0.8:
            snowball_filtered_tuples.add((head, tail))
            
    print(f"Snowball Confidence Score 적용 후 살아남은 튜플 수: {len(snowball_filtered_tuples)}개")
    
    if len(extracted_tuples_dipre) > 0:
        eliminated = 100 - (len(snowball_filtered_tuples)/len(extracted_tuples_dipre)*100)
        print(f"노이즈 필터링 비율: {eliminated:.1f}% 의 '관계 없는 튜플' 제거됨")
        
    if len(snowball_filtered_tuples) > 0:
        print("Snowball 정제 튜플 예시 (순도 높음):")
        for t in list(snowball_filtered_tuples)[:5]:
            print(f" - {t}")

if __name__ == "__main__":
    run_dipre_and_snowball()
