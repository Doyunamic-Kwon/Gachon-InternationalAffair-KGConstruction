"""
Step 1 재분석: Unsupervised RE 성능 심화 분석
OpenAI 개선 후 성능 비교
"""
import json, re
from pathlib import Path
from collections import Counter
import numpy as np

def analyze_corpus_quality():
    """Analyze corpus_clean.jsonl quality"""
    with open("data/re_fixed_v6/corpus_clean.jsonl") as f:
        corpus = [json.loads(line) for line in f]
    
    # 1. Context source 분포
    context_dist = Counter(r.get("context_src", "unknown") for r in corpus)
    print("📊 Context Source Distribution:")
    for src, cnt in context_dist.most_common():
        print(f"  {src:15s}: {cnt:4d} ({cnt/len(corpus)*100:5.1f}%)")
    
    # 2. 패턴 길이 분포
    def get_pattern(row):
        marked = str(row.get("marked_text", ""))
        m = re.search(r'\[/E1\](.*?)\[E2\]', marked, re.DOTALL)
        if not m:
            m = re.search(r'\[/E2\](.*?)\[E1\]', marked, re.DOTALL)
        return m.group(1).strip() if m else ""
    
    patterns = [get_pattern(r) for r in corpus]
    pattern_lengths = [len(p) for p in patterns]
    
    print("\n📏 Pattern Length Distribution:")
    print(f"  Empty (0자):     {sum(1 for l in pattern_lengths if l == 0):4d} ({sum(1 for l in pattern_lengths if l == 0)/len(pattern_lengths)*100:5.1f}%)")
    print(f"  Short (1-9자):   {sum(1 for l in pattern_lengths if 0 < l < 10):4d} ({sum(1 for l in pattern_lengths if 0 < l < 10)/len(pattern_lengths)*100:5.1f}%)")
    print(f"  Rich (10+자):    {sum(1 for l in pattern_lengths if l >= 10):4d} ({sum(1 for l in pattern_lengths if l >= 10)/len(pattern_lengths)*100:5.1f}%)")
    
    # 3. Relation 분포
    rel_dist = Counter(r.get("relation", "UNKNOWN") for r in corpus)
    print(f"\n📋 Relation Distribution ({len(rel_dist)}개):")
    for rel, cnt in rel_dist.most_common():
        print(f"  {rel:35s}: {cnt:4d} ({cnt/len(corpus)*100:5.1f}%)")
    
    return corpus

if __name__ == "__main__":
    print("--- Step 2 분석: OpenAI 개선 전후 ---\n")
    analyze_corpus_quality()
