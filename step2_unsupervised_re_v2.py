import os
import re
import spacy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import v_measure_score
from sentence_transformers import SentenceTransformer
from step1_data_loader import load_gold_standard

# Mac 한글 폰트 설정
plt.rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

def extract_open_ie_tuple(text, nlp):
    """
    Open IE 시뮬레이션:
    사전 정의된 관계 라벨 없이, SpaCy의 의존 구문 분석을 이용해 
    (Subject, Relation/Verb, Object) 형태의 Raw Tuple을 추출합니다.
    """
    if not nlp: return None
    
    doc = nlp(text.replace("[E1]", "").replace("[/E1]", "").replace("[E2]", "").replace("[/E2]", ""))
    
    subject, verb, obj = "", "", ""
    for token in doc:
        if token.dep_ in ('nsubj', 'csubj') and not subject:
            subject = token.text
        if token.pos_ == 'VERB' and not verb:
            verb = token.text
        if token.dep_ in ('obj', 'iobj', 'pobj') and not obj:
            obj = token.text
            
    if subject and verb and obj:
        return f"({subject}, {verb}, {obj})"
    return None

def run_unsupervised_re():
    print("--- 🚀 Step 2. Unsupervised RE (Open IE / Pattern / Distributional / Embedding) ---")
    
    gold_df = load_gold_standard()
    if gold_df.empty: return
    
    try:
        nlp = spacy.load("ko_core_news_sm")
    except Exception:
        nlp = None

    texts = gold_df['marked_text'].fillna(gold_df['sentence']).tolist()
    labels_true = gold_df['final_relation'].tolist()
    k = len(set(labels_true))
    
    print(f"데이터 준비 완료 (K={k}개 군집).")
    
    # ---------------------------------------------------------
    # 1. Open IE (Rule-based Extraction)
    # ---------------------------------------------------------
    print("\n▶ 1. Open IE (의존 구문 기반 관계 추출)")
    open_ie_results = []
    for text in texts[:5]: # 예시 5개만 출력
        res = extract_open_ie_tuple(text, nlp)
        if res: open_ie_results.append(res)
    print("Open IE 추출 예시:", open_ie_results)
    
    # ---------------------------------------------------------
    # 2. Pattern-based Clustering (TF-IDF on Words Between)
    # ---------------------------------------------------------
    print("\n▶ 2. Pattern-based Clustering")
    patterns = []
    for text in texts:
        match = re.search(r'\[/E1\](.*?)\[E2\]', text)
        if not match:
            match = re.search(r'\[/E2\](.*?)\[E1\]', text)
        patterns.append(match.group(1).strip() if match else text)
        
    vec_pattern = TfidfVectorizer(max_features=500)
    X_pattern = vec_pattern.fit_transform(patterns)
    
    kmeans_pattern = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_pattern = kmeans_pattern.fit_predict(X_pattern)
    v_score_pattern = v_measure_score(labels_true, labels_pattern)
    print(f"Pattern-based V-Measure Score: {v_score_pattern:.4f}")
    
    # ---------------------------------------------------------
    # 3. Distributional Similarity / Embedding-based Clustering
    # ---------------------------------------------------------
    print("\n▶ 3. Embedding-based Clustering (Distributional Similarity)")
    print("Sentence-BERT로 문맥 벡터 임베딩 중...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    X_embed = model.encode(texts, show_progress_bar=False)
    
    kmeans_embed = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_embed = kmeans_embed.fit_predict(X_embed)
    v_score_embed = v_measure_score(labels_true, labels_embed)
    print(f"Embedding-based V-Measure Score: {v_score_embed:.4f}")
    
    # ---------------------------------------------------------
    # 시각화: Unsupervised RE 방법론별 성능 비교
    # ---------------------------------------------------------
    methods = ['Pattern-based\n(Lexical TF-IDF)', 'Embedding-based\n(Distributional SBERT)']
    scores = [v_score_pattern, v_score_embed]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(methods, scores, color=['#ffb3b3', '#99ccff'], width=0.5)
    plt.title("Unsupervised RE 방법론별 군집화 성능 비교 (V-Measure)", fontsize=14)
    plt.ylabel("V-Measure Score (0~1)", fontsize=12)
    plt.ylim(0, 1.0)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.4f}", ha='center', fontsize=12, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig('unsupervised_comparison.png', dpi=300)
    print("✅ unsupervised_comparison.png 저장 완료")

if __name__ == "__main__":
    run_unsupervised_re()
