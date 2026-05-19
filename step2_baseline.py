import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import v_measure_score
from sentence_transformers import SentenceTransformer
from step1_data_loader import load_gold_standard

def run_unsupervised_baseline():
    print("--- 🚀 Step 2. Unsupervised Clustering 베이스라인 (Sentence-BERT 적용) ---")
    
    gold_df = load_gold_standard()
    if gold_df.empty:
        print("평가할 데이터가 없습니다.")
        return
        
    texts = gold_df['marked_text'].fillna(gold_df['sentence']).tolist()
    labels_true = gold_df['final_relation'].tolist()
    
    unique_labels = list(set(labels_true))
    k = len(unique_labels)
    print(f"데이터 준비 완료. 총 {len(texts)} 문장 / 실제 관계 종류 (K): {k}")
    
    print("텍스트 벡터화 진행 (Sentence-BERT)...")
    # 한국어 처리에 적합한 다국어 임베딩 모델 사용
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    X = model.encode(texts, show_progress_bar=True)
    
    print(f"K-Means 클러스터링 학습 중 (n_clusters={k})...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_pred = kmeans.fit_predict(X)
    
    v_score = v_measure_score(labels_true, labels_pred)
    
    print("\n[평가 결과]")
    print(f"▶ V-Measure Score: {v_score:.4f}")
    print("설명: 단순 단어 빈도(TF-IDF)가 아닌 Sentence-BERT 임베딩을 사용하여 문맥의 의미를 파악하게 함으로써, 성능을 현실적인 베이스라인 수준으로 끌어올렸습니다.")

if __name__ == "__main__":
    run_unsupervised_baseline()
