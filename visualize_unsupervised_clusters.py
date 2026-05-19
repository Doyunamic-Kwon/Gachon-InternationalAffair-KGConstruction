import os
import re
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
from step1_data_loader import load_gold_standard, load_silver_standard

# Mac 한글 폰트 설정
plt.rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

def run_unsupervised_cluster_visualization():
    print("데이터 로딩 중 (Gold + Silver 전체)...")
    gold_df = load_gold_standard()
    silver_df = load_silver_standard()

    texts = []
    labels_true = []

    # Gold Data
    for _, row in gold_df.iterrows():
        texts.append(str(row.get('marked_text', row.get('sentence', ''))))
        labels_true.append(row['final_relation'])

    # Silver Data
    for _, row in silver_df.iterrows():
        rel = row.get('relation', '')
        if not rel or rel == 'NA':
            continue
        texts.append(str(row.get('marked_text', '')))
        labels_true.append(rel)

    print(f"총 데이터: {len(texts)}건 (Gold {len(gold_df)} + Silver {len(texts)-len(gold_df)})")
    k = len(set(labels_true))
    print(f"군집 수 K = {k}")

    # ---------------------------------------------------------
    # 1. Pattern-based (TF-IDF) Representation
    # ---------------------------------------------------------
    print("Pattern-based TF-IDF 추출 및 군집화 중...")
    patterns = []
    for text in texts:
        match = re.search(r'\[/E1\](.*?)\[E2\]', text)
        if not match:
            match = re.search(r'\[/E2\](.*?)\[E1\]', text)
        patterns.append(match.group(1).strip() if match else text)

    vec_pattern = TfidfVectorizer(max_features=500)
    X_pattern = vec_pattern.fit_transform(patterns)

    kmeans_pattern = KMeans(n_clusters=k, random_state=42, n_init=10)
    y_pattern_pred = kmeans_pattern.fit_predict(X_pattern)

    # ---------------------------------------------------------
    # 2. Distributional Similarity (Sentence-BERT) Representation
    # ---------------------------------------------------------
    print("Sentence-BERT 임베딩 추출 및 군집화 중...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    X_embed = model.encode(texts, show_progress_bar=True)

    kmeans_embed = KMeans(n_clusters=k, random_state=42, n_init=10)
    y_embed_pred = kmeans_embed.fit_predict(X_embed)

    # ---------------------------------------------------------
    # 3. 차원 축소 (t-SNE) - 1400건 이상이므로 perplexity=50
    # ---------------------------------------------------------
    print("t-SNE 2D 차원 축소 진행 중 (데이터 수가 많아 시간이 소요됩니다)...")
    tsne = TSNE(n_components=2, random_state=42, init='random', perplexity=50)

    X_pattern_dense = X_pattern.toarray()
    X_pattern_2d = tsne.fit_transform(X_pattern_dense)
    X_embed_2d = tsne.fit_transform(X_embed)

    # ---------------------------------------------------------
    # 4. 시각화 (Subplots)
    # ---------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    palette = sns.color_palette("husl", k)

    # Plot 1: Pattern-based
    sns.scatterplot(
        x=X_pattern_2d[:, 0], y=X_pattern_2d[:, 1],
        hue=y_pattern_pred,
        palette=palette,
        ax=ax1,
        legend=False,
        s=25, alpha=0.7
    )
    ax1.set_title(f"1. Pattern-based 군집화 결과 ({len(texts)}건)\n단어의 표면적 일치에 의존 → 군집이 산재함", fontsize=13)
    ax1.set_xlabel("t-SNE Dimension 1")
    ax1.set_ylabel("t-SNE Dimension 2")

    # Plot 2: Embedding-based
    sns.scatterplot(
        x=X_embed_2d[:, 0], y=X_embed_2d[:, 1],
        hue=y_embed_pred,
        palette=palette,
        ax=ax2,
        legend=False,
        s=25, alpha=0.7
    )
    ax2.set_title(f"2. Embedding-based 군집화 결과 ({len(texts)}건)\nSentence-BERT 문맥 유사도 → 군집이 밀도있게 뭉침", fontsize=13)
    ax2.set_xlabel("t-SNE Dimension 1")
    ax2.set_ylabel("t-SNE Dimension 2")

    plt.tight_layout()
    plt.savefig('unsupervised_clusters_tsne.png', dpi=300)
    print("✅ unsupervised_clusters_tsne.png 저장 완료")

if __name__ == "__main__":
    run_unsupervised_cluster_visualization()
