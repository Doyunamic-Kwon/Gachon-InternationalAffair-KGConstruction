import os
import spacy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from step1_data_loader import load_gold_standard, load_silver_standard
from step3c_kernel_based_re import extract_sequence_and_tree
from step3_feature_based_re_v2 import parse_entity_type

plt.rc('font', family='DejaVu Sans')
plt.rcParams['axes.unicode_minus'] = False

def compute_improved_composite_kernel(X_seq, X_tree, X_sem, alpha=0.3, beta=0.3, gamma=0.4):
    """
    K_composite = alpha * K_seq + beta * K_tree + gamma * K_semantic
    개체 타입(Semantic) 커널을 추가하여 성능 대폭 향상
    """
    N = len(X_seq)
    K = np.zeros((N, N))
    
    for i in range(N):
        for j in range(N):
            if i <= j:
                seq_i, tree_i = X_seq[i], X_tree[i]
                seq_j, tree_j = X_seq[j], X_tree[j]
                
                # 1. Sequence Kernel (Jaccard)
                k_seq = len(seq_i & seq_j) / len(seq_i | seq_j) if len(seq_i | seq_j) > 0 else 0
                    
                # 2. Tree Kernel (Jaccard)
                k_tree = len(tree_i & tree_j) / len(tree_i | tree_j) if len(tree_i | tree_j) > 0 else 0
                
                # 3. Semantic Kernel (Entity Type Match)
                # 둘 다 일치하면 1, 아니면 0
                k_sem = 1.0 if X_sem[i] == X_sem[j] else 0.0
                
                composite_score = (alpha * k_seq) + (beta * k_tree) + (gamma * k_sem)
                K[i, j] = composite_score
                K[j, i] = composite_score
                
    return K

def run_kernel_visualizations():
    print("데이터 로딩 및 SpaCy 초기화 중...")
    gold_df = load_gold_standard()
    silver_df = load_silver_standard()
    
    try:
        nlp = spacy.load('ko_core_news_sm')
    except Exception:
        nlp = None

    X_seq = []
    X_tree = []
    X_sem = []
    labels = []
    
    # 1. Gold Data 파싱
    for idx, row in gold_df.iterrows():
        seq, tree = extract_sequence_and_tree(row['marked_text'], nlp)
        X_seq.append(seq)
        X_tree.append(tree)
        X_sem.append(f"{row['head_type']}|{row['tail_type']}")
        labels.append(row['final_relation'])
        
    # 2. Silver Data 파싱
    # 커널 행렬은 O(N^2) 연산이므로 최대 743건 샘플링 → 총 1,000건으로 맞춤
    silver_valid = silver_df.dropna(subset=['relation'])
    silver_valid = silver_valid[silver_valid['relation'].apply(lambda x: x and x != 'NA')]
    silver_sampled = silver_valid.sample(min(743, len(silver_valid)), random_state=42)
    for idx, row in silver_sampled.iterrows():
        rel = row.get('relation', '')
        if not rel or rel == "NA": continue
        seq, tree = extract_sequence_and_tree(row.get('marked_text', ''), nlp)
        h_type = parse_entity_type(row.get('head'))
        t_type = parse_entity_type(row.get('tail'))
        X_seq.append(seq)
        X_tree.append(tree)
        X_sem.append(f"{h_type}|{t_type}")
        labels.append(rel)
        
    print(f"총 {len(labels)}건의 데이터로 N x N 커널 행렬 계산 중... (Kernel은 O(N²) 제약으로 1,000건 수준)")
    # K_composite = 0.3*K_seq + 0.3*K_tree + 0.4*K_semantic (균형있게 재조정)
    K_matrix = compute_improved_composite_kernel(X_seq, X_tree, X_sem, alpha=0.3, beta=0.3, gamma=0.4)
    
    indices = np.arange(len(labels))
    idx_train, idx_test, y_train, y_test = train_test_split(indices, labels, test_size=0.2, random_state=42)
    
    K_train = K_matrix[np.ix_(idx_train, idx_train)]
    K_test = K_matrix[np.ix_(idx_test, idx_train)]
    
    print(f"SVM 학습 중...")
    clf = SVC(kernel='precomputed', C=1.0)
    clf.fit(K_train, y_train)
    
    y_pred = clf.predict(K_test)
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    print(f"▶ 개선된 Kernel-based RE Macro F1-Score: {macro_f1:.4f}")
    
    # ----------------------------------------------------
    # 시각화 1: Kernel PCA + t-SNE 시각화 (군집 확인)
    # ----------------------------------------------------
    print("Kernel Matrix 기반 t-SNE 2D 투영 시각화 생성 중...")
    # 주요 라벨 5개만 시각화하여 명확히 보기
    top_labels = pd.Series(labels).value_counts().head(5).index.tolist()
    
    # t-SNE는 거리를 기반으로 하므로 1 - K (거리행렬) 사용
    dist_matrix = 1.0 - K_matrix
    tsne = TSNE(n_components=2, metric='precomputed', random_state=42, init='random')
    embedding_2d = tsne.fit_transform(dist_matrix)
    
    plt.figure(figsize=(10, 8))
    for label in top_labels:
        idx = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(embedding_2d[idx, 0], embedding_2d[idx, 1], label=label, alpha=0.7)
        
    plt.title("Kernel Matrix Visualization (t-SNE Projection)", fontsize=14)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(title='Relations', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('kernel_tsne.png', dpi=300)
    print("✅ kernel_tsne.png 저장 완료")
    
    # ----------------------------------------------------
    # 시각화 2: Confusion Matrix (오차 행렬)
    # ----------------------------------------------------
    print("Confusion Matrix 생성 중...")
    test_top_labels = pd.Series(y_test).value_counts().head(8).index.tolist()
    mask = np.isin(y_test, test_top_labels)
    y_test_filtered = np.array(y_test)[mask]
    y_pred_filtered = np.array(y_pred)[mask]
    
    cm = confusion_matrix(y_test_filtered, y_pred_filtered, labels=test_top_labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=test_top_labels, yticklabels=test_top_labels)
    plt.title("Kernel SVM Confusion Matrix", fontsize=14)
    plt.xlabel("Predicted Relation")
    plt.ylabel("True Relation")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('kernel_confusion_matrix.png', dpi=300)
    print("✅ kernel_confusion_matrix.png 저장 완료")

if __name__ == "__main__":
    run_kernel_visualizations()
