import os
import spacy
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from step1_data_loader import load_gold_standard

def extract_sequence_and_tree(marked_text, nlp):
    """
    Sequence(단어 시퀀스)와 Tree(구문 트리 간선)를 추출합니다.
    """
    import re
    # 1. Sequence (Words between entities)
    words_between = ""
    between_match = re.search(r'\[/E1\](.*?)\[E2\]', str(marked_text), flags=re.DOTALL)
    if not between_match:
        between_match = re.search(r'\[/E2\](.*?)\[E1\]', str(marked_text), flags=re.DOTALL)
    if between_match:
        words_between = between_match.group(1).strip()
        
    seq_tokens = set(words_between.split())
    
    # 2. Tree (Dependency Edges)
    text_clean = re.sub(r'\[/?E1\]|\[/?E2\]', '', str(marked_text))
    tree_edges = set()
    if nlp and text_clean:
        doc = nlp(text_clean)
        # Tree Kernel을 모사하기 위해 부모-자식(의존성) 쌍을 추출
        for token in doc:
            tree_edges.add(f"{token.head.pos_}_{token.dep_}_{token.pos_}")
            
    return seq_tokens, tree_edges

def compute_composite_kernel(X, alpha=0.5, beta=0.5):
    """
    N x N Composite Kernel Matrix를 계산합니다.
    K(x, y) = alpha * K_seq(x, y) + beta * K_tree(x, y)
    (단순화를 위해 Jaccard Similarity 커널 사용)
    """
    N = len(X)
    K = np.zeros((N, N))
    
    for i in range(N):
        for j in range(N):
            if i <= j:
                seq_i, tree_i = X[i]
                seq_j, tree_j = X[j]
                
                # Sequence Kernel (단어 교집합)
                if len(seq_i | seq_j) == 0:
                    k_seq = 0
                else:
                    k_seq = len(seq_i & seq_j) / len(seq_i | seq_j)
                    
                # Tree Kernel (구문 트리 엣지 교집합)
                if len(tree_i | tree_j) == 0:
                    k_tree = 0
                else:
                    k_tree = len(tree_i & tree_j) / len(tree_i | tree_j)
                    
                composite_score = (alpha * k_seq) + (beta * k_tree)
                K[i, j] = composite_score
                K[j, i] = composite_score # 대칭 행렬
                
    return K

def run_kernel_based_re():
    print("--- 🚀 Step 3-C. Kernel-based RE (Sequence + Tree Composite Kernel) ---")
    
    gold_df = load_gold_standard()
    if gold_df.empty: return
    
    print("\nSpaCy 로드 중...")
    try:
        nlp = spacy.load("ko_core_news_sm")
    except:
        nlp = None
        
    print("Sequence 및 Tree 자질 추출 중...")
    X_raw = []
    labels = []
    
    # Kernel Matrix는 O(N^2) 연산이므로 257건의 Gold Data만 사용하여 실험
    for idx, row in gold_df.iterrows():
        seq, tree = extract_sequence_and_tree(row['marked_text'], nlp)
        X_raw.append((seq, tree))
        labels.append(row['final_relation'])
        
    print(f"데이터 {len(X_raw)}건의 N x N Composite Kernel 계산 중... (시간이 소요될 수 있습니다)")
    K_matrix = compute_composite_kernel(X_raw, alpha=0.4, beta=0.6) # Tree에 약간 더 가중치
    
    # 인덱스로 split (Kernel Matrix 대응을 위해)
    indices = np.arange(len(labels))
    idx_train, idx_test, y_train, y_test = train_test_split(indices, labels, test_size=0.2, random_state=42, stratify=labels)
    
    # 학습/평가용 커널 매트릭스 분리
    K_train = K_matrix[np.ix_(idx_train, idx_train)]
    K_test = K_matrix[np.ix_(idx_test, idx_train)]
    
    print(f"SVM (Kernel='precomputed') 학습 중... (학습 셋: {len(idx_train)}건)")
    clf = SVC(kernel='precomputed', C=1.0)
    clf.fit(K_train, y_train)
    
    y_pred = clf.predict(K_test)
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    print("\n[평가 결과 - Composite Kernel-based RE]")
    print(f"▶ Macro F1-Score: {macro_f1:.4f}")
    
    unique_labels = sorted(list(set(y_test)))
    print("\n상세 Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0, labels=unique_labels))

if __name__ == "__main__":
    run_kernel_based_re()
