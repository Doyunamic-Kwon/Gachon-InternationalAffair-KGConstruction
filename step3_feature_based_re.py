import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
from step1_data_loader import load_gold_standard

def run_feature_based_re():
    print("--- 🚀 Step 3. Feature-based RE 베이스라인 (TF-IDF + Entity Types) ---")
    
    gold_df = load_gold_standard()
    if gold_df.empty:
        print("평가할 데이터가 없습니다.")
        return
        
    print("문장 텍스트 및 개체 타입(Entity Type) 특징 추출 중...")
    
    # 텍스트 특징 (전체 문장 사용)
    texts = gold_df['sentence'].fillna("").tolist()
    
    # 개체 타입 특징
    head_types = gold_df['head_type'].fillna("UNKNOWN").values.reshape(-1, 1)
    tail_types = gold_df['tail_type'].fillna("UNKNOWN").values.reshape(-1, 1)
    
    labels = gold_df['final_relation'].tolist()
    
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=1000)
    X_text = vectorizer.fit_transform(texts)
    
    # One-Hot Encoding for Entity Types
    encoder = OneHotEncoder(handle_unknown='ignore')
    X_types = encoder.fit_transform(np.hstack((head_types, tail_types)))
    
    # Combine features
    X_combined = hstack([X_text, X_types])
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_combined, labels, test_size=0.2, random_state=42, stratify=labels)
    print(f"학습 데이터: {X_train.shape[0]}건, 테스트 데이터: {X_test.shape[0]}건")
    
    print("Random Forest Classifier 학습 중...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    print("\n[평가 결과 - Feature-based RE]")
    print(f"▶ Macro F1-Score: {macro_f1:.4f}")
    print("\n상세 Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

if __name__ == "__main__":
    run_feature_based_re()
