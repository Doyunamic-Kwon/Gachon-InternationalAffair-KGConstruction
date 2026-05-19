import re
import ast
import spacy
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
from step1_data_loader import load_gold_standard, load_silver_standard

def parse_entity_type(ent):
    if isinstance(ent, dict):
        return ent.get('type', 'UNKNOWN')
    elif isinstance(ent, str):
        try:
            ent_dict = ast.literal_eval(ent)
            if isinstance(ent_dict, dict):
                return ent_dict.get('type', 'UNKNOWN')
        except:
            pass
    return "UNKNOWN"

def extract_all_features(marked_text, head_type, tail_type, nlp):
    if pd.isna(marked_text):
        return "", "", "", ""
        
    text_clean = re.sub(r'\[/?E1\]|\[/?E2\]', '', str(marked_text))
    
    match1 = re.search(r'\[E1\](.*?)\[/E1\]', str(marked_text))
    match2 = re.search(r'\[E2\](.*?)\[/E2\]', str(marked_text))
    
    e1_text = match1.group(1).strip() if match1 else ""
    e2_text = match2.group(1).strip() if match2 else ""
    
    words_between = ""
    between_match = re.search(r'\[/E1\](.*?)\[E2\]', str(marked_text), flags=re.DOTALL)
    if not between_match:
        between_match = re.search(r'\[/E2\](.*?)\[E1\]', str(marked_text), flags=re.DOTALL)
    if between_match:
        words_between = between_match.group(1).strip()
        
    context_words = text_clean.replace(e1_text, "").replace(e2_text, "").strip()
    
    dep_path = ""
    if nlp and text_clean:
        doc = nlp(text_clean)
        path_list = [f"{token.pos_}({token.dep_})" for token in doc if token.pos_ in ['NOUN', 'VERB', 'ADJ']]
        dep_path = " -> ".join(path_list[:5])
    if not dep_path:
        dep_path = "NO_DEP"
        
    semantic = f"{head_type} | {tail_type}"
    
    return context_words, words_between, semantic, dep_path

def run_advanced_feature_based_re():
    print("--- 🚀 Step 3. 고도화된 Feature-based RE (Gold + Silver 데이터 통합) ---")
    
    gold_df = load_gold_standard()
    silver_df = load_silver_standard()
    
    print("\nSpaCy ko_core_news_sm 로드 중...")
    try:
        nlp = spacy.load("ko_core_news_sm")
        print("✅ SpaCy 모델 로드 완료!")
    except Exception as e:
        print(f"❌ SpaCy Load Error: {e}")
        nlp = None

    features_context, features_between, features_semantic, features_dep = [], [], [], []
    labels = []
    
    print("\n다양한 Linguistic Feature 추출 중...")
    
    # 1. Gold Data 파싱
    for idx, row in gold_df.iterrows():
        ctx, btw, sem, dep = extract_all_features(row['marked_text'], row['head_type'], row['tail_type'], nlp)
        features_context.append(ctx)
        features_between.append(btw)
        features_semantic.append(sem)
        features_dep.append(dep)
        labels.append(row['final_relation'])
        
    # 2. Silver Data 파싱
    for idx, row in silver_df.iterrows():
        h_type = parse_entity_type(row.get('head'))
        t_type = parse_entity_type(row.get('tail'))
        rel = row.get('relation', '')
        
        if not rel or rel == "NA": continue
            
        ctx, btw, sem, dep = extract_all_features(row.get('marked_text', ''), h_type, t_type, nlp)
        features_context.append(ctx)
        features_between.append(btw)
        features_semantic.append(sem)
        features_dep.append(dep)
        labels.append(rel)
    
    print(f"총 데이터: {len(labels)}건")
    
    # 예시 하나 출력
    print("\n[추출된 Feature 예시 (Dependency 확인)]")
    print(f"▶ 원문: {gold_df.iloc[0]['marked_text']}")
    print(f"1) Context Words (주변 단어): {features_context[0][:50]}...")
    print(f"2) Words Between Mentions (사이 단어): {features_between[0]}")
    print(f"3) Semantic Feature (개체 타입): {features_semantic[0]}")
    print(f"4) Dependency Path (구문 의존 경로): {features_dep[0]}")
    print("-" * 50)
            
    # TF-IDF Vectorization
    vec_context = TfidfVectorizer(max_features=500)
    X_ctx = vec_context.fit_transform(features_context)
    
    vec_between = TfidfVectorizer(max_features=500)
    X_btw = vec_between.fit_transform(features_between)
    
    vec_sem = CountVectorizer()
    X_sem = vec_sem.fit_transform(features_semantic)
    
    vec_dep = TfidfVectorizer(max_features=500)
    X_dep = vec_dep.fit_transform(features_dep)
    
    X_combined = hstack([X_ctx, X_btw, X_sem, X_dep])
    
    X_train, X_test, y_train, y_test = train_test_split(X_combined, labels, test_size=0.2, random_state=42)
    
    print(f"\nRandom Forest 학습 중... (학습 셋: {X_train.shape[0]}건, 테스트 셋: {X_test.shape[0]}건, 총 피처 수: {X_combined.shape[1]})")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    print("\n[평가 결과 - 고도화된 Feature-based RE (확장 데이터)]")
    print(f"▶ Macro F1-Score: {macro_f1:.4f}")
    
    # 상위 10개 라벨에 대해서만 Classification Report 출력 (가독성을 위해)
    unique_labels = pd.Series(y_test).value_counts().head(10).index.tolist()
    print("\n상세 Classification Report (상위 10개 관계):")
    
    # 필터링
    mask = np.isin(y_test, unique_labels)
    y_test_filtered = np.array(y_test)[mask]
    y_pred_filtered = np.array(y_pred)[mask]
    
    print(classification_report(y_test_filtered, y_pred_filtered, zero_division=0, labels=unique_labels))

if __name__ == "__main__":
    run_advanced_feature_based_re()
