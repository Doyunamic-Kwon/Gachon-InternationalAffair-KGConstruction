import os
import ast
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from scipy.sparse import hstack
from step3_feature_based_re_v2 import extract_all_features
from step1_data_loader import load_gold_standard, load_silver_standard
import matplotlib.font_manager as fm

# Mac 한글 폰트 설정
plt.rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

def parse_entity_type(ent):
    """실버 데이터의 Dictionary 형태 엔티티에서 Type 추출"""
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

def run_ml_visualizations():
    print("데이터 로딩 및 SpaCy 초기화 중...")
    gold_df = load_gold_standard()
    silver_df = load_silver_standard()
    
    try:
        nlp = spacy.load('ko_core_news_sm')
        print("✅ SpaCy Model Loaded!")
    except Exception as e:
        print(f"❌ SpaCy Load Error: {e}")
        nlp = None

    features_context, features_between, features_semantic, features_dep = [], [], [], []
    labels = []
    
    # Gold Data
    for idx, row in gold_df.iterrows():
        ctx, btw, sem, dep = extract_all_features(row['marked_text'], row['head_type'], row['tail_type'], nlp)
        features_context.append(ctx)
        features_between.append(btw)
        features_semantic.append(sem)
        features_dep.append(dep)
        labels.append(row['final_relation'])
        
    # Silver Data (데이터 증강)
    for idx, row in silver_df.iterrows():
        h_type = parse_entity_type(row.get('head'))
        t_type = parse_entity_type(row.get('tail'))
        rel = row.get('relation', '')
        
        if not rel or rel == "NA":
            continue
            
        ctx, btw, sem, dep = extract_all_features(row.get('marked_text', ''), h_type, t_type, nlp)
        features_context.append(ctx)
        features_between.append(btw)
        features_semantic.append(sem)
        features_dep.append(dep)
        labels.append(rel)
        
    print(f"총 추출된 데이터(Gold+Silver): {len(labels)}건")
    
    # Vectorization
    vec_context = TfidfVectorizer(max_features=500)
    X_ctx = vec_context.fit_transform(features_context)
    
    vec_between = TfidfVectorizer(max_features=500)
    X_btw = vec_between.fit_transform(features_between)
    
    vec_sem = CountVectorizer()
    X_sem = vec_sem.fit_transform(features_semantic)
    
    vec_dep = TfidfVectorizer(max_features=500)
    X_dep = vec_dep.fit_transform(features_dep)
    
    X_combined = hstack([X_ctx, X_btw, X_sem, X_dep])
    
    # Train/Test Split (80% Train, 20% Test)
    # 이제 데이터가 1400건 이상이므로 Test 셋도 300건 가까이 됨
    X_train, X_test, y_train, y_test = train_test_split(X_combined, labels, test_size=0.2, random_state=42)
    
    print(f"Random Forest 학습 중... (학습 셋: {X_train.shape[0]}건, 테스트 셋: {X_test.shape[0]}건)")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # 1. Feature Importance (그룹별 통합) 계산
    importances = clf.feature_importances_
    
    len_ctx = X_ctx.shape[1]
    len_btw = X_btw.shape[1]
    len_sem = X_sem.shape[1]
    len_dep = X_dep.shape[1]
    
    imp_ctx = np.sum(importances[0 : len_ctx])
    imp_btw = np.sum(importances[len_ctx : len_ctx+len_btw])
    imp_sem = np.sum(importances[len_ctx+len_btw : len_ctx+len_btw+len_sem])
    imp_dep = np.sum(importances[len_ctx+len_btw+len_sem : len_ctx+len_btw+len_sem+len_dep])
    
    groups = ['Context Words', 'Words Between', 'Semantic Feature', 'Dependency Path']
    imp_scores = [imp_ctx, imp_btw, imp_sem, imp_dep]
    
    # 시각화 1: Feature Importance Bar Plot
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette('pastel')
    bars = plt.bar(groups, imp_scores, color=colors)
    plt.title("Linguistic Features 중요도 분석 (Gold + Silver 1400건 학습)", fontsize=16)
    plt.ylabel("Importance Score 합계", fontsize=12)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.3f}", ha='center', fontsize=12, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300)
    print("✅ feature_importance.png (Dependency 포함 대규모 학습본) 저장 완료")
    
    # 2. Confusion Matrix
    y_pred = clf.predict(X_test)
    # 상위 10개 라벨만 필터링해서 보기 좋게 출력
    unique_labels = pd.Series(y_test).value_counts().head(10).index.tolist()
    
    # 필터링
    mask = np.isin(y_test, unique_labels)
    y_test_filtered = np.array(y_test)[mask]
    y_pred_filtered = np.array(y_pred)[mask]
    
    cm = confusion_matrix(y_test_filtered, y_pred_filtered, labels=unique_labels)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
    plt.title(f"관계 추출 분류 오차 행렬 (Test Set = {len(y_test_filtered)}건)", fontsize=16)
    plt.xlabel("예측된 관계 (Predicted)", fontsize=12)
    plt.ylabel("실제 관계 (Actual)", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    print("✅ confusion_matrix.png (Test 셋 확대본) 저장 완료")

if __name__ == "__main__":
    run_ml_visualizations()
