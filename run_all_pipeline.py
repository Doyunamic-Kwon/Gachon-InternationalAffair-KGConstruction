"""
전체 OIA RE 파이프라인 마스터 실행 스크립트
- Step 2: Unsupervised (Pattern-based + Embedding-based)
- Step 3b: Semi-supervised (DIPRE + Snowball)
- Step 3_v2: Supervised Feature-based (Random Forest)
- visualize_kernel_ml: Supervised Kernel-based (Composite SVM)
- Step 4: Deep Learning (Bi-LSTM + Attention)
결과를 docs/results.json에 저장하고 발표용 시각화를 docs/에 출력합니다.
"""
import os, json, re, ast, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

os.makedirs("docs", exist_ok=True)
plt.rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False
RESULTS = {}

# ─────────────────────────────────────────────────────────────────────
# 공통 유틸
# ─────────────────────────────────────────────────────────────────────
def save_fig(name):
    path = f"docs/{name}.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ docs/{name}.png 저장 완료")


# ─────────────────────────────────────────────────────────────────────
# 0. 데이터 로드 + 데이터셋 개요 시각화
# ─────────────────────────────────────────────────────────────────────
def step0_data_overview():
    print("\n" + "="*60)
    print("  STEP 0. 데이터셋 개요")
    print("="*60)
    from step1_data_loader import load_gold_standard, load_silver_standard
    gold_df   = load_gold_standard()
    silver_df = load_silver_standard()

    # Silver relation 정리
    silver_valid = silver_df.dropna(subset=['relation'])
    silver_valid = silver_valid[
        silver_valid['relation'].apply(lambda x: bool(x) and x != 'NA')
    ]

    print(f"  Gold:   {len(gold_df)}건  |  Silver: {len(silver_valid)}건  |  합계: {len(gold_df)+len(silver_valid)}건")
    print(f"  Gold 관계 수: {gold_df['final_relation'].nunique()}개")

    # 관계 분포 시각화
    gold_counts = gold_df['final_relation'].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # (1) Gold 관계 분포
    colors = sns.color_palette("Set2", len(gold_counts))
    axes[0].barh(gold_counts.index, gold_counts.values, color=colors)
    for i, v in enumerate(gold_counts.values):
        axes[0].text(v + 1, i, str(v), va='center', fontsize=9, fontweight='bold')
    axes[0].set_title("Gold Standard 관계 분포\n(수작업 레이블 257건)", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("샘플 수")
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, gold_counts.max() * 1.2)

    # (2) Gold vs Silver 파이 차트
    sizes  = [len(gold_df), len(silver_valid)]
    labels = [f'Gold Standard\n(수작업 레이블)\n{len(gold_df)}건',
              f'Silver Standard\n(LLM 레이블)\n{len(silver_valid)}건']
    explode = (0.05, 0)
    wedge_props = dict(edgecolor='white', linewidth=2)
    axes[1].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,
                colors=['#4c72b0', '#dd8452'], explode=explode,
                wedgeprops=wedge_props, textprops={'fontsize': 12})
    axes[1].set_title("데이터셋 구성\n(Gold + Silver = 총 1,462건)", fontsize=13, fontweight='bold')

    plt.suptitle("OIA 관계 추출 데이터셋 개요\n(가천대 학교 공지/행정 텍스트)",
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_fig("step0_data_overview")
    return gold_df, silver_df, silver_valid


# ─────────────────────────────────────────────────────────────────────
# 1. Unsupervised RE
# ─────────────────────────────────────────────────────────────────────
def step1_unsupervised(gold_df):
    print("\n" + "="*60)
    print("  STEP 1. Unsupervised RE")
    print("="*60)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.metrics import v_measure_score
    from sentence_transformers import SentenceTransformer
    from sklearn.manifold import TSNE

    texts       = gold_df['marked_text'].fillna('').tolist()
    labels_true = gold_df['final_relation'].tolist()
    k = len(set(labels_true))

    # ── Pattern-based ──────────────────────────────────
    patterns = []
    for t in texts:
        m = re.search(r'\[/E1\](.*?)\[E2\]', t) or re.search(r'\[/E2\](.*?)\[E1\]', t)
        patterns.append(m.group(1).strip() if m else t)

    X_pat = TfidfVectorizer(max_features=500).fit_transform(patterns)
    y_pat = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_pat)
    v_pat = v_measure_score(labels_true, y_pat)
    print(f"  Pattern-based  V-Measure: {v_pat:.4f}")

    # ── Embedding-based ────────────────────────────────
    print("  SBERT 임베딩 중 (paraphrase-multilingual-MiniLM-L12-v2)...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    X_emb = model.encode(texts, batch_size=64, show_progress_bar=False)
    y_emb = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_emb)
    v_emb = v_measure_score(labels_true, y_emb)
    print(f"  Embedding-based V-Measure: {v_emb:.4f}")

    # ── 시각화 1: 성능 비교 막대 ────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(['Pattern-based\n(Lexical TF-IDF)', 'Embedding-based\n(Sentence-BERT)'],
                  [v_pat, v_emb],
                  color=['#ffb3b3', '#99ccff'], width=0.45, edgecolor='gray')
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                f"{b.get_height():.4f}", ha='center', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_ylabel("V-Measure Score", fontsize=12)
    ax.set_title("Unsupervised RE — 군집화 성능 비교 (V-Measure)\n"
                 "Gold 257건 | K-Means (K=관계 수)", fontsize=13, fontweight='bold')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='0.5 기준선')
    # 방법론 설명 박스
    desc = ("V-Measure = Homogeneity × Completeness의 조화 평균\n"
            "• Homogeneity: 군집 내 같은 관계끼리만 모였는가\n"
            "• Completeness: 같은 관계가 한 군집에 모였는가\n"
            "→ SBERT는 문맥 의미를 파악해 패턴 기반 대비 +{:.1f}% 향상".format(
                (v_emb - v_pat) / v_pat * 100))
    ax.text(0.98, 0.97, desc, transform=ax.transAxes, fontsize=9,
            va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    plt.tight_layout()
    save_fig("step1_unsupervised_comparison")

    # ── 시각화 2: t-SNE ────────────────────────────────
    print("  t-SNE 차원 축소 중...")
    all_labels_set = sorted(set(labels_true))
    palette = sns.color_palette("tab10", len(all_labels_set))
    color_map = {l: palette[i] for i, l in enumerate(all_labels_set)}
    colors_true = [color_map[l] for l in labels_true]

    tsne = TSNE(n_components=2, random_state=42, init='random', perplexity=30, max_iter=1000)
    emb2d = tsne.fit_transform(X_emb)
    pat2d = tsne.fit_transform(X_pat.toarray())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    for lbl in all_labels_set:
        idx = [i for i, l in enumerate(labels_true) if l == lbl]
        ax1.scatter([pat2d[i,0] for i in idx], [pat2d[i,1] for i in idx],
                    c=[color_map[lbl]], label=lbl, s=25, alpha=0.7)
        ax2.scatter([emb2d[i,0] for i in idx], [emb2d[i,1] for i in idx],
                    c=[color_map[lbl]], label=lbl, s=25, alpha=0.7)

    ax1.set_title(f"Pattern-based (TF-IDF K-Means)\nV-Measure: {v_pat:.4f}", fontsize=13)
    ax1.set_xlabel("t-SNE dim 1"); ax1.set_ylabel("t-SNE dim 2")
    ax2.set_title(f"Embedding-based (SBERT K-Means)\nV-Measure: {v_emb:.4f}", fontsize=13)
    ax2.set_xlabel("t-SNE dim 1")
    handles = [plt.Line2D([0],[0], marker='o', color='w',
                           markerfacecolor=color_map[l], markersize=8) for l in all_labels_set]
    ax2.legend(handles, all_labels_set, bbox_to_anchor=(1.02, 1),
               loc='upper left', fontsize=8, title="관계 유형")
    plt.suptitle("Unsupervised RE — t-SNE 군집 시각화 (Gold 257건)\n"
                 "같은 색 = 같은 정답 관계 | 군집이 뚜렷할수록 V-Measure ↑",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_fig("step1_unsupervised_tsne")

    RESULTS['unsup_pattern'] = round(v_pat, 4)
    RESULTS['unsup_embed']   = round(v_emb, 4)
    return v_pat, v_emb


# ─────────────────────────────────────────────────────────────────────
# 2. Semi-supervised RE (DIPRE + Snowball)
# ─────────────────────────────────────────────────────────────────────
def step2_semi_supervised(gold_df, silver_df):
    print("\n" + "="*60)
    print("  STEP 2. Semi-supervised RE (DIPRE & Snowball)")
    print("="*60)
    from step3b_semi_supervised import run_dipre_and_snowball, extract_pattern
    from sklearn.metrics import f1_score
    from collections import defaultdict

    macro_dipre_f1, macro_snowball_f1 = run_dipre_and_snowball()
    print(f"  DIPRE    Macro F1: {macro_dipre_f1:.4f}")
    print(f"  Snowball Macro F1: {macro_snowball_f1:.4f}")

    # ── DIPRE Semantic Drift 시뮬레이션 (HAS_FEE 관계 예시) ────────
    silver_valid = silver_df.dropna(subset=['relation'])
    silver_valid = silver_valid[
        silver_valid['relation'].apply(lambda x: bool(x) and x != 'NA')
    ].reset_index(drop=True)

    target_rel = 'HAS_FEE'
    if target_rel in gold_df['final_relation'].values:
        seeds = gold_df[gold_df['final_relation'] == target_rel].head(5)
        pats_dict = defaultdict(int)
        for _, row in seeds.iterrows():
            p = extract_pattern(row.get('marked_text', ''))
            if p:
                pats_dict[p] += 1

        # 5 iteration bootstrapping precision 추적
        dipre_pats     = [p for p, _ in sorted(pats_dict.items(), key=lambda x: -x[1])[:3]]
        snowball_pats  = dipre_pats.copy()
        dipre_pool     = silver_valid.copy()
        snowball_pool  = silver_valid.copy()
        dipre_precs    = []
        snowball_precs = []

        seed_head_types = set(seeds['head_type'].dropna())
        seed_tail_types = set(seeds['tail_type'].dropna())

        for it in range(1, 6):
            # DIPRE
            d_mask = dipre_pool['marked_text'].apply(
                lambda t: any(p in str(t) for p in dipre_pats if p))
            d_matched = dipre_pool[d_mask]
            dp = (d_matched['relation'] == target_rel).mean() if len(d_matched) else 0.0
            dipre_precs.append(dp)
            if len(d_matched):
                new_pats = []
                for _, r in d_matched.head(5).iterrows():
                    np_val = extract_pattern(r.get('marked_text', ''))
                    if np_val:
                        new_pats.append(np_val)
                dipre_pats = list(set(dipre_pats + new_pats))[:8]
                dipre_pool = dipre_pool[~d_mask].reset_index(drop=True)

            # Snowball (엔티티 타입 필터 추가)
            def sw_filter(row):
                txt = str(row.get('marked_text', ''))
                h_t = row.get('head_type', '') or ''
                t_t = row.get('tail_type', '') or ''
                if isinstance(row.get('head'), dict):
                    h_t = row['head'].get('type', '')
                if isinstance(row.get('tail'), dict):
                    t_t = row['tail'].get('type', '')
                pat_ok = any(p in txt for p in snowball_pats if p)
                type_ok = (h_t in seed_head_types and t_t in seed_tail_types)
                return pat_ok and type_ok

            sw_mask = snowball_pool.apply(sw_filter, axis=1)
            sw_matched = snowball_pool[sw_mask]
            sp = (sw_matched['relation'] == target_rel).mean() if len(sw_matched) else 0.0
            snowball_precs.append(sp)
            if len(sw_matched):
                high_conf = sw_matched[sw_matched['relation'] == target_rel]
                new_pats = []
                for _, r in high_conf.head(5).iterrows():
                    np_val = extract_pattern(r.get('marked_text', ''))
                    if np_val:
                        new_pats.append(np_val)
                snowball_pats = list(set(snowball_pats + new_pats))[:8]
                snowball_pool = snowball_pool[~sw_mask].reset_index(drop=True)

        # ── 시각화: Semantic Drift + F1 결과 ────────────────
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # (1) Precision 추이
        iters = list(range(1, 6))
        ax1.plot(iters, dipre_precs,    'r--o', linewidth=2, markersize=8, label='DIPRE (Semantic Drift)')
        ax1.plot(iters, snowball_precs, 'b-s',  linewidth=2, markersize=8, label='Snowball (신뢰도 필터)')
        ax1.fill_between(iters, dipre_precs, snowball_precs, alpha=0.1, color='purple',
                         label='Snowball 이득 (정밀도 보존)')
        ax1.set_xlabel("Bootstrap Iteration", fontsize=11)
        ax1.set_ylabel("Precision (정밀도)", fontsize=11)
        ax1.set_ylim(0, 1.05)
        ax1.set_xticks(iters)
        ax1.set_title(f"[{target_rel}] DIPRE vs Snowball\nSemantic Drift 비교 (Precision 변화)", fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(alpha=0.3)
        ax1.text(0.02, 0.05,
                 "DIPRE: 패턴만으로 확장 → 노이즈 축적 → Precision 하락\n"
                 "Snowball: 개체 타입 필터로 노이즈 차단 → Precision 유지",
                 transform=ax1.transAxes, fontsize=8.5,
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        # (2) 최종 Macro F1 비교
        methods_f1 = ['DIPRE\n(패턴 기반)', 'Snowball\n(신뢰도 필터)']
        scores_f1  = [macro_dipre_f1, macro_snowball_f1]
        colors_f1  = ['#ff9999', '#6699cc']
        bars2 = ax2.bar(methods_f1, scores_f1, color=colors_f1, width=0.45, edgecolor='gray')
        for b in bars2:
            ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                     f"{b.get_height():.4f}", ha='center', fontsize=13, fontweight='bold')
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("Macro F1-Score", fontsize=11)
        ax2.set_title("Semi-supervised RE\n최종 Macro F1 비교 (Silver Pool 평가)", fontsize=12)
        ax2.text(0.98, 0.97,
                 "평가 방식: 각 관계별 Gold 5-seed → Silver pool 이진 F1\n"
                 "Macro F1 = 관계별 F1의 단순 평균\n"
                 "레이블 없이도 소수 Seed만으로 관계 추출 가능",
                 transform=ax2.transAxes, fontsize=9, va='top', ha='right',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        plt.suptitle("Semi-supervised RE — DIPRE vs Snowball 부트스트래핑\n"
                     "(Gold Seed 5개 → Silver 1,205건 Pool 평가)",
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        save_fig("step2_semi_supervised")

    RESULTS['semi_dipre']    = round(macro_dipre_f1, 4)
    RESULTS['semi_snowball'] = round(macro_snowball_f1, 4)
    return macro_dipre_f1, macro_snowball_f1


# ─────────────────────────────────────────────────────────────────────
# 3. Supervised — Feature-based RF
# ─────────────────────────────────────────────────────────────────────
def step3_feature_based(gold_df, silver_df):
    print("\n" + "="*60)
    print("  STEP 3. Supervised Feature-based (Random Forest)")
    print("="*60)
    import spacy
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score, confusion_matrix
    from scipy.sparse import hstack
    from step3_feature_based_re_v2 import extract_all_features, parse_entity_type

    try:
        nlp = spacy.load('ko_core_news_sm')
    except Exception:
        nlp = None

    features_context, features_between, features_semantic, features_dep = [], [], [], []
    labels = []

    print("  Feature 추출 중 (Gold + Silver)...")
    for _, row in gold_df.iterrows():
        ctx, btw, sem, dep = extract_all_features(row['marked_text'], row['head_type'], row['tail_type'], nlp)
        features_context.append(ctx); features_between.append(btw)
        features_semantic.append(sem); features_dep.append(dep)
        labels.append(row['final_relation'])

    silver_valid = silver_df.dropna(subset=['relation'])
    silver_valid = silver_valid[
        silver_valid['relation'].apply(lambda x: bool(x) and x != 'NA')]
    for _, row in silver_valid.iterrows():
        h_type = parse_entity_type(row.get('head'))
        t_type = parse_entity_type(row.get('tail'))
        ctx, btw, sem, dep = extract_all_features(row.get('marked_text',''), h_type, t_type, nlp)
        features_context.append(ctx); features_between.append(btw)
        features_semantic.append(sem); features_dep.append(dep)
        labels.append(row['relation'])

    print(f"  총 {len(labels)}건 feature 추출 완료")

    n = len(labels)
    idx_all = list(range(n))
    idx_tr, idx_te, y_tr, y_te = train_test_split(idx_all, labels, test_size=0.2, random_state=42)

    ctx_tr = [features_context[i] for i in idx_tr]
    ctx_te = [features_context[i] for i in idx_te]
    btw_tr = [features_between[i] for i in idx_tr]
    btw_te = [features_between[i] for i in idx_te]
    sem_tr = [features_semantic[i] for i in idx_tr]
    sem_te = [features_semantic[i] for i in idx_te]
    dep_tr = [features_dep[i] for i in idx_tr]
    dep_te = [features_dep[i] for i in idx_te]

    vc = TfidfVectorizer(max_features=500).fit(ctx_tr)
    vb = TfidfVectorizer(max_features=500).fit(btw_tr)
    vs = CountVectorizer().fit(sem_tr)
    vd = TfidfVectorizer(max_features=500).fit(dep_tr)

    X_tr = hstack([vc.transform(ctx_tr), vb.transform(btw_tr),
                   vs.transform(sem_tr), vd.transform(dep_tr)])
    X_te = hstack([vc.transform(ctx_te), vb.transform(btw_te),
                   vs.transform(sem_te), vd.transform(dep_te)])

    print(f"  Random Forest 학습 중... (train {X_tr.shape[0]}건 / test {X_te.shape[0]}건)")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    macro_f1 = f1_score(y_te, y_pred, average='macro', zero_division=0)
    print(f"  Feature-based RF Macro F1: {macro_f1:.4f}")

    # ── 시각화 1: Feature Importance ──────────────────
    imps  = clf.feature_importances_
    n0 = vc.transform(ctx_tr).shape[1]
    n1 = vb.transform(btw_tr).shape[1]
    n2 = vs.transform(sem_tr).shape[1]
    grp_labels = ['Context Words\n(주변 단어)', 'Words Between\n(개체 사이 단어)',
                  'Semantic Feature\n(개체 타입 조합)', 'Dependency Path\n(구문 의존 경로)']
    grp_scores = [imps[:n0].sum(), imps[n0:n0+n1].sum(),
                  imps[n0+n1:n0+n1+n2].sum(), imps[n0+n1+n2:].sum()]
    grp_colors = sns.color_palette('pastel', 4)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    bars1 = ax1.bar(grp_labels, grp_scores, color=grp_colors, edgecolor='gray')
    for b in bars1:
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.002,
                 f"{b.get_height():.3f}", ha='center', fontsize=12, fontweight='bold')
    ax1.set_title(f"Random Forest — Feature Importance 분석\n(Macro F1={macro_f1:.4f}, {len(labels)}건 학습)",
                  fontsize=12, fontweight='bold')
    ax1.set_ylabel("Importance Score 합계", fontsize=11)
    ax1.text(0.98, 0.97,
             "핵심 피처: 개체 타입(PER, ORG, LOC 등)\n"
             "→ 행정 텍스트 특성: 관계가 개체 타입 조합으로\n   거의 결정됨 (HAS_FEE: PROGRAM|MONEY)\n"
             "SpaCy 구문 분석도 보조 기여",
             transform=ax1.transAxes, fontsize=9, va='top', ha='right',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # ── 시각화 2: Confusion Matrix ─────────────────────
    top_labels = pd.Series(y_te).value_counts().head(10).index.tolist()
    mask = np.isin(y_te, top_labels)
    cm = confusion_matrix(np.array(y_te)[mask], np.array(y_pred)[mask], labels=top_labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=top_labels, yticklabels=top_labels, ax=ax2)
    ax2.set_title(f"Confusion Matrix — Feature-based RF\n(Test {X_te.shape[0]}건, 상위 10개 관계)",
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel("예측 관계 (Predicted)", fontsize=10)
    ax2.set_ylabel("실제 관계 (Actual)", fontsize=10)
    ax2.tick_params(axis='x', rotation=40)
    plt.suptitle("Supervised ML — Feature-based Relation Extraction (Random Forest)\n"
                 "다중 언어학적 자질 (4가지 Feature Group) 조합으로 관계 분류",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_fig("step3_feature_based")

    RESULTS['sup_rf'] = round(macro_f1, 4)
    return macro_f1, clf, (vc, vb, vs, vd)


# ─────────────────────────────────────────────────────────────────────
# 4. Supervised — Kernel-based SVM
# ─────────────────────────────────────────────────────────────────────
def step4_kernel_svm(gold_df, silver_df):
    print("\n" + "="*60)
    print("  STEP 4. Supervised Kernel-based SVM (Composite Kernel)")
    print("="*60)
    import spacy
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score, confusion_matrix
    from sklearn.manifold import TSNE
    from step3c_kernel_based_re import extract_sequence_and_tree
    from visualize_kernel_ml import compute_improved_composite_kernel
    from step3_feature_based_re_v2 import parse_entity_type

    try:
        nlp = spacy.load('ko_core_news_sm')
    except Exception:
        nlp = None

    X_seq, X_tree, X_sem, labels = [], [], [], []

    for _, row in gold_df.iterrows():
        s, t = extract_sequence_and_tree(row['marked_text'], nlp)
        X_seq.append(s); X_tree.append(t)
        X_sem.append(f"{row['head_type']}|{row['tail_type']}")
        labels.append(row['final_relation'])

    silver_valid = silver_df.dropna(subset=['relation'])
    silver_valid = silver_valid[
        silver_valid['relation'].apply(lambda x: bool(x) and x != 'NA')]
    silver_sampled = silver_valid.sample(min(743, len(silver_valid)), random_state=42)
    for _, row in silver_sampled.iterrows():
        s, t = extract_sequence_and_tree(row.get('marked_text',''), nlp)
        X_seq.append(s); X_tree.append(t)
        h_type = parse_entity_type(row.get('head')); t_type = parse_entity_type(row.get('tail'))
        X_sem.append(f"{h_type}|{t_type}")
        labels.append(row['relation'])

    print(f"  총 {len(labels)}건 준비 완료")
    print("  Composite Kernel 행렬 계산 중 (O(N²))...")
    K = compute_improved_composite_kernel(X_seq, X_tree, X_sem, alpha=0.3, beta=0.3, gamma=0.4)

    N = len(labels)
    idx_all = np.arange(N)
    idx_tr, idx_te, y_tr, y_te = train_test_split(idx_all, labels, test_size=0.2, random_state=42)

    clf = SVC(kernel='precomputed', C=1.0)
    clf.fit(K[np.ix_(idx_tr, idx_tr)], y_tr)
    y_pred = clf.predict(K[np.ix_(idx_te, idx_tr)])
    macro_f1 = f1_score(y_te, y_pred, average='macro', zero_division=0)
    print(f"  Kernel SVM Macro F1: {macro_f1:.4f}")

    # ── 시각화 1: t-SNE ─────────────────────────────────
    dist = np.clip(1.0 - K, 0, None)
    np.fill_diagonal(dist, 0)
    tsne = TSNE(n_components=2, metric='precomputed', random_state=42,
                init='random', perplexity=30)
    emb2d = tsne.fit_transform(dist)
    top_rels = pd.Series(labels).value_counts().head(8).index.tolist()
    palette = sns.color_palette("tab10", len(top_rels))
    color_map = {l: palette[i] for i, l in enumerate(top_rels)}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 7))

    for lbl in top_rels:
        idx = [i for i, l in enumerate(labels) if l == lbl]
        ax1.scatter([emb2d[i,0] for i in idx], [emb2d[i,1] for i in idx],
                    c=[color_map[lbl]], label=lbl, s=20, alpha=0.7)
    others = [i for i, l in enumerate(labels) if l not in top_rels]
    if others:
        ax1.scatter([emb2d[i,0] for i in others], [emb2d[i,1] for i in others],
                    c='lightgray', s=10, alpha=0.4, label='기타')
    ax1.set_title(f"Composite Kernel t-SNE 투영\n(K_composite = 0.3·K_seq + 0.3·K_tree + 0.4·K_sem)",
                  fontsize=11, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax1.set_xlabel("t-SNE dim 1"); ax1.set_ylabel("t-SNE dim 2")

    # ── 시각화 2: Confusion Matrix ─────────────────────
    top_labels = pd.Series(y_te).value_counts().head(10).index.tolist()
    mask = np.isin(list(y_te), top_labels)
    cm = confusion_matrix(np.array(list(y_te))[mask], np.array(list(y_pred))[mask], labels=top_labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=top_labels, yticklabels=top_labels, ax=ax2)
    ax2.set_title(f"Confusion Matrix — Kernel SVM\n(Macro F1={macro_f1:.4f}, Test {len(y_te)}건)",
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel("예측 관계 (Predicted)", fontsize=10)
    ax2.set_ylabel("실제 관계 (Actual)", fontsize=10)
    ax2.tick_params(axis='x', rotation=40)
    plt.suptitle("Supervised ML — Kernel-based SVM (Composite Kernel)\n"
                 "K_composite = 0.3·K_seq (어휘) + 0.3·K_tree (구문) + 0.4·K_sem (개체 타입)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_fig("step4_kernel_svm")

    RESULTS['sup_kernel'] = round(macro_f1, 4)
    return macro_f1


# ─────────────────────────────────────────────────────────────────────
# 5. Deep Learning — Bi-LSTM + Attention
# ─────────────────────────────────────────────────────────────────────
def step5_deep_learning(gold_df, silver_df):
    print("\n" + "="*60)
    print("  STEP 5. Deep Learning (Bi-LSTM + Attention)")
    print("="*60)
    import torch, torch.nn as nn, torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score

    texts, labels = [], []
    for _, row in gold_df.iterrows():
        texts.append(row['marked_text']); labels.append(row['final_relation'])
    silver_valid = silver_df.dropna(subset=['relation'])
    silver_valid = silver_valid[
        silver_valid['relation'].apply(lambda x: bool(x) and x != 'NA')]
    for _, row in silver_valid.iterrows():
        texts.append(row.get('marked_text','')); labels.append(row['relation'])

    X_tr, X_te, y_tr, y_te = train_test_split(texts, labels, test_size=0.2, random_state=42)

    word2idx = {'<PAD>': 0, '<UNK>': 1}
    for t in X_tr:
        for w in str(t).split():
            if w not in word2idx:
                word2idx[w] = len(word2idx)

    unique_labels = sorted(set(labels))
    label2idx = {l: i for i, l in enumerate(unique_labels)}
    idx2label = {i: l for l, i in label2idx.items()}
    MAX_LEN = 60

    class REDataset(Dataset):
        def __init__(self, texts, labels):
            self.texts = texts; self.labels = labels
        def __len__(self): return len(self.texts)
        def __getitem__(self, i):
            ws  = str(self.texts[i]).split()
            seq = [word2idx.get(w, 1) for w in ws[:MAX_LEN]] + [0] * (MAX_LEN - min(len(ws), MAX_LEN))
            return torch.tensor(seq, dtype=torch.long), torch.tensor(label2idx[self.labels[i]], dtype=torch.long)

    class BiLSTMAttn(nn.Module):
        def __init__(self, V, E, H, C):
            super().__init__()
            self.emb = nn.Embedding(V, E, padding_idx=0)
            self.lstm = nn.LSTM(E, H, bidirectional=True, batch_first=True)
            self.attn_w = nn.Linear(H*2, H*2)
            self.attn_v = nn.Linear(H*2, 1, bias=False)
            self.fc = nn.Linear(H*2, C)
        def forward(self, x):
            e = self.emb(x)
            h, _ = self.lstm(e)
            a = torch.softmax(self.attn_v(torch.tanh(self.attn_w(h))).squeeze(2), dim=1)
            ctx = torch.bmm(a.unsqueeze(1), h).squeeze(1)
            return self.fc(ctx), a

    device = torch.device('mps' if torch.backends.mps.is_available() else
                          'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    tr_loader = DataLoader(REDataset(X_tr, y_tr), batch_size=32, shuffle=True)
    te_loader = DataLoader(REDataset(X_te, y_te), batch_size=32)

    model = BiLSTMAttn(len(word2idx), 128, 64, len(unique_labels)).to(device)
    opt   = optim.Adam(model.parameters(), lr=5e-3)
    crit  = nn.CrossEntropyLoss()

    epoch_losses = []
    EPOCHS = 10
    for ep in range(EPOCHS):
        model.train(); total = 0
        for seq, lbl in tr_loader:
            seq, lbl = seq.to(device), lbl.to(device)
            opt.zero_grad()
            out, _ = model(seq)
            loss = crit(out, lbl)
            loss.backward(); opt.step()
            total += loss.item()
        avg_loss = total / len(tr_loader)
        epoch_losses.append(avg_loss)
        print(f"  Epoch {ep+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for seq, lbl in te_loader:
            out, _ = model(seq.to(device))
            preds.extend(torch.argmax(out,1).cpu().numpy())
            trues.extend(lbl.numpy())
    macro_f1 = f1_score(trues, preds, average='macro', zero_division=0)
    print(f"  Bi-LSTM+Attention Macro F1: {macro_f1:.4f}")

    # ── 시각화 1: Training Curve ────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.plot(range(1, EPOCHS+1), epoch_losses, 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=11)
    ax1.set_title(f"Bi-LSTM+Attention 학습 곡선\n(Vocab={len(word2idx)}, Embed=128, Hidden=64)",
                  fontsize=12, fontweight='bold')
    ax1.grid(alpha=0.3)
    ax1.text(0.98, 0.97,
             "Scratch 학습 (사전학습 임베딩 미사용)\n"
             "→ 소량 도메인 데이터로 인해 수렴이 빠르나\n"
             "   일반화 성능에 한계 존재\n"
             "→ BERT/RoBERTa 사용 시 대폭 향상 기대",
             transform=ax1.transAxes, fontsize=9, va='top', ha='right',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # ── Attention Heatmap ─────────────────────────────
    sample_idx = next((i for i, l in enumerate(y_te) if l != 'NO_RELATION'), 0)
    sample_text  = X_te[sample_idx]
    sample_label = y_te[sample_idx]
    words = str(sample_text).split()[:MAX_LEN]
    seq   = [word2idx.get(w, 1) for w in words] + [0]*(MAX_LEN - len(words))
    with torch.no_grad():
        out, attn = model(torch.tensor([seq], dtype=torch.long).to(device))
    pred_label  = idx2label[torch.argmax(out,1).item()]
    weights = attn.cpu().numpy()[0][:len(words)]

    short_words = [w[:6] for w in words]
    sns.heatmap([weights], xticklabels=short_words, yticklabels=['Attn'],
                cmap='Reds', annot=True, fmt=".2f", cbar=False, ax=ax2)
    ax2.set_title(f"Attention Heatmap\n정답: {sample_label} | 예측: {pred_label}",
                  fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45, labelsize=8)
    plt.suptitle("Deep Learning — Bi-LSTM + Attention Mechanism\n"
                 "양방향 LSTM으로 문맥 포착, Attention으로 핵심 단어에 가중치 집중",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_fig("step5_deep_learning")

    RESULTS['dl_bilstm'] = round(macro_f1, 4)
    return macro_f1


# ─────────────────────────────────────────────────────────────────────
# 6. 최종 비교 시각화
# ─────────────────────────────────────────────────────────────────────
def step6_final_comparison():
    print("\n" + "="*60)
    print("  STEP 6. 최종 성능 비교 시각화")
    print("="*60)
    r = RESULTS
    categories = ['Unsupervised\n(V-Measure)', 'Semi-supervised\n(Macro F1)',
                  'Supervised ML\n(Macro F1)', 'Deep Learning\n(Macro F1)']
    models_pairs = [
        ('Pattern-based\n(TF-IDF)', 'Embedding-based\n(SBERT)'),
        ('DIPRE',                    'Snowball'),
        ('Feature-based (RF)',       'Kernel-based (SVM)'),
        ('Bi-LSTM + Attention',      ''),
    ]
    scores_pairs = [
        (r['unsup_pattern'],  r['unsup_embed']),
        (r['semi_dipre'],     r['semi_snowball']),
        (r['sup_rf'],         r['sup_kernel']),
        (r['dl_bilstm'],      0.0),
    ]

    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(categories))
    w = 0.35

    bar1 = ax.bar(x - w/2, [s[0] for s in scores_pairs], w,
                  color='#4c72b0', label='모델 A', edgecolor='white')
    bar2 = ax.bar(x + w/2, [s[1] for s in scores_pairs], w,
                  color='#dd8452', label='모델 B', edgecolor='white')

    def annotate(rects, names, idx):
        for i, rect in enumerate(rects):
            h = rect.get_height()
            if h > 0.005:
                ax.annotate(f'{names[i][idx]}\n{h:.4f}',
                            xy=(rect.get_x() + rect.get_width()/2, h),
                            xytext=(0, 6), textcoords='offset points',
                            ha='center', va='bottom', fontsize=9, fontweight='bold')

    annotate(bar1, models_pairs, 0)
    annotate(bar2, models_pairs, 1)

    # 패러다임별 배경 색
    bg_colors = ['#fff0f0', '#f0f8ff', '#f0fff0', '#fff8f0']
    for i, (bc, cat) in enumerate(zip(bg_colors, categories)):
        ax.axvspan(i - 0.5, i + 0.5, alpha=0.25, color=bc, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score (0~1)", fontsize=12)
    ax.legend().set_visible(False)
    ax.grid(axis='y', alpha=0.3)

    # 핵심 인사이트 텍스트
    insight = (
        "▶ Supervised ML 최고 성능 이유:\n"
        "   행정/공지 텍스트는 개체 타입이 관계를 거의 결정\n"
        "   (HAS_FEE → PROGRAM | MONEY)\n\n"
        "▶ Deep Learning 상대적 저성능:\n"
        "   사전학습 임베딩 없음 + 소량 데이터 → 일반화 한계\n"
        "   → PLM(BERT) 사용 시 최고 성능 달성 가능\n\n"
        "▶ Semi-supervised 의의:\n"
        "   레이블 없이 Seed 5개만으로 작동"
    )
    ax.text(0.98, 0.97, insight, transform=ax.transAxes, fontsize=9,
            va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    ax.set_title("Relation Extraction 파이프라인 — 방법론별 최종 성능 비교\n"
                 "OIA 데이터 (Gold 257건 + Silver 1,205건 = 1,462건)",
                 fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    save_fig("step6_final_comparison")
    save_fig("final_pipeline_comparison")

    print("\n" + "="*60)
    print("  최종 결과 요약")
    print("="*60)
    print(f"  Unsupervised  Pattern-based  V-Measure : {r['unsup_pattern']:.4f}")
    print(f"  Unsupervised  Embedding-based V-Measure: {r['unsup_embed']:.4f}")
    print(f"  Semi-supervised DIPRE    Macro F1      : {r['semi_dipre']:.4f}")
    print(f"  Semi-supervised Snowball Macro F1      : {r['semi_snowball']:.4f}")
    print(f"  Supervised RF  Macro F1                : {r['sup_rf']:.4f}")
    print(f"  Supervised Kernel SVM  Macro F1        : {r['sup_kernel']:.4f}")
    print(f"  Deep Learning Bi-LSTM  Macro F1        : {r['dl_bilstm']:.4f}")

    # JSON 저장
    with open('docs/results.json', 'w', encoding='utf-8') as f:
        json.dump(RESULTS, f, ensure_ascii=False, indent=2)
    print("\n  ✅ docs/results.json 저장 완료")


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time
    t0 = time.time()

    gold_df, silver_df, silver_valid = step0_data_overview()
    step1_unsupervised(gold_df)
    step2_semi_supervised(gold_df, silver_df)
    step3_feature_based(gold_df, silver_df)
    step4_kernel_svm(gold_df, silver_df)
    step5_deep_learning(gold_df, silver_df)
    step6_final_comparison()

    elapsed = time.time() - t0
    print(f"\n⏱️  총 실행 시간: {elapsed/60:.1f}분")
    print("🎉 전체 파이프라인 완료! docs/ 폴더에서 결과를 확인하세요.")
