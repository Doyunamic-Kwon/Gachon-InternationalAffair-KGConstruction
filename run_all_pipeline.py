"""
OIA RE Pipeline — Master Runner
Runs all 5 paradigms and saves PPT-ready figures to docs/.
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


def save_fig(name):
    path = f"docs/{name}.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ docs/{name}.png")


# ─────────────────────────────────────────────────────────────────────
# 0. Data Overview
# ─────────────────────────────────────────────────────────────────────
def step0_data_overview():
    print("\n" + "="*60)
    print("  STEP 0. Data Overview")
    print("="*60)
    from step1_data_loader import load_gold_standard, load_silver_standard
    gold_df   = load_gold_standard()
    silver_df = load_silver_standard()

    silver_valid = silver_df.dropna(subset=['relation'])
    silver_valid = silver_valid[
        silver_valid['relation'].apply(lambda x: bool(x) and x != 'NA')
    ]
    print(f"  Gold: {len(gold_df)}  Silver: {len(silver_valid)}  Total: {len(gold_df)+len(silver_valid)}")

    gold_counts = gold_df['final_relation'].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Relation distribution (Gold)
    axes[0].barh(gold_counts.index, gold_counts.values,
                 color=sns.color_palette("Set2", len(gold_counts)))
    for i, v in enumerate(gold_counts.values):
        axes[0].text(v + 0.3, i, str(v), va='center', fontsize=9, fontweight='bold')
    axes[0].set_title("Gold Standard — Relation Distribution", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Count")
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, gold_counts.max() * 1.18)

    # Gold vs Silver pie
    sizes  = [len(gold_df), len(silver_valid)]
    labels = [f'Gold\n(Human Label)\n{len(gold_df)}', f'Silver\n(LLM Label)\n{len(silver_valid)}']
    axes[1].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,
                colors=['#4c72b0', '#dd8452'], explode=(0.05, 0),
                wedgeprops=dict(edgecolor='white', linewidth=2),
                textprops={'fontsize': 11})
    axes[1].set_title("Dataset Composition (Total: 1,462)", fontsize=12, fontweight='bold')

    plt.suptitle("OIA Dataset Overview — Gachon University International Affairs",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_fig("step0_data_overview")
    return gold_df, silver_df, silver_valid


# ─────────────────────────────────────────────────────────────────────
# 1. Unsupervised RE (Gold + Silver clustering, V-Measure on Gold)
# ─────────────────────────────────────────────────────────────────────
def step1_unsupervised(gold_df, silver_valid):
    print("\n" + "="*60)
    print("  STEP 1. Unsupervised RE (Gold + Silver)")
    print("="*60)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.metrics import v_measure_score
    from sentence_transformers import SentenceTransformer
    from sklearn.manifold import TSNE

    # Gold + Silver combined for clustering
    gold_texts  = gold_df['marked_text'].fillna('').tolist()
    silver_texts = silver_valid['marked_text'].fillna('').tolist() if len(silver_valid) else []
    all_texts   = gold_texts + silver_texts

    gold_labels   = gold_df['final_relation'].tolist()
    silver_labels = silver_valid['relation'].tolist() if len(silver_valid) else []
    all_labels    = gold_labels + silver_labels

    k = gold_df['final_relation'].nunique()
    n_gold = len(gold_df)
    print(f"  Cluster on {len(all_texts)} texts (Gold {n_gold} + Silver {len(silver_texts)}), K={k}")

    # ── Pattern-based ──────────────────────────────────
    patterns = []
    for t in all_texts:
        m = re.search(r'\[/E1\](.*?)\[E2\]', t) or re.search(r'\[/E2\](.*?)\[E1\]', t)
        patterns.append(re.sub(r'<[^>]+>', ' ', m.group(1)).strip() if m else t)

    X_pat = TfidfVectorizer(max_features=500).fit_transform(patterns)
    y_pat = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_pat)
    # V-Measure on Gold only
    v_pat = v_measure_score(gold_labels, y_pat[:n_gold])
    print(f"  Pattern-based  V-Measure (Gold): {v_pat:.4f}")

    # ── Embedding-based ────────────────────────────────
    print("  SBERT encoding (paraphrase-multilingual-MiniLM-L12-v2)...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    X_emb = model.encode(all_texts, batch_size=64, show_progress_bar=False)
    y_emb = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_emb)
    v_emb = v_measure_score(gold_labels, y_emb[:n_gold])
    print(f"  Embedding-based V-Measure (Gold): {v_emb:.4f}")

    # ── Bar Chart ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(['Pattern-based\n(Lexical TF-IDF)', 'Embedding-based\n(Sentence-BERT)'],
                  [v_pat, v_emb],
                  color=['#ffb3b3', '#99ccff'], width=0.45, edgecolor='#666666', linewidth=0.8)
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.012,
                f"{b.get_height():.4f}", ha='center', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_ylabel("V-Measure Score", fontsize=11)
    ax.set_title("Unsupervised RE — Clustering Performance (V-Measure)\n"
                 f"Train: Gold {n_gold} + Silver {len(silver_texts)} | Eval: Gold {n_gold}",
                 fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    sns.despine()
    plt.tight_layout()
    save_fig("step1_unsupervised_comparison")

    # ── t-SNE (Gold only for clarity) ─────────────────
    print("  t-SNE projection (Gold)...")
    all_labels_set = sorted(set(gold_labels))
    palette    = sns.color_palette("tab10", len(all_labels_set))
    color_map  = {l: palette[i] for i, l in enumerate(all_labels_set)}

    tsne = TSNE(n_components=2, random_state=42, init='random', perplexity=25, max_iter=1000)
    gold_emb2d = tsne.fit_transform(X_emb[:n_gold])
    gold_pat2d = tsne.fit_transform(X_pat[:n_gold].toarray())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for lbl in all_labels_set:
        idx = [i for i, l in enumerate(gold_labels) if l == lbl]
        ax1.scatter([gold_pat2d[i,0] for i in idx], [gold_pat2d[i,1] for i in idx],
                    c=[color_map[lbl]], label=lbl, s=30, alpha=0.75, edgecolors='none')
        ax2.scatter([gold_emb2d[i,0] for i in idx], [gold_emb2d[i,1] for i in idx],
                    c=[color_map[lbl]], label=lbl, s=30, alpha=0.75, edgecolors='none')

    ax1.set_title(f"Pattern-based (TF-IDF)  V-Measure: {v_pat:.4f}", fontsize=12, fontweight='bold')
    ax2.set_title(f"Embedding-based (SBERT)  V-Measure: {v_emb:.4f}", fontsize=12, fontweight='bold')
    for ax in (ax1, ax2):
        ax.set_xlabel("t-SNE dim 1", fontsize=10); ax.set_ylabel("t-SNE dim 2", fontsize=10)
        sns.despine(ax=ax)
    handles = [plt.Line2D([0],[0], marker='o', color='w',
                           markerfacecolor=color_map[l], markersize=8, label=l)
               for l in all_labels_set]
    ax2.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc='upper left',
               fontsize=8, title="Relation")
    plt.suptitle("Unsupervised RE — t-SNE Cluster Visualization (Gold 257)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_fig("step1_unsupervised_tsne")

    RESULTS['unsup_pattern'] = round(v_pat, 4)
    RESULTS['unsup_embed']   = round(v_emb, 4)
    return v_pat, v_emb


# ─────────────────────────────────────────────────────────────────────
# 2. Semi-supervised RE (DIPRE + Snowball, 5-iteration bootstrap)
# ─────────────────────────────────────────────────────────────────────
def step2_semi_supervised(gold_df, silver_valid):
    print("\n" + "="*60)
    print("  STEP 2. Semi-supervised RE (DIPRE & Snowball)")
    print("="*60)
    from step3b_semi_supervised import (
        run_dipre_and_snowball,
        extract_text_pattern as extract_pattern,
        _extract_context_before_e1,
        _parse_type,
        load_corpus_unlabeled,
        norm_rel,
    )
    from collections import defaultdict

    macro_dipre_f1, macro_snowball_f1 = run_dipre_and_snowball(n_seeds=10)

    # ── Per-iteration precision simulation for HAS_FEE ────────
    target_rel = 'HAS_FEE'
    # Normalize gold relation labels first
    gold_df_norm = gold_df.copy()
    gold_df_norm['final_relation'] = gold_df_norm['final_relation'].apply(norm_rel)

    if target_rel in gold_df_norm['final_relation'].values:
        seeds = gold_df_norm[gold_df_norm['final_relation'] == target_rel].head(10)

        def build_patterns(seed_df):
            pats = set()
            for _, r in seed_df.iterrows():
                p = extract_pattern(r.get('marked_text',''))
                if p: pats.add(p)
                ctx = _extract_context_before_e1(r.get('marked_text',''))
                if ctx and len(ctx) >= 2: pats.add(ctx)
            return list(pats)

        seed_head_types = set(seeds['head_type'].dropna())
        seed_tail_types = set(seeds['tail_type'].dropna())

        # Use rebuilt corpus as the unlabeled pool
        corpus_rows = load_corpus_unlabeled()
        corpus_pool  = corpus_rows[:]  # working copy (list of dicts)
        dipre_seeds    = seeds.copy()
        snowball_seeds = seeds.copy()
        dipre_precs    = []
        snowball_precs = []
        dipre_sizes    = []
        snowball_sizes = []

        remaining_dipre    = corpus_pool[:]
        remaining_snowball = corpus_pool[:]

        for it in range(1, 6):
            dipre_pats    = build_patterns(dipre_seeds)
            snowball_pats = build_patterns(snowball_seeds)

            # DIPRE
            if dipre_pats:
                d_match = [r for r in remaining_dipre
                           if any(p in str(r.get('marked_text','')) for p in dipre_pats)]
                dp = (sum(1 for r in d_match if norm_rel(r.get('true_relation','')) == target_rel)
                      / max(len(d_match), 1))
                dipre_precs.append(dp)
                dipre_sizes.append(len(d_match))
                # Add ALL matched (noisy) → Semantic Drift
                match_ids = {r.get('id') for r in d_match}
                remaining_dipre = [r for r in remaining_dipre if r.get('id') not in match_ids]
            else:
                dipre_precs.append(0.0); dipre_sizes.append(0)

            # Snowball (entity type filter)
            if snowball_pats:
                def sw_filter(row):
                    txt   = str(row.get('marked_text',''))
                    h_t   = _parse_type(row.get('head', {}))
                    t_t   = _parse_type(row.get('tail', {}))
                    return (any(p in txt for p in snowball_pats)
                            and h_t in seed_head_types
                            and t_t in seed_tail_types)
                sw_match = [r for r in remaining_snowball if sw_filter(r)]
                sp = (sum(1 for r in sw_match if norm_rel(r.get('true_relation','')) == target_rel)
                      / max(len(sw_match), 1))
                snowball_precs.append(sp)
                snowball_sizes.append(len(sw_match))
                match_ids = {r.get('id') for r in sw_match}
                remaining_snowball = [r for r in remaining_snowball if r.get('id') not in match_ids]
            else:
                snowball_precs.append(0.0); snowball_sizes.append(0)

            print(f"  Iter {it}: DIPRE prec={dipre_precs[-1]:.3f} (n={dipre_sizes[-1]}) | "
                  f"Snowball prec={snowball_precs[-1]:.3f} (n={snowball_sizes[-1]})")

        # ── Precision Drift plot ─────────────────────────
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        iters = list(range(1, 6))
        ax1.plot(iters, dipre_precs,    'r--o', linewidth=2, markersize=8, label='DIPRE')
        ax1.plot(iters, snowball_precs, 'b-s',  linewidth=2, markersize=8, label='Snowball')
        ax1.fill_between(iters, dipre_precs, snowball_precs, alpha=0.12, color='blue')
        ax1.set_xlabel("Bootstrap Iteration", fontsize=11)
        ax1.set_ylabel("Precision", fontsize=11)
        ax1.set_ylim(-0.05, 1.05)
        ax1.set_xticks(iters)
        ax1.set_title(f"Semantic Drift — Precision over Iterations\n({target_rel})",
                      fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(alpha=0.3)
        sns.despine(ax=ax1)

        # ── Final F1 bar ──────────────────────────────────
        bars2 = ax2.bar(['DIPRE\n(Pattern only)', 'Snowball\n(+ Entity Type Filter)'],
                        [macro_dipre_f1, macro_snowball_f1],
                        color=['#ff9999', '#6699cc'], width=0.45,
                        edgecolor='#666666', linewidth=0.8)
        for b in bars2:
            ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.006,
                     f"{b.get_height():.4f}", ha='center', fontsize=13, fontweight='bold')
        ax2.set_ylim(0, 0.7)
        ax2.set_ylabel("Macro F1-Score", fontsize=11)
        ax2.set_title("Semi-supervised RE — Final Macro F1\n(Binary F1 per relation, Gold 10-seed)",
                      fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        sns.despine(ax=ax2)

        plt.suptitle("Semi-supervised RE — DIPRE vs Snowball Bootstrapping",
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        save_fig("step2_semi_supervised")

    RESULTS['semi_dipre']    = round(macro_dipre_f1, 4)
    RESULTS['semi_snowball'] = round(macro_snowball_f1, 4)
    return macro_dipre_f1, macro_snowball_f1


# ─────────────────────────────────────────────────────────────────────
# 3. Supervised Feature-based RF
# ─────────────────────────────────────────────────────────────────────
def step3_feature_based(gold_df, silver_valid):
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

    feats_ctx, feats_btw, feats_sem, feats_dep, labels = [], [], [], [], []

    for _, row in gold_df.iterrows():
        ctx, btw, sem, dep = extract_all_features(row['marked_text'], row['head_type'], row['tail_type'], nlp)
        feats_ctx.append(ctx); feats_btw.append(btw)
        feats_sem.append(sem); feats_dep.append(dep)
        labels.append(row['final_relation'])

    for _, row in silver_valid.iterrows():
        h_type = parse_entity_type(row.get('head')); t_type = parse_entity_type(row.get('tail'))
        ctx, btw, sem, dep = extract_all_features(row.get('marked_text',''), h_type, t_type, nlp)
        feats_ctx.append(ctx); feats_btw.append(btw)
        feats_sem.append(sem); feats_dep.append(dep)
        labels.append(row['relation'])

    print(f"  {len(labels)} samples")

    n = len(labels)
    idx_tr, idx_te, y_tr, y_te = train_test_split(
        list(range(n)), labels, test_size=0.2, random_state=42)

    vc = TfidfVectorizer(max_features=500).fit([feats_ctx[i] for i in idx_tr])
    vb = TfidfVectorizer(max_features=500).fit([feats_btw[i] for i in idx_tr])
    vs = CountVectorizer().fit([feats_sem[i] for i in idx_tr])
    vd = TfidfVectorizer(max_features=500).fit([feats_dep[i] for i in idx_tr])

    def tx(idx):
        return hstack([vc.transform([feats_ctx[i] for i in idx]),
                       vb.transform([feats_btw[i] for i in idx]),
                       vs.transform([feats_sem[i] for i in idx]),
                       vd.transform([feats_dep[i] for i in idx])])

    X_tr, X_te = tx(idx_tr), tx(idx_te)
    print(f"  RF training ({X_tr.shape[0]} train / {X_te.shape[0]} test)...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    y_pred  = clf.predict(X_te)
    macro_f1 = f1_score(y_te, y_pred, average='macro', zero_division=0)
    print(f"  Macro F1: {macro_f1:.4f}")

    # Feature Importance
    imps = clf.feature_importances_
    n0 = vc.transform([feats_ctx[0]]).shape[1]
    n1 = vb.transform([feats_btw[0]]).shape[1]
    n2 = vs.transform([feats_sem[0]]).shape[1]
    grp_names  = ['Context\nWords', 'Words\nBetween', 'Semantic\nFeature', 'Dep\nPath']
    grp_scores = [imps[:n0].sum(), imps[n0:n0+n1].sum(),
                  imps[n0+n1:n0+n1+n2].sum(), imps[n0+n1+n2:].sum()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = sns.color_palette('pastel', 4)
    b = ax1.bar(grp_names, grp_scores, color=colors, edgecolor='#666666', linewidth=0.8)
    for bar in b:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                 f"{bar.get_height():.3f}", ha='center', fontsize=12, fontweight='bold')
    ax1.set_title(f"Feature Importance (RF)\nMacro F1 = {macro_f1:.4f} | n={len(labels)}",
                  fontsize=12, fontweight='bold')
    ax1.set_ylabel("Importance Sum", fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    sns.despine(ax=ax1)

    top_labels = pd.Series(y_te).value_counts().head(10).index.tolist()
    mask = np.isin(y_te, top_labels)
    cm = confusion_matrix(np.array(y_te)[mask], np.array(y_pred)[mask], labels=top_labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=top_labels, yticklabels=top_labels, ax=ax2)
    ax2.set_title(f"Confusion Matrix — RF Classifier\n(Test {X_te.shape[0]}, Top-10 relations)",
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel("Predicted", fontsize=10); ax2.set_ylabel("Actual", fontsize=10)
    ax2.tick_params(axis='x', rotation=40, labelsize=8)
    ax2.tick_params(axis='y', rotation=0, labelsize=8)

    plt.suptitle("Supervised ML — Feature-based RE (Random Forest)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_fig("step3_feature_based")

    RESULTS['sup_rf'] = round(macro_f1, 4)
    return macro_f1


# ─────────────────────────────────────────────────────────────────────
# 4. Supervised Kernel SVM
# ─────────────────────────────────────────────────────────────────────
def step4_kernel_svm(gold_df, silver_valid):
    print("\n" + "="*60)
    print("  STEP 4. Supervised Kernel SVM (Composite Kernel)")
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

    silver_sampled = silver_valid.sample(min(743, len(silver_valid)), random_state=42)
    for _, row in silver_sampled.iterrows():
        s, t = extract_sequence_and_tree(row.get('marked_text',''), nlp)
        X_seq.append(s); X_tree.append(t)
        h_type = parse_entity_type(row.get('head')); t_type = parse_entity_type(row.get('tail'))
        X_sem.append(f"{h_type}|{t_type}")
        labels.append(row['relation'])

    print(f"  {len(labels)} samples — computing composite kernel (O(N²))...")
    K = compute_improved_composite_kernel(X_seq, X_tree, X_sem, alpha=0.3, beta=0.3, gamma=0.4)

    N = len(labels)
    idx_tr, idx_te, y_tr, y_te = train_test_split(
        np.arange(N), labels, test_size=0.2, random_state=42)

    clf = SVC(kernel='precomputed', C=1.0)
    clf.fit(K[np.ix_(idx_tr, idx_tr)], y_tr)
    y_pred = clf.predict(K[np.ix_(idx_te, idx_tr)])
    macro_f1 = f1_score(y_te, y_pred, average='macro', zero_division=0)
    print(f"  Kernel SVM Macro F1: {macro_f1:.4f}")

    dist = np.clip(1.0 - K, 0, None); np.fill_diagonal(dist, 0)
    tsne = TSNE(n_components=2, metric='precomputed', random_state=42, init='random', perplexity=30)
    emb2d = tsne.fit_transform(dist)

    top_rels = pd.Series(labels).value_counts().head(8).index.tolist()
    palette  = sns.color_palette("tab10", len(top_rels))
    color_map = {l: palette[i] for i, l in enumerate(top_rels)}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    for lbl in top_rels:
        idx = [i for i, l in enumerate(labels) if l == lbl]
        ax1.scatter([emb2d[i,0] for i in idx], [emb2d[i,1] for i in idx],
                    c=[color_map[lbl]], label=lbl, s=20, alpha=0.75, edgecolors='none')
    others = [i for i, l in enumerate(labels) if l not in top_rels]
    if others:
        ax1.scatter([emb2d[i,0] for i in others], [emb2d[i,1] for i in others],
                    c='lightgray', s=12, alpha=0.4)
    ax1.set_title("Composite Kernel — t-SNE Projection\n"
                  r"K = 0.3·K$_{seq}$ + 0.3·K$_{tree}$ + 0.4·K$_{sem}$",
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel("t-SNE dim 1"); ax1.set_ylabel("t-SNE dim 2")
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8, title="Relation")
    sns.despine(ax=ax1)

    top_labels = pd.Series(y_te).value_counts().head(10).index.tolist()
    mask = np.isin(list(y_te), top_labels)
    cm = confusion_matrix(np.array(list(y_te))[mask], np.array(list(y_pred))[mask], labels=top_labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=top_labels, yticklabels=top_labels, ax=ax2)
    ax2.set_title(f"Confusion Matrix — Kernel SVM\n(Macro F1 = {macro_f1:.4f}, Test {len(y_te)})",
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel("Predicted"); ax2.set_ylabel("Actual")
    ax2.tick_params(axis='x', rotation=40, labelsize=8)
    ax2.tick_params(axis='y', rotation=0, labelsize=8)

    plt.suptitle("Supervised ML — Kernel-based SVM (Composite Kernel)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_fig("step4_kernel_svm")

    RESULTS['sup_kernel'] = round(macro_f1, 4)
    return macro_f1


# ─────────────────────────────────────────────────────────────────────
# 5. Deep Learning — Bi-LSTM + Attention
# ─────────────────────────────────────────────────────────────────────
def step5_deep_learning(gold_df, silver_valid):
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
            seq = [word2idx.get(w, 1) for w in ws[:MAX_LEN]] + [0]*(MAX_LEN - min(len(ws), MAX_LEN))
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
            return self.fc(torch.bmm(a.unsqueeze(1), h).squeeze(1)), a

    device = torch.device('mps' if torch.backends.mps.is_available() else
                          'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    tr_loader = DataLoader(REDataset(X_tr, y_tr), batch_size=32, shuffle=True)
    te_loader = DataLoader(REDataset(X_te, y_te), batch_size=32)

    model = BiLSTMAttn(len(word2idx), 128, 64, len(unique_labels)).to(device)
    opt   = optim.Adam(model.parameters(), lr=5e-3)
    crit  = nn.CrossEntropyLoss()

    EPOCHS = 10
    epoch_losses = []
    for ep in range(EPOCHS):
        model.train(); total = 0
        for seq, lbl in tr_loader:
            seq, lbl = seq.to(device), lbl.to(device)
            opt.zero_grad()
            out, _ = model(seq)
            loss = crit(out, lbl); loss.backward(); opt.step()
            total += loss.item()
        avg = total / len(tr_loader)
        epoch_losses.append(avg)
        print(f"  Epoch {ep+1}/{EPOCHS} | Loss: {avg:.4f}")

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for seq, lbl in te_loader:
            out, _ = model(seq.to(device))
            preds.extend(torch.argmax(out,1).cpu().numpy())
            trues.extend(lbl.numpy())
    macro_f1 = f1_score(trues, preds, average='macro', zero_division=0)
    print(f"  Macro F1: {macro_f1:.4f}")

    # Training curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(range(1, EPOCHS+1), epoch_losses, 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel("Epoch", fontsize=11); ax1.set_ylabel("Cross-Entropy Loss", fontsize=11)
    ax1.set_title(f"Bi-LSTM+Attention — Training Curve\nVocab={len(word2idx)}, E=128, H=64",
                  fontsize=12, fontweight='bold')
    ax1.grid(alpha=0.3); sns.despine(ax=ax1)

    # Attention Heatmap
    sample_idx   = next((i for i, l in enumerate(y_te) if l != 'NO_RELATION'), 0)
    sample_text  = X_te[sample_idx]
    sample_label = y_te[sample_idx]
    words = str(sample_text).split()[:MAX_LEN]
    seq   = [word2idx.get(w, 1) for w in words] + [0]*(MAX_LEN - len(words))
    with torch.no_grad():
        out, attn = model(torch.tensor([seq], dtype=torch.long).to(device))
    pred_label = idx2label[torch.argmax(out,1).item()]
    weights    = attn.cpu().numpy()[0][:len(words)]
    short_words = [w[:7] for w in words]
    sns.heatmap([weights], xticklabels=short_words, yticklabels=['Attn'],
                cmap='Reds', annot=True, fmt=".2f", cbar=False, ax=ax2)
    ax2.set_title(f"Attention Heatmap\nTrue: {sample_label}  |  Pred: {pred_label}",
                  fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45, labelsize=8)

    plt.suptitle(f"Deep Learning — Bi-LSTM + Attention (Scratch)  Macro F1 = {macro_f1:.4f}",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_fig("step5_deep_learning")

    RESULTS['dl_bilstm'] = round(macro_f1, 4)
    return macro_f1


# ─────────────────────────────────────────────────────────────────────
# 6. Final Comparison
# ─────────────────────────────────────────────────────────────────────
def step6_final_comparison():
    print("\n" + "="*60)
    print("  STEP 6. Final Comparison")
    print("="*60)
    r = RESULTS
    categories = ['Unsupervised\n(V-Measure)', 'Semi-supervised\n(Macro F1)',
                  'Supervised ML\n(Macro F1)', 'Deep Learning\n(Macro F1)']
    model_pairs = [
        ('Pattern-based', 'Embedding-based'),
        ('DIPRE', 'Snowball'),
        ('Feature RF', 'Kernel SVM'),
        ('Bi-LSTM+Attn', ''),
    ]
    score_pairs = [
        (r['unsup_pattern'], r['unsup_embed']),
        (r['semi_dipre'],    r['semi_snowball']),
        (r['sup_rf'],        r['sup_kernel']),
        (r['dl_bilstm'],     0.0),
    ]

    fig, ax = plt.subplots(figsize=(13, 7))
    x, w = np.arange(len(categories)), 0.35

    bar1 = ax.bar(x - w/2, [s[0] for s in score_pairs], w,
                  color='#4c72b0', edgecolor='white', linewidth=0.8)
    bar2 = ax.bar(x + w/2, [s[1] for s in score_pairs], w,
                  color='#dd8452', edgecolor='white', linewidth=0.8)

    def annotate(rects, names, idx):
        for i, rect in enumerate(rects):
            h = rect.get_height()
            if h > 0.01:
                ax.annotate(f'{names[i][idx]}\n{h:.4f}',
                            xy=(rect.get_x() + rect.get_width()/2, h),
                            xytext=(0, 5), textcoords='offset points',
                            ha='center', va='bottom', fontsize=9, fontweight='bold')

    annotate(bar1, model_pairs, 0)
    annotate(bar2, model_pairs, 1)

    bg_colors = ['#fff0f0', '#f0f8ff', '#f0fff0', '#fff8f0']
    for i, bc in enumerate(bg_colors):
        ax.axvspan(i - 0.5, i + 0.5, alpha=0.2, color=bc, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.legend().set_visible(False)
    sns.despine(ax=ax)
    ax.set_title("OIA Relation Extraction — Performance Comparison\n"
                 "Gold 257 + Silver 1,205 = 1,462 samples | 12 relation types",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_fig("step6_final_comparison")
    save_fig("final_pipeline_comparison")

    print("\n  === Final Results ===")
    for k, v in r.items():
        print(f"  {k:25s}: {v:.4f}")

    with open('docs/results.json', 'w', encoding='utf-8') as f:
        json.dump(RESULTS, f, ensure_ascii=False, indent=2)
    print("  ✅ docs/results.json")


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time; t0 = time.time()

    # 코퍼스 파일이 없으면 자동 생성 (step0_rebuild_corpus.py)
    if not os.path.exists("data/re_fixed_v6/corpus_unlabeled.jsonl"):
        print("corpus_unlabeled.jsonl 없음 → step0_rebuild_corpus.py 실행 중...")
        from step0_rebuild_corpus import rebuild_corpus
        rebuild_corpus(use_openai=False)

    gold_df, silver_df, silver_valid = step0_data_overview()
    step1_unsupervised(gold_df, silver_valid)
    step2_semi_supervised(gold_df, silver_valid)
    step3_feature_based(gold_df, silver_valid)
    step4_kernel_svm(gold_df, silver_valid)
    step5_deep_learning(gold_df, silver_valid)
    step6_final_comparison()

    print(f"\n⏱  Total: {(time.time()-t0)/60:.1f} min")
    print("🎉 Done! Check docs/")
