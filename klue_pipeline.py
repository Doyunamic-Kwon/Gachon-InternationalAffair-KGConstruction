"""
KLUE-RE 전체 파이프라인
기존 스크립트 함수들을 최대한 재사용합니다.
"""
import re
import spacy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, v_measure_score
from sklearn.manifold import TSNE
from scipy.sparse import hstack
from sentence_transformers import SentenceTransformer
from klue_data_loader import load_klue_re

# 기존 스크립트 함수 재사용
from step3_feature_based_re_v2 import extract_all_features
from step3c_kernel_based_re import extract_sequence_and_tree
from visualize_kernel_ml import compute_improved_composite_kernel

plt.rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False
SAVE_DIR = "docs"

# ─────────────────────────────────────────────
# 1. UNSUPERVISED
# ─────────────────────────────────────────────
def run_klue_unsupervised(train_df):
    print("\n=== [KLUE] 1. Unsupervised ===")
    texts       = train_df['marked_text'].tolist()
    labels_true = train_df['final_relation'].tolist()
    k = train_df['final_relation'].nunique()

    # Pattern-based
    patterns = []
    for t in texts:
        m = re.search(r'\[/E1\](.*?)\[E2\]', t) or re.search(r'\[/E2\](.*?)\[E1\]', t)
        patterns.append(m.group(1).strip() if m else t)

    X_pat = TfidfVectorizer(max_features=500).fit_transform(patterns)
    y_pat = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_pat)
    v_pat = v_measure_score(labels_true, y_pat)
    print(f"  Pattern-based  V-Measure: {v_pat:.4f}")

    # Embedding-based
    print("  SBERT 임베딩 중...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    X_emb = model.encode(texts, batch_size=64, show_progress_bar=True)
    y_emb = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_emb)
    v_emb = v_measure_score(labels_true, y_emb)
    print(f"  Embedding-based V-Measure: {v_emb:.4f}")

    # 시각화 1 — Bar
    plt.figure(figsize=(8, 5))
    bars = plt.bar(['Pattern-based\n(TF-IDF)', 'Embedding-based\n(SBERT)'],
                   [v_pat, v_emb], color=['#ffb3b3', '#99ccff'], width=0.45)
    for b in bars:
        plt.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                 f"{b.get_height():.4f}", ha='center', fontweight='bold')
    plt.title("[KLUE-RE] Unsupervised 군집화 성능 비교 (V-Measure)", fontsize=14)
    plt.ylabel("V-Measure Score")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/klue_unsupervised_comparison.png", dpi=300)
    plt.close()

    # 시각화 2 — t-SNE (샘플 3000건)
    print("  t-SNE 차원 축소 중...")
    idx = np.random.choice(len(texts), 3000, replace=False)
    tsne = TSNE(n_components=2, random_state=42, init='random', perplexity=50)
    emb2d = tsne.fit_transform(X_emb[idx])
    pat2d = tsne.fit_transform(X_pat[idx].toarray())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    palette = sns.color_palette("husl", k)
    sns.scatterplot(x=pat2d[:,0], y=pat2d[:,1], hue=np.array(y_pat)[idx],
                    palette=palette, ax=ax1, legend=False, s=15, alpha=0.6)
    ax1.set_title(f"Pattern-based (TF-IDF)\nV-Measure: {v_pat:.4f}", fontsize=13)
    sns.scatterplot(x=emb2d[:,0], y=emb2d[:,1], hue=np.array(y_emb)[idx],
                    palette=palette, ax=ax2, legend=False, s=15, alpha=0.6)
    ax2.set_title(f"Embedding-based (SBERT)\nV-Measure: {v_emb:.4f}", fontsize=13)
    plt.suptitle("[KLUE-RE] Unsupervised 군집화 t-SNE 시각화 (3,000건 샘플)", fontsize=15)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/klue_unsupervised_tsne.png", dpi=300)
    plt.close()
    print("  ✅ klue_unsupervised_*.png 저장 완료")
    return v_pat, v_emb


# ─────────────────────────────────────────────
# 2. SUPERVISED — Feature-based
# ─────────────────────────────────────────────
def run_klue_feature_based(train_df, test_df):
    print("\n=== [KLUE] 2. Supervised — Feature-based ===")
    try:
        nlp = spacy.load('ko_core_news_sm')
    except Exception:
        nlp = None

    def build_feats(df):
        ctx_l, btw_l, sem_l, dep_l, lbl_l = [], [], [], [], []
        for _, row in df.iterrows():
            c, b, s, d = extract_all_features(row['marked_text'], row['head_type'], row['tail_type'], nlp)
            ctx_l.append(c); btw_l.append(b); sem_l.append(s); dep_l.append(d)
            lbl_l.append(row['final_relation'])
        return ctx_l, btw_l, sem_l, dep_l, lbl_l

    print("  Train feature 추출 중...")
    ctx_tr, btw_tr, sem_tr, dep_tr, y_tr = build_feats(train_df)
    print("  Test  feature 추출 중...")
    ctx_te, btw_te, sem_te, dep_te, y_te = build_feats(test_df)

    vc = TfidfVectorizer(max_features=1000).fit(ctx_tr)
    vb = TfidfVectorizer(max_features=1000).fit(btw_tr)
    vs = CountVectorizer().fit(sem_tr)
    vd = TfidfVectorizer(max_features=500).fit(dep_tr)

    X_tr = hstack([vc.transform(ctx_tr), vb.transform(btw_tr),
                   vs.transform(sem_tr), vd.transform(dep_tr)])
    X_te = hstack([vc.transform(ctx_te), vb.transform(btw_te),
                   vs.transform(sem_te), vd.transform(dep_te)])

    print("  Random Forest 학습 중...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    f1 = f1_score(y_te, y_pred, average='macro', zero_division=0)
    print(f"  Feature-based RF Macro F1: {f1:.4f}")

    # Feature Importance 시각화
    imps = clf.feature_importances_
    n0 = vc.transform(ctx_tr).shape[1]
    n1 = vb.transform(btw_tr).shape[1]
    n2 = vs.transform(sem_tr).shape[1]
    n3 = vd.transform(dep_tr).shape[1]
    groups = ['Context\nWords', 'Words\nBetween', 'Semantic\nFeature', 'Dependency\nPath']
    scores = [imps[:n0].sum(), imps[n0:n0+n1].sum(),
              imps[n0+n1:n0+n1+n2].sum(), imps[n0+n1+n2:].sum()]

    plt.figure(figsize=(9, 5))
    bars = plt.bar(groups, scores, color=sns.color_palette('pastel'))
    for b in bars:
        plt.text(b.get_x()+b.get_width()/2, b.get_height()+0.002,
                 f"{b.get_height():.3f}", ha='center', fontweight='bold')
    plt.title("[KLUE-RE] Feature Importance (Random Forest)", fontsize=14)
    plt.ylabel("Importance Score 합계")
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/klue_feature_importance.png", dpi=300)
    plt.close()

    # Confusion Matrix (상위 12 라벨)
    top12 = pd.Series(y_te).value_counts().head(12).index.tolist()
    mask  = np.isin(y_te, top12)
    cm = confusion_matrix(np.array(y_te)[mask], np.array(y_pred)[mask], labels=top12)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=top12, yticklabels=top12)
    plt.title(f"[KLUE-RE] Feature-based RF Confusion Matrix (F1={f1:.4f})", fontsize=14)
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/klue_feature_confusion_matrix.png", dpi=300)
    plt.close()
    print("  ✅ klue_feature_*.png 저장 완료")
    return f1


# ─────────────────────────────────────────────
# 3. SUPERVISED — Kernel-based (샘플 1000건)
# ─────────────────────────────────────────────
def run_klue_kernel(train_df, test_df):
    print("\n=== [KLUE] 3. Supervised — Kernel-based (1,000건 샘플) ===")
    try:
        nlp = spacy.load('ko_core_news_sm')
    except Exception:
        nlp = None

    # train_df에서만 1000건 샘플 (test_df 오염 방지)
    combined = train_df.sample(min(1000, len(train_df)), random_state=42).reset_index(drop=True)
    labels_all = combined['final_relation'].tolist()

    X_seq, X_tree, X_sem = [], [], []
    for _, row in combined.iterrows():
        s, t = extract_sequence_and_tree(row['marked_text'], nlp)
        X_seq.append(s); X_tree.append(t)
        X_sem.append(f"{row['head_type']}|{row['tail_type']}")

    print("  N×N Composite Kernel 계산 중...")
    K = compute_improved_composite_kernel(X_seq, X_tree, X_sem, alpha=0.3, beta=0.3, gamma=0.4)

    idx_all = np.arange(1000)
    idx_tr, idx_te, y_tr, y_te = train_test_split(idx_all, labels_all, test_size=0.2, random_state=42)
    clf = SVC(kernel='precomputed', C=1.0)
    clf.fit(K[np.ix_(idx_tr, idx_tr)], y_tr)
    y_pred = clf.predict(K[np.ix_(idx_te, idx_tr)])
    f1 = f1_score(y_te, y_pred, average='macro', zero_division=0)
    print(f"  Kernel SVM Macro F1: {f1:.4f}")

    # t-SNE
    dist = 1.0 - K
    tsne = TSNE(n_components=2, metric='precomputed', random_state=42, init='random')
    emb2d = tsne.fit_transform(dist)
    top5 = pd.Series(labels_all).value_counts().head(5).index.tolist()
    plt.figure(figsize=(9, 7))
    for lbl in top5:
        idx = [i for i, l in enumerate(labels_all) if l == lbl]
        plt.scatter(emb2d[idx,0], emb2d[idx,1], label=lbl, alpha=0.7, s=20)
    plt.title(f"[KLUE-RE] Kernel Matrix t-SNE (F1={f1:.4f}, 1,000건)", fontsize=14)
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/klue_kernel_tsne.png", dpi=300)
    plt.close()
    print("  ✅ klue_kernel_tsne.png 저장 완료")
    return f1


# ─────────────────────────────────────────────
# 4. SEMI-SUPERVISED (DIPRE + Snowball 시뮬레이션)
# ─────────────────────────────────────────────
def run_klue_semi_supervised(train_df, test_df):
    print("\n=== [KLUE] 4. Semi-supervised (DIPRE / Snowball) ===")
    from collections import defaultdict

    target_rel = 'per:employee_of'
    seed_df    = train_df[train_df['final_relation'] == target_rel].head(10)

    # Pool: seed를 제외한 나머지 전체 train (target_rel 포함 — 제거하면 precision이 항상 0)
    pool_df = train_df.drop(seed_df.index).sample(min(3000, len(train_df) - len(seed_df)),
                                                   random_state=42).reset_index(drop=True)

    def extract_top_patterns(seed_frame, n=5):
        pats = defaultdict(int)
        for _, row in seed_frame.iterrows():
            m = re.search(r'\[/E1\](.*?)\[E2\]', str(row['marked_text']))
            if not m:
                m = re.search(r'\[/E2\](.*?)\[E1\]', str(row['marked_text']))
            if m:
                pats[m.group(1).strip()[:30]] += 1
        return [p for p, _ in sorted(pats.items(), key=lambda x: -x[1])[:n] if p]

    # ── 실제 5-iteration 부트스트래핑 ──────────────────────────────
    dipre_precisions    = []
    snowball_precisions = []

    dipre_seeds    = seed_df.copy()        # DIPRE: 노이즈 포함 확장
    snowball_seeds = seed_df.copy()        # Snowball: 신뢰도 필터 후 확장
    dipre_pool     = pool_df.copy()
    snowball_pool  = pool_df.copy()

    for iteration in range(1, 6):
        # ── DIPRE ──────────────────────────────────────────────
        top_pats = extract_top_patterns(dipre_seeds)
        if top_pats:
            matched_mask = dipre_pool['marked_text'].apply(
                lambda txt: any(p in str(txt) for p in top_pats)
            )
            matched = dipre_pool[matched_mask]
            prec = (matched['final_relation'] == target_rel).mean() if len(matched) else 0.0
            dipre_precisions.append(prec)
            # 매칭된 것 전부 시드에 추가 (노이즈도 포함 → Semantic Drift)
            if len(matched):
                dipre_seeds = pd.concat([dipre_seeds, matched.head(20)], ignore_index=True)
                dipre_pool  = dipre_pool[~matched_mask].reset_index(drop=True)
        else:
            dipre_precisions.append(0.0)

        # ── Snowball ────────────────────────────────────────────
        top_pats_sw = extract_top_patterns(snowball_seeds)
        if top_pats_sw:
            matched_mask_sw = snowball_pool.apply(
                lambda row: (any(p in str(row['marked_text']) for p in top_pats_sw)
                             and row['head_type'] == 'PER'
                             and row['tail_type'] == 'ORG'),
                axis=1
            )
            matched_sw = snowball_pool[matched_mask_sw]
            prec_sw = (matched_sw['final_relation'] == target_rel).mean() if len(matched_sw) else 0.0
            snowball_precisions.append(prec_sw)
            # 신뢰도 높은 것만 시드에 추가
            high_conf = matched_sw[matched_sw['final_relation'] == target_rel]
            if len(high_conf):
                snowball_seeds = pd.concat([snowball_seeds, high_conf.head(20)], ignore_index=True)
            snowball_pool = snowball_pool[~matched_mask_sw].reset_index(drop=True)
        else:
            snowball_precisions.append(0.0)

        print(f"  Iter {iteration}: DIPRE Precision={dipre_precisions[-1]:.3f} | "
              f"Snowball Precision={snowball_precisions[-1]:.3f}")

    # F1: 마지막 iteration precision + 실제 recall 계산
    # recall = pool에서 target_rel 문장 중 실제로 찾은 비율
    target_in_pool = (pool_df['final_relation'] == target_rel).sum()
    if target_in_pool > 0:
        dipre_found    = len(dipre_pool[dipre_pool['final_relation'] != target_rel])  # 남은 pool에서 역산
        dipre_recall   = 1 - (dipre_pool['final_relation'] == target_rel).sum() / target_in_pool
        snow_recall    = 1 - (snowball_pool['final_relation'] == target_rel).sum() / target_in_pool
    else:
        dipre_recall = snow_recall = 0.0

    dp = dipre_precisions[-1]
    dr = dipre_recall
    dipre_f1 = 2*dp*dr/(dp+dr) if (dp+dr) > 0 else 0.0

    sp = snowball_precisions[-1]
    sr = snow_recall
    snowball_f1 = 2*sp*sr/(sp+sr) if (sp+sr) > 0 else 0.0

    print(f"  DIPRE   최종 Precision={dp:.3f}, Recall={dr:.3f}, F1={dipre_f1:.4f}")
    print(f"  Snowball 최종 Precision={sp:.3f}, Recall={sr:.3f}, F1={snowball_f1:.4f}")

    # 실측 precision 추이 시각화
    iterations = list(range(1, 6))
    plt.figure(figsize=(8, 5))
    plt.plot(iterations, dipre_precisions,    'r--o', label='DIPRE (Semantic Drift)')
    plt.plot(iterations, snowball_precisions, 'b-s',  label='Snowball (Confidence Filter)')
    plt.title('[KLUE-RE] DIPRE vs Snowball Precision 변화 (실측)', fontsize=14)
    plt.xlabel('Bootstrap Iteration')
    plt.ylabel('Precision (실측)')
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/klue_semantic_drift_comparison.png', dpi=300)
    plt.close()
    print("  ✅ klue_semantic_drift_comparison.png 저장 완료")
    return dipre_f1, snowball_f1


# ─────────────────────────────────────────────
# 5. DEEP LEARNING — Bi-LSTM + Attention
# ─────────────────────────────────────────────
def run_klue_deep_learning(train_df, test_df):
    print("\n=== [KLUE] 5. Deep Learning — Bi-LSTM + Attention ===")
    import torch, torch.nn as nn, torch.optim as optim
    from torch.utils.data import DataLoader, Dataset

    texts_tr = train_df['marked_text'].tolist()
    labels_tr = train_df['final_relation'].tolist()
    texts_te = test_df['marked_text'].tolist()
    labels_te = test_df['final_relation'].tolist()

    unique_labels = sorted(set(labels_tr + labels_te))
    label2idx = {l: i for i, l in enumerate(unique_labels)}
    idx2label = {i: l for l, i in label2idx.items()}

    word2idx = {'<PAD>': 0, '<UNK>': 1}
    for t in texts_tr:
        for w in str(t).split():
            if w not in word2idx:
                word2idx[w] = len(word2idx)

    MAX_LEN = 80

    class REDataset(Dataset):
        def __init__(self, texts, labels):
            self.texts = texts; self.labels = labels
        def __len__(self): return len(self.texts)
        def __getitem__(self, i):
            ws = str(self.texts[i]).split()
            seq = [word2idx.get(w, 1) for w in ws[:MAX_LEN]]
            seq += [0] * (MAX_LEN - len(seq))
            return torch.tensor(seq, dtype=torch.long), torch.tensor(label2idx[self.labels[i]], dtype=torch.long)

    class BiLSTMAttn(nn.Module):
        def __init__(self, V, E, H, C):
            super().__init__()
            self.emb  = nn.Embedding(V, E, padding_idx=0)
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

    tr_loader = DataLoader(REDataset(texts_tr, labels_tr), batch_size=64, shuffle=True)
    te_loader = DataLoader(REDataset(texts_te, labels_te), batch_size=64)

    model = BiLSTMAttn(len(word2idx), 128, 128, len(unique_labels)).to(device)
    opt   = optim.Adam(model.parameters(), lr=1e-3)
    crit  = nn.CrossEntropyLoss()

    for ep in range(5):
        model.train(); total = 0
        for seq, lbl in tr_loader:
            seq, lbl = seq.to(device), lbl.to(device)
            opt.zero_grad()
            out, _ = model(seq)
            loss = crit(out, lbl)
            loss.backward(); opt.step()
            total += loss.item()
        print(f"  Epoch {ep+1}/5 | Loss: {total/len(tr_loader):.4f}")

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for seq, lbl in te_loader:
            seq = seq.to(device)
            out, _ = model(seq)
            preds.extend(torch.argmax(out,1).cpu().numpy())
            trues.extend(lbl.numpy())

    f1 = f1_score(trues, preds, average='macro', zero_division=0)
    print(f"  Bi-LSTM+Attention Macro F1: {f1:.4f}")

    # Attention Heatmap — 예시 1문장
    sample_text  = texts_te[0]
    sample_label = labels_te[0]
    words = str(sample_text).split()[:MAX_LEN]
    seq = [word2idx.get(w, 1) for w in words] + [0]*(MAX_LEN-len(words))
    out, attn = model(torch.tensor([seq], dtype=torch.long).to(device))
    pred_label = idx2label[torch.argmax(out,1).item()]
    weights = attn.cpu().detach().numpy()[0][:len(words)]

    plt.figure(figsize=(max(12, len(words)*0.7), 3))
    sns.heatmap([weights], xticklabels=words, yticklabels=['Attn'],
                cmap='Reds', annot=True, fmt=".2f", cbar=False)
    plt.title(f"[KLUE-RE] Attention Heatmap | 정답: {sample_label} / 예측: {pred_label}", fontsize=13)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/klue_attention_heatmap.png", dpi=300)
    plt.close()
    print("  ✅ klue_attention_heatmap.png 저장 완료")
    return f1


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import os; os.makedirs(SAVE_DIR, exist_ok=True)

    print("KLUE-RE 데이터 로딩 중...")
    train_df = load_klue_re('train')
    test_df  = load_klue_re('validation')

    results = {}

    v_pat, v_emb             = run_klue_unsupervised(train_df)
    results['unsup_pattern']  = v_pat
    results['unsup_embed']    = v_emb

    results['feature_rf']     = run_klue_feature_based(train_df, test_df)
    results['kernel_svm']     = run_klue_kernel(train_df, test_df)
    results['bilstm_attn']    = run_klue_deep_learning(train_df, test_df)

    print("\n\n" + "="*55)
    print("  [KLUE-RE] 최종 성능 요약")
    print("="*55)
    for k, v in results.items():
        print(f"  {k:25s}: {v:.4f}")

    # Summary Bar Chart
    categories = ['Unsupervised\n(V-Measure)', 'Unsupervised\n(V-Measure)',
                  'Supervised\n(Macro F1)', 'Supervised\n(Macro F1)', 'Deep Learning\n(Macro F1)']
    labels_bar  = ['Pattern-based', 'Embedding-based', 'Feature RF', 'Kernel SVM', 'Bi-LSTM+Attn']
    scores_bar  = [results['unsup_pattern'], results['unsup_embed'],
                   results['feature_rf'],    results['kernel_svm'],  results['bilstm_attn']]
    colors_bar  = ['#ffb3b3','#99ccff','#b3ffb3','#ffe599','#d5b3ff']

    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels_bar, scores_bar, color=colors_bar, width=0.6)
    for b in bars:
        plt.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                 f"{b.get_height():.4f}", ha='center', fontweight='bold', fontsize=11)
    plt.title("[KLUE-RE] 전체 파이프라인 성능 비교", fontsize=15)
    plt.ylabel("Score (0~1)")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/klue_final_comparison.png", dpi=300)
    plt.close()
    print("✅ klue_final_comparison.png 저장 완료")
