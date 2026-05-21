"""
Standalone script: regenerate docs/step1_unsupervised_tsne.png
Uses the same data & feature extraction as step2_unsupervised_re_v2.py
(corpus_clean.jsonl 1730 items, char n-gram TF-IDF, SBERT clean text)
so the V-Measure titles match the bar chart values.
"""
import os, re, json, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("docs", exist_ok=True)
plt.rc('font', family='DejaVu Sans')
plt.rcParams['axes.unicode_minus'] = False

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import v_measure_score
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from step1_data_loader import load_gold_standard

# ── Normalization (same as step2) ─────────────────────
NORMALIZE_REL = {
    "requires_document":      "REQUIRES_DOCUMENT",
    "has_deadline":            "HAS_DEADLINE",
    "announced_by":            "ANNOUNCED_BY",
    "mentions":                "MENTIONS",
    "requires_qualification":  "REQUIRES_QUALIFICATION",
}
def norm_rel(r): return NORMALIZE_REL.get(str(r).strip(), str(r).strip())

def get_pattern(row):
    marked = str(row.get("marked_text", ""))
    m = re.search(r'\[/E1\](.*?)\[E2\]', marked, re.DOTALL) or \
        re.search(r'\[/E2\](.*?)\[E1\]', marked, re.DOTALL)
    between = m.group(1).strip() if m else ""
    head = row.get("head") or {}
    tail = row.get("tail") or {}
    h_type = head.get("type", "") if isinstance(head, dict) else ""
    t_type = tail.get("type", "") if isinstance(tail, dict) else ""
    return f"{between} __H_{h_type}__ __T_{t_type}__"

# ── Load full corpus (same as step2) ──────────────────
corpus_path = "data/re_fixed_v6/corpus_clean.jsonl"
print(f"Loading corpus from {corpus_path}...")
with open(corpus_path, encoding="utf-8") as f:
    corpus = [json.loads(line) for line in f]
print(f"  Loaded: {len(corpus)} items")

texts     = [r.get("marked_text", r.get("sentence", "")) for r in corpus]
true_rels = [norm_rel(r.get("relation", "UNKNOWN")) for r in corpus]
k = len(set(true_rels))
print(f"  Relations: {k}")

# ── Load gold labels (for coloring t-SNE) ─────────────
gold_df = load_gold_standard()
gold_df["final_relation"] = gold_df["final_relation"].apply(norm_rel)
gold_texts_set = set(gold_df["marked_text"].fillna("").tolist())

# Find which corpus indices are gold
gold_indices = []
gold_labels  = []
for i, r in enumerate(corpus):
    mt = r.get("marked_text", "")
    rel = r.get("relation", "")
    if mt in gold_texts_set:
        gold_indices.append(i)
        gold_labels.append(norm_rel(rel))

print(f"  Gold items found in corpus: {len(gold_indices)}")

# ── Pattern-based clustering ───────────────────────────
print("Pattern-based clustering...")
patterns  = [get_pattern(r) for r in corpus]
vec_pat   = TfidfVectorizer(max_features=1000, analyzer="char_wb", ngram_range=(2, 4))
X_pat     = vec_pat.fit_transform(patterns)
km_pat    = KMeans(n_clusters=k, random_state=42, n_init=10)
labels_pat = km_pat.fit_predict(X_pat)
v_pat      = v_measure_score(true_rels, labels_pat)
print(f"  Pattern-based V-Measure: {v_pat:.4f}")

# ── Embedding-based clustering ─────────────────────────
print("SBERT encoding...")
sbert = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
clean_texts = [re.sub(r'\[/?E[12]\]', '', t).strip() for t in texts]
X_emb  = sbert.encode(clean_texts, batch_size=64, show_progress_bar=True)
km_emb = KMeans(n_clusters=k, random_state=42, n_init=10)
labels_emb = km_emb.fit_predict(X_emb)
v_emb  = v_measure_score(true_rels, labels_emb)
print(f"  Embedding-based V-Measure: {v_emb:.4f}")

# ── t-SNE on full corpus, plot gold items only ─────────
print("t-SNE projection (full corpus, plot gold items)...")
tsne    = TSNE(n_components=2, random_state=42, init='random', perplexity=30, max_iter=1000)
all_emb2d = tsne.fit_transform(X_emb)
tsne2   = TSNE(n_components=2, random_state=42, init='random', perplexity=30, max_iter=1000)
all_pat2d = tsne2.fit_transform(X_pat.toarray())

# Use corpus labels for gold items (from corpus, not gold_df)
if gold_indices:
    gold_rel_from_corpus = [norm_rel(corpus[i].get("relation","UNKNOWN")) for i in gold_indices]
    all_labels_set = sorted(set(gold_rel_from_corpus))
else:
    all_labels_set = sorted(set(true_rels))
    gold_indices = list(range(min(257, len(corpus))))
    gold_rel_from_corpus = [true_rels[i] for i in gold_indices]

palette   = sns.color_palette("tab10", len(all_labels_set))
color_map = {l: palette[i] for i, l in enumerate(all_labels_set)}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
for lbl in all_labels_set:
    idx = [gold_indices[j] for j, l in enumerate(gold_rel_from_corpus) if l == lbl]
    ax1.scatter([all_pat2d[i,0] for i in idx], [all_pat2d[i,1] for i in idx],
                c=[color_map[lbl]], label=lbl, s=30, alpha=0.75, edgecolors='none')
    ax2.scatter([all_emb2d[i,0] for i in idx], [all_emb2d[i,1] for i in idx],
                c=[color_map[lbl]], label=lbl, s=30, alpha=0.75, edgecolors='none')

ax1.set_title(f"Pattern-based (Char TF-IDF)  V-Measure: {v_pat:.4f}", fontsize=12, fontweight='bold')
ax2.set_title(f"Embedding-based (SBERT)  V-Measure: {v_emb:.4f}", fontsize=12, fontweight='bold')
for ax in (ax1, ax2):
    ax.set_xlabel("t-SNE dim 1", fontsize=10)
    ax.set_ylabel("t-SNE dim 2", fontsize=10)
    sns.despine(ax=ax)
handles = [plt.Line2D([0],[0], marker='o', color='w',
                       markerfacecolor=color_map[l], markersize=8, label=l)
           for l in all_labels_set]
ax2.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc='upper left',
           fontsize=8, title="Relation")
plt.suptitle(f"Unsupervised RE — t-SNE Cluster Visualization (Corpus {len(corpus)} items, Gold highlighted)",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("docs/step1_unsupervised_tsne.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ docs/step1_unsupervised_tsne.png saved  (pattern={v_pat:.4f}, embed={v_emb:.4f})")
