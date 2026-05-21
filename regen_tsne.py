"""
Standalone script: regenerate docs/step1_unsupervised_tsne.png

2-panel comparison:
  Left  — colored by Ground Truth relation  (true_rels)
  Right — colored by KMeans cluster ID      (predicted)

Shows clearly WHY V-Measure is low: same relation splits across clusters,
and clusters mix multiple relations.
"""
import os, re, json, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

# ── Normalization ──────────────────────────────────────
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

# ── Load corpus ────────────────────────────────────────
corpus_path = "data/re_fixed_v6/corpus_clean.jsonl"
print(f"Loading corpus from {corpus_path}...")
with open(corpus_path, encoding="utf-8") as f:
    corpus = [json.loads(line) for line in f]
print(f"  Loaded: {len(corpus)} items")

texts     = [r.get("marked_text", r.get("sentence", "")) for r in corpus]
true_rels = [norm_rel(r.get("relation", "UNKNOWN")) for r in corpus]
k         = len(set(true_rels))
print(f"  Relations: {k}")

# ── Pattern-based clustering ───────────────────────────
print("Pattern-based clustering...")
patterns   = [get_pattern(r) for r in corpus]
vec_pat    = TfidfVectorizer(max_features=1000, analyzer="char_wb", ngram_range=(2, 4))
X_pat      = vec_pat.fit_transform(patterns)
km_pat     = KMeans(n_clusters=k, random_state=42, n_init=10)
labels_pat = km_pat.fit_predict(X_pat)
v_pat      = v_measure_score(true_rels, labels_pat)
print(f"  Pattern-based V-Measure: {v_pat:.4f}")

# ── Embedding-based clustering ─────────────────────────
print("SBERT encoding...")
sbert       = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
clean_texts = [re.sub(r'\[/?E[12]\]', '', t).strip() for t in texts]
X_emb       = sbert.encode(clean_texts, batch_size=64, show_progress_bar=True)
km_emb      = KMeans(n_clusters=k, random_state=42, n_init=10)
labels_emb  = km_emb.fit_predict(X_emb)
v_emb       = v_measure_score(true_rels, labels_emb)
print(f"  Embedding-based V-Measure: {v_emb:.4f}")

# ── t-SNE on SBERT embeddings (full 1730 items) ────────
print("t-SNE projection (all 1730 items)...")
tsne   = TSNE(n_components=2, random_state=42, init='random', perplexity=30, max_iter=1000)
emb2d  = tsne.fit_transform(X_emb)

# ── Color palettes ─────────────────────────────────────
all_relations = sorted(set(true_rels))
rel_palette   = sns.color_palette("tab20", len(all_relations))
rel_color_map = {r: rel_palette[i] for i, r in enumerate(all_relations)}

cluster_ids  = sorted(set(labels_emb))
clus_palette = sns.color_palette("tab20b", len(cluster_ids))
clus_color_map = {c: clus_palette[i] for i, c in enumerate(cluster_ids)}

rel_colors  = [rel_color_map[r] for r in true_rels]
clus_colors = [clus_color_map[c] for c in labels_emb]

# ── Plot ───────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Left: Ground Truth
ax1.scatter(emb2d[:, 0], emb2d[:, 1],
            c=rel_colors, s=12, alpha=0.55, linewidths=0)
ax1.set_title("Ground Truth Labels", fontsize=13, fontweight='bold')
ax1.set_xlabel("t-SNE dim 1", fontsize=10)
ax1.set_ylabel("t-SNE dim 2", fontsize=10)
sns.despine(ax=ax1)

gt_handles = [mpatches.Patch(color=rel_color_map[r], label=r)
              for r in all_relations]
ax1.legend(handles=gt_handles, fontsize=6.5, loc='upper left',
           title="Relation", title_fontsize=7,
           framealpha=0.8, ncol=1)

# Right: KMeans Clusters
ax2.scatter(emb2d[:, 0], emb2d[:, 1],
            c=clus_colors, s=12, alpha=0.55, linewidths=0)
ax2.set_title(f"KMeans Clusters (k={k})", fontsize=13, fontweight='bold')
ax2.set_xlabel("t-SNE dim 1", fontsize=10)
ax2.set_ylabel("t-SNE dim 2", fontsize=10)
sns.despine(ax=ax2)

clus_handles = [mpatches.Patch(color=clus_color_map[c], label=f"Cluster {c}")
                for c in cluster_ids]
ax2.legend(handles=clus_handles, fontsize=6.5, loc='upper left',
           title="Cluster ID", title_fontsize=7,
           framealpha=0.8, ncol=1)

fig.suptitle(
    f"Unsupervised RE — t-SNE (SBERT, {len(corpus)} items)\n"
    f"V-Measure: {v_emb:.4f}  |  Embedding-based KMeans (k={k})\n"
    f"Same color in Left ≠ Same color in Right  →  Poor cluster alignment",
    fontsize=11, fontweight='bold', y=1.02
)

plt.tight_layout()
plt.savefig("docs/step1_unsupervised_tsne.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ docs/step1_unsupervised_tsne.png saved  (embed V-Measure={v_emb:.4f})")
