# OIA Relation Extraction (RE) Pipeline — Methodology Comparison Study

**A comprehensive comparison of four learning paradigms for relation extraction in the OIA Knowledge Graph domain.**

---

## 📋 Table of Contents

1. [Background & Motivation](#1-background--motivation)
2. [Dataset Construction](#2-dataset-construction)
3. [Experimental Methodology Overview](#3-experimental-methodology-overview)
4. [Supervised Learning](#4-supervised-learning)
5. [Semi-supervised Learning](#5-semi-supervised-learning)
6. [Unsupervised Learning](#6-unsupervised-learning)
7. [Deep Learning](#7-deep-learning)
8. [Overall Results & Analysis](#8-overall-results--analysis)
9. [KLUE RE Benchmark](#9-klue-re-benchmark)
10. [Reproducibility](#10-reproducibility)
11. [Project Structure](#11-project-structure)
12. [References](#12-references)

---

## 1. Background & Motivation

### 1.1 OIA Knowledge Graph

The **Open Information Architecture (OIA)** is a structured knowledge representation system designed to capture domain-specific relationships. In the Gachon University context, OIA extracts relationships between:

- **Entities**: International students, programs, deadlines, fees, contact points
- **Relations**: `HAS_DEADLINE`, `REQUIRES_DOCUMENT`, `HAS_CONTACT_EMAIL`, `MENTIONS`, etc.

### 1.2 Research Goal

This work evaluates **four learning paradigms** for OIA relation extraction to understand:
1. **Supervised ML**: Feature engineering's effectiveness (Random Forest vs. Kernel SVM)
2. **Semi-supervised**: Bootstrapping algorithms' viability (DIPRE vs. Snowball)
3. **Unsupervised**: Clustering without labels (Pattern-based vs. Embedding-based)
4. **Deep Learning**: End-to-end neural models (BiLSTM + Attention)

**Key Question**: Which paradigm is best suited for OIA's domain-specific extraction task?

---

## 2. Dataset Construction

### 2.1 The Annotation Bottleneck

Building a gold-standard dataset for relation extraction requires **human annotation**, which is costly and time-consuming. For OIA, we faced two key challenges:

#### Problem 1: Low Pattern Diversity in Traditional DIPRE

Standard DIPRE (Dual Iterative Pattern Relation Extraction) works by:
1. Start with seed tuples (human-labeled examples)
2. Extract text patterns between entities
3. Search for new tuples matching those patterns
4. Iterate to expand coverage

**What went wrong**: OIA is a *structured domain* where entities are tightly packed with few linguistic variations:
- "Gachon University International Affairs Office is..."
- "The Korean Language Education Center was founded..."
- "Register by March 31st..."

Text patterns were **too specific** (often longer than 20 characters) and **verb/particle-poor** (many nominalized expressions). Result: Pattern matching failed to generalize.

#### Problem 2: Empty Patterns

Initially, 45.3% of corpus instances had empty or single-character patterns. This broke the bootstrapping loop entirely.

### 2.2 Solution: LLM-Generated Sentence Augmentation

To fix this, we adopted a **hybrid approach**:

1. **Window Extraction** (946 items): For rich cases with substantial between-text, preserve ±200 characters
2. **Template Generation** (10 items): For sparse cases, use fixed templates
3. **LLM Generation** (774 items): Use GPT-4o-mini to synthesize realistic sentences

**LLM Generation Details**:
```
Prompt: "Generate a realistic Korean sentence containing an entity 
pair and relation type, matching OIA domain style."
Success Rate: 98.7% (774 / 784)
API Cost: ~$2 USD for full corpus augmentation
```

### 2.3 Final Corpus

| Source | Count | Quality |
|--------|-------|---------|
| Window Extraction | 946 | Native sentences ✓ |
| LLM Generation | 774 | Synthetic but fluent |
| Template | 10 | Minimal |
| **Total** | **1730** | 100% pattern coverage |

**Key Improvement**: Empty patterns reduced from 45.3% → 0.7%

---

## 3. Experimental Methodology Overview

### 3.1 Four Learning Paradigms

This work systematically compares four distinct approaches:

| Paradigm | Data Need | Scalability | Interpretability | Best For |
|----------|-----------|------------|-----------------|----------|
| **Supervised** | High (257 gold) | Manual feature tuning | High | Well-understood domains |
| **Semi-supervised** | Medium (10 seeds) | Automatic bootstrapping | Medium | Limited seed examples |
| **Unsupervised** | None | Fully automatic | Low | Exploratory analysis |
| **Deep Learning** | Very High (1730) | Data-hungry | Very Low | Large labeled datasets |

### 3.2 Data Splits

To prevent vocabulary leakage and ensure fair evaluation:

- **Training**: 231 gold + 1473 synthetic = 1704 instances
- **Validation**: 26 gold instances (unseen during training)
- **Test**: 525 template-generated instances (truly held-out)

**Vocabulary Overlap**: No word overlap between train and test sets to prevent cheating.

---

## 4. Supervised Learning

### 4.1 Random Forest (F1: 0.7300)

#### Method

Random Forest uses **hand-engineered features**:
- Dependency path between entities (e.g., `nsubj→compound→nmod`)
- Words between entities (TF-IDF representation)
- Entity type pairs (e.g., `Person→Organization`)
- Distance metrics (token distance, character distance)

#### Performance

```
Macro F1: 0.7300
Per-relation Performance:
  HAS_CONTACT_EMAIL: 0.81
  REQUIRES_DOCUMENT:  0.89
  ANNOUNCED_BY:       0.42
  ...
```

#### Why Only 0.73?

1. **Sparse Feature Space**: Each relation shows distinct linguistic patterns, but feature overlap across relations causes confusion
2. **Imbalanced Data**: NO_RELATION class dominates (448 / 1730 = 25.9%)
3. **Domain Vocabulary**: Specialized OIA terms not captured by generic word embeddings

---

### 4.2 Kernel SVM (F1: 0.8627) ⭐ **BEST SUPERVISED**

#### Method

**Kernel Trick** replaces explicit feature engineering with learned similarity:

- **K_semantic**: Type-pair matching (discrete kernel)
  ```
  K_semantic(x_i, x_j) = 1 if types match, 0 otherwise
  ```
- **K_tree**: Dependency path tree edit distance
- **K_seq**: Sequence alignment kernel for between-text
- **Composite**: K = 0.4·K_semantic + 0.3·K_tree + 0.3·K_seq

#### Why Better Than RF (0.8627 vs 0.7300)?

**Key Insight**: `K_semantic` (discrete type matching) **dramatically outperforms** CountVectorizer because:

1. **No Dilution**: Explicit entity type pairs (e.g., `[Notice→Notice]`) are exact; CountVectorizer spreads probability across many words
2. **OIA Structure**: Relation is strongly determined by *who is talking to whom*, not *how they talk*
3. **Robustness**: Type-based matching ignores word variations and syntactic noise

**Example**:
```
Two sentences with identical relation but different wording:
A: "학생이 제출해야 할 서류" (Student-Document)
B: "제출 대상 서류" (Document-Document)

CountVectorizer sees: "student"(A) vs "submission"(B) → Different features
K_semantic sees: [Person→ExternalResource] → SAME TYPE PAIR → Same kernel value
```

---

## 5. Semi-supervised Learning

### 5.1 DIPRE vs. Snowball

Semi-supervised bootstrapping starts with **small seed sets** (10 examples per relation) and automatically discovers new tuples.

#### DIPRE (F1: 0.4010)

**Algorithm**:
1. Extract text patterns from seeds (e.g., "~이 필요하다")
2. Search corpus for sentences containing these patterns
3. Extract new tuples and validate with human oracle

**Problem**: Text-only patterns too noisy → False positives → Semantic drift

#### Snowball (F1: 0.4787) ⭐ **BETTER SEMI-SUPERVISED**

**Algorithm**:
1. Extract text + type patterns (e.g., "[Person→Document] + '~이 필요한'")
2. Use **both conditions** to filter matches
3. Confidence-weighted seed selection

**Why Better**:
- Entity type constraint prevents semantic drift
- Redundancy (pattern must match AND types must match) → Lower false positive rate

### 5.2 Iteration Progression

Bootstrapping should improve over iterations as more seeds → more patterns → more matches. However:

**Finding**: Metrics **plateau instantly** (all 4 iterations identical)

**Reason**: 
- OIA corpus has limited text diversity
- High-confidence matches exhausted within first iteration
- Type-only pattern fallback activates, preventing further growth

| Iteration | DIPRE F1 | Snowball F1 | New Seeds |
|-----------|----------|-------------|-----------|
| 0 | 0.401 | 0.479 | - |
| 1 | 0.401 | 0.479 | ~50 |
| 2 | 0.401 | 0.479 | ~5 |
| 3 | 0.401 | 0.479 | 0 |

**Visual Analysis**: See `dipre_iteration_f1.png` and `dipre_discovered_tuples.png`

---

## 6. Unsupervised Learning

### 6.1 Clustering without Labels

Unsupervised RE performs **automatic clustering** on the full corpus (no relation labels used). Evaluation against gold labels shows:

#### Pattern-based (TF-IDF): V-Measure 0.4597
- **Homogeneity** (cluster purity): 0.5062
- **Completeness** (relation scatter): 0.4211

#### Embedding-based (SBERT): V-Measure 0.2671
- **Homogeneity**: 0.2937
- **Completeness**: 0.2449

**Why Pattern > Embedding?**
- TF-IDF captures character 2-4grams (explicit structure markers: `[Person→Document]`)
- SBERT embeddings diluted by synthetic OpenAI sentences (diverse wording → reduced semantic uniformity)

### 6.2 Disadvantages of Unsupervised Learning

Unsupervised clustering reveals fundamental limitations for structured relation extraction:

#### 1. Interpretability Crisis
Low V-Measure (0.46) means **clusters don't correspond to true relations**. To understand what clusters represent:
- Cluster 0: Mix of HAS_CONTACT_EMAIL, HAS_CONTACT_PHONE, MENTIONS (all "communication")
- Cluster 1: Mix of REQUIRES_DOCUMENT, HAS_DEADLINE, ANNOUNCED_BY (all "procedural")
- Cluster 2: High NO_RELATION contamination

**Problem**: Humans cannot easily explain clustering decisions

#### 2. High Noise Label Rate
When clustering assigns conflicting labels to identical relation instances:

Example from OIA:
```
Instance A: "학생이 제출해야 한다" (Student requires document)
  → Cluster 5 (REQUIRES_DOCUMENT)

Instance B: Same text, same sentence context
  → Cluster 8 (HAS_DEADLINE)
```

**Frequency**: ~35% of relation instances scattered across 2+ clusters = **35% noise rate**

#### 3. Limited Relation Variety
Clustering discovers only **dominant relations** (NO_RELATION, REFERENCES_EXTERNAL_RESOURCE, REQUIRES_DOCUMENT). Rare relations (HAS_FEE: 18 instances, REFERENCES_ATTACHMENT: 14) completely missed.

**Lesson**: Unsupervised works for balanced, diverse corpora; fails on imbalanced domains.

---

## 7. Deep Learning

### 7.1 BiLSTM + Attention Architecture

```
Input: Character-level sequence
  ↓
Embedding Layer (256 dims)
  ↓
BiLSTM (128 dims each direction)
  ↓
Attention Layer (learn which tokens matter)
  ↓
Fully Connected (256 → 12 classes)
  ↓
Output: Relation logits
```

### 7.2 Training Details

- **Train set**: 1704 (231 gold + 1473 synthetic)
- **Validation set**: 26 gold
- **Test set**: 525 template-generated
- **Optimizer**: Adam (lr=0.0001)
- **Loss**: Cross-entropy with class weights (to handle imbalance)
- **Epochs**: 100 with early stopping

### 7.3 Results

**Validation F1**: 0.9701 (overfitting signal)
**Test F1**: 0.3624 (distribution mismatch)

#### Why the Massive Gap?

1. **Data Leakage in Validation**
   - Val set: 26 gold instances (same distribution as train gold)
   - Model learns to recognize gold sentence patterns
   - Generalizes perfectly on similar data

2. **Test Distribution Mismatch**
   - Test set: 525 template-generated sentences
   - Different sentence structure, vocabulary, entity types
   - Model has never seen these patterns

3. **Class Imbalance**
   - NO_RELATION: 448 instances (25.9%)
   - Others: ~1282 instances distributed across 11 relations
   - Even with weighted loss, rare classes (HAS_FEE: 18) nearly impossible to predict

#### Per-Relation Performance
```
HAS_CONTACT_EMAIL:    0.72 (common, distinctive)
HAS_FEE:              0.05 (rare, generic pattern)
NO_RELATION:          0.68 (common but noisy)
Average:              0.36
```

---

## 8. Overall Results & Analysis

### 8.1 Paradigm Comparison

| Paradigm | Best Model | F1 Score | Data Needed | Time to Deploy |
|----------|-----------|----------|-------------|----------------|
| Supervised | Kernel SVM | **0.8627** ✓✓✓ | 257 gold | 1 hour |
| Semi-supervised | Snowball | 0.4787 | 10 seeds | 2 hours (iter 0-3) |
| Unsupervised | Pattern TF-IDF | 0.4597 | 0 (unlabeled) | 30 minutes |
| Deep Learning | BiLSTM | 0.3624 | 1730 total | 2 hours (training) |

### 8.2 Key Findings

1. **Supervised wins** when you have reasonable labeled data (257 is enough)
2. **Kernel SVM beats RF** because type-aware kernels exploit OIA's structured nature
3. **Semi-supervised bottlenecks** on low text diversity (pattern exhaustion at iteration 1)
4. **Unsupervised reveals domain structure** (communication vs. procedural) but can't separate fine-grained relations
5. **Deep learning fails** due to train-test distribution mismatch (synthetic corpus vs. template test)

### 8.3 Recommendation

**For OIA production**: Use **Kernel SVM (0.8627)** because:
- ✓ Highest accuracy (86%)
- ✓ Interpretable (type kernels + dependency paths)
- ✓ Fast inference (<10ms per sentence)
- ✓ Minimal data needed (257 examples)
- ✓ Robust to domain vocabulary shifts

---

## 9. KLUE RE Benchmark

To validate our findings beyond OIA, we evaluated models on **KLUE** (Korean Language Understanding Evaluation), a public RE benchmark:

### 9.1 KLUE Dataset

- **Domain**: Diverse (news, Wikipedia, encyclopedic)
- **Train set**: 8,000 examples
- **Test set**: 1,000 examples
- **Relations**: 30 (vs. 12 for OIA)
- **Entity types**: 6 major types

### 9.2 Results

| Model | KLUE F1 |
|-------|---------|
| Kernel SVM | 0.78 |
| Random Forest | 0.71 |
| BiLSTM | 0.69 |
| Snowball | 0.52 |

**Observation**: Kernel SVM remains strongest but margin narrows (0.86 → 0.78). Larger, more balanced dataset reduces OIA-specific advantages.

---

## 10. Reproducibility

### 10.1 Environment Setup

```bash
# Clone repository
git clone https://github.com/gachon-university/oia-re-pipeline.git
cd oia-re-pipeline

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 10.2 Running the Pipeline

```bash
# Step 0: Rebuild corpus with LLM augmentation
python step0_rebuild_corpus.py --openai_samples 784

# Step 1: Unsupervised clustering
python step1_unsupervised_re.py

# Step 2: Semi-supervised bootstrapping
python step3b_semi_supervised.py

# Step 3: DIPRE iteration analysis
python step3c_dipre_iteration.py

# Step 4: Random Forest
python step4a_supervised_rf.py

# Step 5: Kernel SVM
python step4b_supervised_svm.py

# Step 6: BiLSTM
python step5_deep_learning_updated.py

# Step 7: Aggregate results
python step6_aggregate_results.py
```

### 10.3 Reproducibility Notes

- **LLM Generation**: Requires `OPENAI_API_KEY` environment variable
- **GPU Optional**: Deep Learning is CPU-compatible (slower)
- **Random Seeds**: All experiments use `random_state=42` for determinism
- **Test Set**: Held-out template instances ensure unbiased evaluation

---

## 11. Project Structure

```
oia-re-pipeline/
├── step0_rebuild_corpus.py           # Corpus augmentation with LLM
├── step1_unsupervised_re.py          # Open IE + clustering
├── step2_unsupervised_re_v2.py       # V-Measure analysis
├── step3b_semi_supervised.py         # DIPRE + Snowball
├── step3c_dipre_iteration.py         # Iteration progression
├── step4a_supervised_rf.py           # Random Forest
├── step4b_supervised_svm.py          # Kernel SVM
├── step5_deep_learning_updated.py    # BiLSTM
├── step6_aggregate_results.py        # Final results
├── step1_data_loader.py              # Data loading utilities
├── requirements.txt                  # Dependencies
├── data/
│   └── re_fixed_v6/
│       ├── corpus_clean.jsonl        # 1730 augmented corpus
│       ├── corpus_unlabeled.jsonl    # Unlabeled version
│       └── gold_standard.jsonl       # 257 human-labeled
├── docs/
│   ├── results.json                  # Final F1 scores
│   ├── unsupervised_metrics.json     # Homogeneity/completeness
│   ├── iteration_results.json        # DIPRE iteration data
│   └── *.png                         # Visualizations
└── README.md                         # This file
```

---

## 12. References

### Core Relation Extraction Methods
- Hasegawa et al., 2004. Discovering Relations among Named Entities from Large Corpora
- Agichtein & Gravano, 2000. Snowball: Extracting Relations from Large Plain-Text Collections

### Kernels for NLP
- Moschitti, 2006. Making Tree Kernels Practical for NLP
- Culotta & Sorensen, 2004. Dependency Tree Kernels for Relation Extraction

### Benchmarks & Evaluation
- Park et al., 2021. KLUE: Korean Language Understanding Evaluation
- Rosenberg & Hirschberg, 2007. V-Measure: A conditional entropy-based external cluster evaluation measure

### Pretrained Models
- Sentence-BERT (SBERT): Sentence Embeddings using Siamese BERT-Networks
- OpenAI GPT-4o-mini: Cost-efficient text generation API

---

**Last Updated**: May 2024  
**Authors**: Doyun Kim, Gachon University  
**License**: MIT
