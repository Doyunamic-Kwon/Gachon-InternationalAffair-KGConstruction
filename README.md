# Relation Extraction (RE) 실험 파이프라인 — 방법론 비교 분석

관계 추출(RE)의 고전 머신러닝부터 딥러닝까지 4가지 패러다임을 통시적으로 비교 분석하는 파이프라인 실험입니다.  
**가천대학교 국제교류처(OIA) 공지·행정 텍스트**를 주 데이터셋으로 사용하고, **KLUE-RE** 공개 벤치마크로 일반화 성능을 검증했습니다.

---

## 0. 데이터셋 개요

| 구분 | 설명 | 건수 |
|---|---|---|
| **Gold Standard** | 수작업 레이블링 (관계 12종) | 257건 |
| **Silver Standard** | LLM 자동 레이블링 | 1,205건 |
| **합계** | Gold + Silver | **1,462건** |

![데이터셋 개요](docs/step0_data_overview.png)
<p align="center"><i>[그림 0] OIA 데이터셋 구성 및 관계 분포 (Gold 257건 기준)</i></p>

---

## 🏆 Executive Summary — 4대 패러다임 최종 성능 비교

![최종 성능 비교](docs/step6_final_comparison.png)
<p align="center"><i>[그림 0-1] Relation Extraction 4대 패러다임 · 7개 모델 성능 종합 비교 (OIA 1,462건)</i></p>

| 패러다임 | 모델 | 지표 | 점수 | 데이터 수 |
|---|---|---|---|---|
| Unsupervised | Pattern-based (TF-IDF KMeans) | V-Measure | 0.2325 | Gold 257건 |
| Unsupervised | Embedding-based (SBERT KMeans) | V-Measure | 0.3695 | Gold 257건 |
| Semi-supervised | DIPRE (Bootstrapping) | Macro F1 | 0.1590 | Seed 5개 |
| Semi-supervised | Snowball (신뢰도 필터) | Macro F1 | 0.2507 | Seed 5개 |
| Supervised ML | Feature-based (Random Forest) | Macro F1 | 0.7300 | 1,462건 |
| Supervised ML | Kernel-based SVM (Composite) | Macro F1 | **0.8627** | 1,000건* |
| Deep Learning | Bi-LSTM + Attention (Scratch) | Macro F1 | 0.5669 | 1,462건 |

> \*Kernel SVM은 O(N²) 행렬 연산 특성상 최대 1,000건으로 제한됩니다. 이는 Kernel 방법론의 확장성 한계를 의미합니다.

**핵심 인사이트:**
- **Supervised ML 최고 성능**: OIA 행정 텍스트는 개체 타입(Entity Type)이 관계를 거의 결정짓는 구조 → Feature/Kernel 모델에 유리
- **Deep Learning 상대적 저성능**: 사전학습 임베딩 없이 Scratch 학습, 소량 도메인 데이터 → PLM(BERT) 적용 시 대폭 향상 기대
- **Semi-supervised 의의**: 레이블 없이 Seed 5개만으로 작동하는 데이터 효율성 증명

---

## 1. Unsupervised RE (비지도 학습 관계 추출)

정답 레이블 없이 텍스트의 패턴과 분포만으로 관계를 군집화하는 두 가지 기법을 비교했습니다. **(Gold 257건)**

### ① Open Information Extraction (Open IE)
- **방법**: SpaCy 의존 구문 분석(Dependency Parsing)으로 `(Subject, Verb, Object)` 원시 튜플 추출
- **결과 및 한계**: `(사감실, >6개월</, 생활관)` 처럼 행정/공지사항 특유의 파편화된 텍스트에서 구문 트리가 붕괴됩니다. Rule-based Open IE는 문법적으로 완전한 자연어 문장에서만 유효합니다.

### ② Pattern-based Clustering (어휘 패턴 군집화)
- **방법**: 두 개체(Entity) 사이 문자열을 TF-IDF로 벡터화 → K-Means 군집화 (K = 관계 종류 수)
- **평가 (V-Measure: 0.2325)**: 표면 어휘만 보므로 단어가 겹치면 다른 관계를 같은 군집으로 묶는 **Lexical Sparsity** 문제 발생

### ③ Embedding-based RE (Distributional Similarity)
- **방법**: *"비슷한 문맥에 나오는 단어·문장은 비슷한 의미를 갖는다"* — 분포 의미론(Distributional Hypothesis). **Sentence-BERT (paraphrase-multilingual-MiniLM-L12-v2)** 로 문장 전체 의미를 벡터화 → K-Means 군집화
- **평가 (V-Measure: 0.3695)**: 어휘 공유 없이도 문맥 유사도로 판단 → 패턴 기반 대비 **+58.9% 향상**

![Unsupervised 성능 비교](docs/step1_unsupervised_comparison.png)
<p align="center"><i>[그림 1] Unsupervised RE — 방법론별 V-Measure 비교 및 평가 방식 설명 (Gold 257건)</i></p>

### 🌟 t-SNE 군집 시각화
정답 라벨 없이 학습한 결과를 2D 공간에 투영하여 군집 형성 품질을 확인합니다.

![Unsupervised t-SNE](docs/step1_unsupervised_tsne.png)
<p align="center"><i>[그림 2] Pattern-based vs Embedding-based 군집화 비교 — t-SNE 투영 (Gold 257건)</i></p>

- **Pattern-based (좌)**: 같은 색상(같은 관계)이 전 공간에 분산 → 군집 응집도 낮음
- **Embedding-based (우)**: 같은 관계끼리 뚜렷하게 뭉침 → SBERT의 의미 공간 우수성 확인

---

## 2. Semi-supervised RE (DIPRE vs Snowball 부트스트래핑)

소수의 Seed(Gold 5건)만으로 Silver 코퍼스에서 관계 인스턴스를 자동 증식하는 알고리즘입니다. 레이블이 없어도 동작하는 **데이터 효율성**이 핵심이며, 두 알고리즘의 정밀도 유지 전략을 비교합니다.

### ① DIPRE (Dual Iterative Pattern Relation Extraction)
- **방법**: Gold Seed → 텍스트 패턴 추출 → Silver 코퍼스 탐색 → 새 튜플 추가 → 반복
- **문제점 — Semantic Drift**: 패턴만 일치하면 무조건 확장 → 노이즈 축적 → 패턴이 점점 일반화 → **정밀도 하락**
- **결과 (Macro F1: 0.1590)**: 실제 HAS_FEE 관계에서 4개 튜플 추출, 그 중 의미 없는 텍스트 포함

### ② Snowball (Confidence Score 필터링)
- **개선**: 추출된 튜플의 개체 타입(Entity Type)이 Seed와 일치하는 경우만 채택
- **효과**: HAS_FEE 관계에서 노이즈 **75% 즉각 제거** (4개 → 1개 잔류)
- **결과 (Macro F1: 0.2507)**: DIPRE 대비 **+57.7% 향상**, 정밀도 유지 확인

![Semi-supervised 비교](docs/step2_semi_supervised.png)
<p align="center"><i>[그림 3] DIPRE vs Snowball — Semantic Drift 추이 (좌) & 최종 Macro F1 비교 (우)</i></p>

**평가 방식**: 각 관계별 Gold 5-seed → 패턴 추출 → Silver 전체에 대한 이진 분류(해당 관계 vs OTHER) F1 계산 → 관계별 F1의 Macro 평균

---

## 3. Supervised ML — Feature-based RE (Random Forest)

패턴이나 군집화 없이, 명시적으로 설계된 **4가지 언어학적 자질(Linguistic Feature)**을 추출하여 Random Forest로 분류합니다. **(Gold 257건 + Silver 1,205건 = 1,462건)**

### 피처 설계

| 피처 그룹 | 추출 방법 | 예시 |
|---|---|---|
| **Context Words** | 문장 전체 TF-IDF (max 500) | "신청서 다운로드 제출" |
| **Words Between** | 두 개체 사이 TF-IDF (max 500) | "의 수수료는" |
| **Semantic Feature** | 개체 타입 조합 (CountVectorizer) | "PROGRAM\|MONEY" |
| **Dependency Path** | SpaCy 구문 의존 경로 TF-IDF | "NOUN(obj) → VERB(root)" |

- **결과 (Test 293건, Macro F1: 0.7300)**
- **Feature Importance**: **Semantic Feature(개체 타입 조합)** 압도적 1위 — 행정 텍스트 특성 반영
- **Data Leakage 방지**: TF-IDF vectorizer는 Train 데이터에만 `fit`, Test에는 `transform`만 적용

![Feature-based RF 분석](docs/step3_feature_based.png)
<p align="center"><i>[그림 4] Feature Importance 분석 (좌) & Confusion Matrix (우) — Feature-based RF (1,462건)</i></p>

---

## 4. Supervised ML — Kernel-based RE (Composite SVM)

Feature를 직접 추출하는 대신, **두 문장 간 구조적 유사도를 커널 함수로 정의**하여 SVM으로 분류합니다.

### Composite Kernel 수식

```
K_composite = 0.3 · K_seq + 0.3 · K_tree + 0.4 · K_semantic
```

| 커널 | 측정 대상 | 계산 방식 |
|---|---|---|
| **K_seq** (α=0.3) | 개체 사이 단어 어휘 집합 유사도 | Jaccard Similarity |
| **K_tree** (β=0.3) | SpaCy 구문 트리 간선 집합 유사도 | Jaccard Similarity |
| **K_semantic** (γ=0.4) | 개체 타입 쌍 일치 여부 | 0 or 1 (Boolean) |

- **결과 (1,000건 샘플, Macro F1: 0.8627)** — 전체 최고 성능
- **한계**: N×N 커널 행렬 연산 → **O(N²) 시간·공간 복잡도** → 대규모 데이터 적용 불가

![Kernel SVM 분석](docs/step4_kernel_svm.png)
<p align="center"><i>[그림 5] Kernel Matrix t-SNE 투영 (좌, 관계별 분리 확인) & Confusion Matrix (우) — Kernel SVM (1,000건)</i></p>

---

## 5. Deep Learning — Bi-LSTM + Attention

Feature Engineering 없이, 단어 시퀀스에서 직접 관계를 학습하는 딥러닝 모델을 구축했습니다. **(Gold + Silver 1,462건, Scratch 학습)**

### 아키텍처

```
Input Tokens → Embedding (dim=128) → Bi-LSTM (hidden=64) →
Attention Mechanism → Context Vector → Linear Classifier → Softmax
```

- **Attention**: `score = softmax(v · tanh(W · h))` — 관계를 결정하는 핵심 단어에 집중
- **결과 (Test 293건, Macro F1: 0.5669, 10 Epochs)**

### Supervised ML 대비 낮은 이유

1. **사전학습 임베딩 없음**: 단어 의미를 1,462건만으로 학습 → 어휘 일반화 한계
2. **단답형 텍스트 특성**: 문맥이 짧아 LSTM이 활용할 시퀀스 정보 부족
3. **개체 타입 힌트 없음**: RF가 쉽게 쓰는 `PROGRAM|MONEY` 타입 정보를 직접 주지 않음

![Deep Learning 학습 결과](docs/step5_deep_learning.png)
<p align="center"><i>[그림 6] 학습 곡선 (좌) & Attention Heatmap (우) — Bi-LSTM+Attention Scratch 학습 (1,462건)</i></p>

- **Attention Heatmap**: 붉을수록 모델이 집중한 단어 → 관계 트리거 단어에 높은 가중치 부여 확인 (모델 해석 가능성 증명)

---

## Part 2. KLUE-RE 공개 데이터셋 벤치마크

> 커스텀 OIA 데이터의 **도메인 특화 편향** — 단순 행정 텍스트 구조가 Feature ML에 유리 — 를 검증하기 위해, 한국어 RE 표준 벤치마크 **KLUE-RE**로 동일 파이프라인을 재실험했습니다.

### 데이터셋 개요

| 항목 | 내용 |
|---|---|
| **출처** | HuggingFace `klue/re` |
| **Train** | 32,470건 |
| **Test (Validation)** | 7,765건 |
| **관계 수** | 30개 (`no_relation` 포함) |
| **언어** | 한국어 (뉴스·위키백과, 자연어 문장) |
| **특이사항** | `no_relation` 약 29% — 심각한 클래스 불균형 |

### KLUE-RE 성능 비교

| 패러다임 | 모델 | 지표 | 점수 |
|---|---|---|---|
| Unsupervised | Pattern-based (TF-IDF KMeans) | V-Measure | 0.0897 |
| Unsupervised | Embedding-based (SBERT KMeans) | V-Measure | 0.1392 |
| Supervised ML | Feature-based (Random Forest) | Macro F1 | 0.1626 |
| Supervised ML | Kernel-based SVM (Composite) | Macro F1 | **0.2222** |
| Deep Learning | Bi-LSTM + Attention | Macro F1 | 0.0706 |

![KLUE 최종 비교](docs/klue_final_comparison.png)
<p align="center"><i>[그림 7] KLUE-RE 전체 파이프라인 성능 비교</i></p>

### 🔬 KLUE-RE 저성능 원인 분석

#### 1. 30개 다중 클래스 + 극심한 클래스 불균형 → Macro F1 붕괴
- `no_relation` 29%, 일부 관계 수십 건 → 모델이 다수 클래스에 편향
- 소수 클래스 F1 → 0점 수렴 → Macro F1 전체 급락

#### 2. 자연어 문장의 복잡성 — 표면적 자질 붕괴
- OIA: 획일적 행정 패턴 반복 → Feature ML 유리
- KLUE: 피동/사동/도치/장문 → Words Between, TF-IDF 패턴 무력화

#### 3. 사전학습 모델 부재 — Scratch 임베딩 한계
- BERT 없이 32K 문장만으로 30개 관계의 의미 경계 수렴 불가
- SpaCy `ko_core_news_sm`의 KLUE 텍스트 파싱 오류 → Dependency Path 노이즈화

---

### 💡 OIA vs KLUE 비교 — 핵심 아키텍처 인사이트

| 비교 항목 | 🏢 OIA (행정 특화) | 📰 KLUE-RE (일반 자연어) |
|---|---|---|
| **Supervised ML F1** | **0.73 ~ 0.86** | **0.16 ~ 0.22** |
| **Deep Learning F1** | **0.57** | **0.07** |
| **핵심 요인** | 단순 반복 패턴 + 12 관계 → Entity Type으로 거의 결정 | 30 관계 + 복잡 문장 + 불균형 → Scratch 모델 붕괴 |
| **결론** | Feature Engineering + Kernel SVM = 최고 효율 | **Pre-trained PLM (BERT/RoBERTa) 적용 필수** |

---

### KLUE-RE 시각화

![KLUE Unsupervised](docs/klue_unsupervised_comparison.png)
<p align="center"><i>[그림 8] KLUE-RE Unsupervised 군집화 성능 비교 (V-Measure)</i></p>

![KLUE t-SNE](docs/klue_unsupervised_tsne.png)
<p align="center"><i>[그림 9] KLUE-RE Pattern vs Embedding 군집화 t-SNE (3,000건 샘플)</i></p>

![KLUE Feature Importance](docs/klue_feature_importance.png)
<p align="center"><i>[그림 10] KLUE-RE Feature-based RF 피처 중요도</i></p>

![KLUE Confusion Matrix](docs/klue_feature_confusion_matrix.png)
<p align="center"><i>[그림 11] KLUE-RE Feature-based RF Confusion Matrix</i></p>

![KLUE Kernel t-SNE](docs/klue_kernel_tsne.png)
<p align="center"><i>[그림 12] KLUE-RE Kernel Matrix t-SNE (1,000건 샘플)</i></p>

![KLUE Semantic Drift](docs/klue_semantic_drift_comparison.png)
<p align="center"><i>[그림 13] KLUE-RE DIPRE vs Snowball Precision 변화 (5 Iterations)</i></p>

![KLUE Attention Heatmap](docs/klue_attention_heatmap.png)
<p align="center"><i>[그림 14] KLUE-RE Bi-LSTM+Attention Heatmap</i></p>

---

## 실험 재현 방법

```bash
# 가상환경 활성화
source .venv/bin/activate

# 전체 OIA 파이프라인 실행 (시각화 자동 생성, ~1분)
python run_all_pipeline.py

# KLUE-RE 파이프라인 (약 30분, 인터넷 필요)
python klue_pipeline.py
```

결과는 `docs/` 폴더에 PNG로, 수치는 `docs/results.json`에 저장됩니다.

---

## 파일 구조

| 파일 | 설명 |
|---|---|
| `run_all_pipeline.py` | **OIA 전체 파이프라인 마스터 실행 스크립트** |
| `step1_data_loader.py` | Gold/Silver 데이터 로더 |
| `step2_unsupervised_re_v2.py` | Unsupervised RE (Open IE / Pattern / SBERT) |
| `step3_feature_based_re_v2.py` | Feature-based RF (Data Leakage 수정 완료) |
| `step3b_semi_supervised.py` | Semi-supervised DIPRE & Snowball (실측 F1 계산) |
| `step3c_kernel_based_re.py` | Kernel 피처 추출 (Sequence + Tree) |
| `step4_deep_learning_re.py` | Bi-LSTM+Attention (vocab leakage 수정 완료) |
| `visualize_kernel_ml.py` | Composite Kernel 계산 + Kernel SVM 시각화 |
| `klue_data_loader.py` | KLUE-RE HuggingFace 로더 |
| `klue_pipeline.py` | KLUE-RE 전체 파이프라인 |
| `docs/results.json` | 실측 성능 수치 (모든 모델) |
