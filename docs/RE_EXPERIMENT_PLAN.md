# Relation Extraction Experiment Plan

## 1. Goal

가천대학교 국제교류처 OIA 크롤링 데이터에서 Relation Extraction(RE) 방법론을 통시적으로 비교한다.

비교 대상은 다음과 같다.

- LLM 기반 Silver data 생성 및 Gold set 검증
- Feature-based supervised RE
- Kernel-based supervised RE
- DIPRE
- Snowball
- Clustering 기반 unsupervised RE
- CNN 기반 RE
- LSTM 기반 RE
- Attention 기반 RE
- Transformer 기반 RE

최종 목표는 각 방법론이 OIA 도메인 KG 구축에 얼마나 유용한지 비교하는 것이다.

## 2. Data Definition

### 2.1 Source Data

현재 크롤링 산출물:

- `data/processed/documents.json`
- `data/kg/kg.json`
- `data/re_fixed_v6/candidates.jsonl`
- `data/re_fixed_v6/labeling_by_relation/*.csv`

`data/re_fixed_v6`를 최신 후보 데이터 기준으로 사용한다.

### 2.2 Gold, Silver, Weak Label 구분

현재 우리가 만든 데이터는 엄밀히 말하면 Gold Truth가 아니다.

| Data Type | Source | 신뢰도 | 용도 |
|---|---|---:|---|
| Weak label | 규칙/HTML 구조/정규식 기반 후보 생성 | 낮음~중간 | 후보 생성, baseline 사전 실험 |
| Silver label | LLM이 문맥을 보고 추출한 relation | 중간~높음 | 학습 데이터 확장 |
| Gold label | 사람이 직접 검수한 relation | 높음 | 최종 평가, LLM 품질 검증 |

따라서 현재 `data/re_fixed_v6`는 **GT가 아니라 후보 데이터**로 본다.

실험에서는 다음 구조를 사용한다.

```text
크롤링 데이터
→ 규칙 기반 후보 생성
→ LLM으로 Silver relation 추출
→ 일부 샘플을 사람이 검수하여 Gold test set 생성
→ 모든 모델은 동일한 Gold test set에서 평가
```

### 2.3 Can We Use Our Current Data as GT?

현재 데이터는 그대로 GT로 사용하면 안 된다.

이유:

- `suggested_relation`은 자동 추천값이다.
- head/tail이 잘못 잡히는 사례가 있다.
- 관계가 실제로 없는데 positive로 들어간 후보가 있다.
- 표/list 구조에서 빠진 문서가 있었고, 일부는 여전히 후보 품질 검토가 필요하다.

사용 가능한 방식:

1. `data/re_fixed_v6`를 후보 데이터로 사용한다.
2. LLM이 후보를 다시 판단하여 Silver label을 만든다.
3. 사람이 일부 샘플의 `gold_relation`을 채워 Gold set을 만든다.
4. 최종 성능 평가는 Gold set에서만 한다.

최소 Gold set 권장:

- relation별 20~50개
- 전체 300~700개
- `NO_RELATION` 포함

## 3. Entity and Relation Schema

### 3.1 Entity Types

| Entity | Description |
|---|---|
| `Notice` | 공지사항 |
| `Department` | 부서/기관 |
| `Person` | 담당자 |
| `Event` | 행사/일정/시험/프로그램 |
| `Scholarship` | 장학금 |
| `Target_Audience` | 지원 자격/대상자 |
| `Visa` | 비자/체류자격 |
| `Document` | 제출 서류 |
| `Deadline` | 마감일/기간 |
| `Fee` | 비용 |
| `Email` | 이메일 |
| `Phone` | 전화번호 |
| `Attachment` | PDF/HWP 등 첨부파일 |
| `ExternalResource` | 외부 링크/포털 |
| `ExamLevel` | TOPIK 등 시험 급수 |

### 3.2 Relation Labels

최종 relation label set:

```text
announced_by
mentions
requires_qualification
requires_document
has_deadline
HAS_FEE
HAS_DEADLINE
REQUIRES_DOCUMENT
HAS_CONTACT_EMAIL
HAS_CONTACT_PHONE
REFERENCES_ATTACHMENT
REFERENCES_EXTERNAL_RESOURCE
MENTIONS_EXAM_LEVEL
NO_RELATION
```

### 3.3 Relation Semantics

| Relation | Head | Tail | Meaning |
|---|---|---|---|
| `announced_by` | `Notice` | `Department` / `Person` | 공지를 발표/안내한 주체 |
| `mentions` | `Notice` | `Event` / `Scholarship` / `Visa` | 공지가 특정 업무 정보를 언급 |
| `requires_qualification` | `Event` / `Scholarship` / `Visa` | `Target_Audience` | 지원 자격/대상 요구 |
| `requires_document` | `Event` / `Scholarship` / `Visa` | `Document` | 제출 서류 요구 |
| `has_deadline` | `Event` / `Scholarship` / `Visa` | `Deadline` | 신청/시험/제출 기간 |
| `HAS_FEE` | `Event` / `Visa` / `Notice` | `Fee` | 비용/응시료/등록비 |
| `HAS_DEADLINE` | any | `Deadline` | 일반 날짜 relation |
| `REQUIRES_DOCUMENT` | any | `Document` | 일반 제출서류 relation |
| `HAS_CONTACT_EMAIL` | any | `Email` | 이메일 연락처 |
| `HAS_CONTACT_PHONE` | any | `Phone` | 전화번호 연락처 |
| `REFERENCES_ATTACHMENT` | any | `Attachment` | 첨부파일 참조 |
| `REFERENCES_EXTERNAL_RESOURCE` | any | `ExternalResource` | 외부 사이트 참조 |
| `MENTIONS_EXAM_LEVEL` | any | `ExamLevel` | 시험 급수 언급 |
| `NO_RELATION` | any | any | 관계 없음 |

## 4. LLM-Based Silver Data and Gold Data

### 4.1 LLM Role

LLM은 다음 작업에 사용한다.

- 문서/공지/표에서 entity 추출
- head/tail 후보 검증
- relation 후보 생성
- 기존 heuristic 후보의 오류 수정
- Silver label 생성

LLM 출력은 Gold가 아니다. LLM 출력은 Silver label이다.

### 4.2 LLM Prompt Requirements

LLM은 반드시 다음 필드를 출력해야 한다.

```json
{
  "source_url": "...",
  "sentence": "...",
  "head": {
    "text": "...",
    "type": "Event"
  },
  "tail": {
    "text": "...",
    "type": "Document"
  },
  "relation": "requires_document",
  "evidence": "...",
  "confidence": 0.0
}
```

조건:

- evidence는 원문 일부여야 한다.
- relation label은 허용된 label set에서만 선택한다.
- 불확실하면 `NO_RELATION`으로 둔다.
- 표 기반 제출서류는 row 단위로 모두 추출한다.

### 4.3 Gold Set Construction

Gold set은 사람이 직접 검수한다.

작업 파일:

- `data/re_fixed_v6/labeling_by_relation/*.csv`

검수 방식:

- `gold_relation`이 비어 있으면 사람이 채운다.
- `suggested_relation`이 맞으면 그대로 복사한다.
- 틀리면 올바른 relation을 입력한다.
- 관계가 없으면 `NO_RELATION`을 입력한다.

최종적으로 다음 파일을 만든다.

```text
data/re_gold/gold.jsonl
data/re_gold/train.jsonl
data/re_gold/dev.jsonl
data/re_gold/test.jsonl
```

## 5. Feature-Based Supervised RE

### 5.1 Purpose

고전적인 feature engineering 기반 supervised RE 성능을 측정한다.

핵심 질문:

- 명시적 linguistic feature가 OIA 도메인에서 얼마나 효과적인가?
- LLM/Silver 데이터가 feature-based model에 얼마나 도움이 되는가?

### 5.2 Features

Feature set:

- internal words
- context words
- words between mentions
- entity type pair
- entity distance
- left/right window words
- POS tags
- dependency path
- semantic features

구체적 feature:

| Feature | Description |
|---|---|
| `internal_words` | entity mention 내부 token |
| `context_words` | head/tail 주변 window token |
| `between_words` | head와 tail 사이 token |
| `entity_type_pair` | `(head_type, tail_type)` |
| `distance` | head-tail token distance |
| `dependency_path` | dependency tree 상 shortest path |
| `trigger_words` | 신청, 제출, 필요, 마감, 응시료 등 relation trigger |
| `semantic_feature` | sentence embedding 또는 synonym group |

### 5.3 Model

Baseline:

- Logistic Regression
- Linear SVM
- Random Forest

주요 모델:

- Linear SVM

### 5.4 Evaluation

Gold test set 기준:

- Precision
- Recall
- Micro F1
- Macro F1
- relation별 F1
- feature ablation

Ablation:

```text
all features
- dependency path
- entity type
- between words
- semantic feature
```

## 6. Kernel-Based RE

### 6.1 Purpose

명시적 feature vector 대신 문장 구조 유사도를 kernel로 비교한다.

### 6.2 Kernel Types

Composite kernel:

```text
K = alpha * sequence_kernel
  + beta  * tree_kernel
  + gamma * entity_type_kernel
```

### 6.3 Sequence Kernel

사용 후보:

- subsequence kernel
- entity-between token sequence kernel
- shortest context sequence kernel

특징:

- `D-4비자 신청 서류`와 `통합신청서` 사이의 표면 패턴 유사도 측정
- 반복되는 행/목록 구조에 강함

### 6.4 Tree Kernel

사용 후보:

- dependency tree kernel
- subtree kernel
- shortest dependency path tree kernel

필요 조건:

- Korean dependency parser 필요
- 문장 단위 parsing 필요

현실적 구현:

1. 먼저 dependency path를 문자열로 직렬화한다.
2. path sequence kernel로 시작한다.
3. 이후 tree kernel로 확장한다.

### 6.5 Evaluation

- SVM with precomputed kernel
- Macro F1
- relation별 F1
- 학습 샘플 수에 따른 성능 변화
- sequence-only vs tree-only vs composite 비교

## 7. DIPRE

### 7.1 Purpose

적은 seed tuple로 relation pattern을 확장하는 semi-supervised RE 성능을 측정한다.

### 7.2 Seed Data

DIPRE는 반드시 seed가 필요하다.

Seed 예시:

```json
{"head": "D-4비자 신청 서류 (어학연수 비자)", "tail": "입학원서", "relation": "requires_document"}
{"head": "D-4비자 신청 서류 (어학연수 비자)", "tail": "자기소개서", "relation": "requires_document"}
{"head": "2026학년도 1학기 한국어능력 졸업인증 대체시험", "tail": "2026.05.30.(토) 10:00", "relation": "has_deadline"}
{"head": "국제교류처 외국인유학생서비스팀", "tail": "rachel39@gachon.ac.kr", "relation": "HAS_CONTACT_EMAIL"}
```

Seed 수:

- relation별 3개
- relation별 5개
- relation별 10개

### 7.3 Experiment

비교 기준:

```text
Supervised model F1에 도달하기 위해 필요한 labeled sample 수
vs
DIPRE seed 수
```

측정:

```text
label_reduction = 1 - (DIPRE_seed_count / supervised_labeled_count)
```

예:

```text
Supervised F1 0.75 도달: 500 labels
DIPRE F1 0.75 도달: 50 seeds
Label reduction = 90%
```

### 7.4 Risks

- semantic drift
- 너무 일반적인 pattern 확산
- `mentions` relation처럼 의미가 넓은 label에서 noise 증가

## 8. Snowball

### 8.1 Purpose

DIPRE의 pattern drift를 confidence score로 제어한다.

### 8.2 Confidence Score

Snowball confidence 구성:

```text
confidence(tuple, pattern)
= pattern_confidence
* entity_type_compatibility
* context_similarity
* frequency_score
```

요소:

| Score | Meaning |
|---|---|
| `pattern_confidence` | 해당 pattern이 기존 seed와 일치한 비율 |
| `entity_type_compatibility` | relation별 허용 entity type pair 여부 |
| `context_similarity` | seed context와 후보 context embedding 유사도 |
| `frequency_score` | 여러 pattern에서 반복 발견되는지 |

### 8.3 Experiment

Threshold별 성능:

```text
confidence >= 0.5
confidence >= 0.6
confidence >= 0.7
confidence >= 0.8
confidence >= 0.9
```

비교:

```text
DIPRE F1
Snowball F1
Improvement = (Snowball_F1 - DIPRE_F1) / DIPRE_F1
```

결과 그래프:

- confidence threshold vs precision
- confidence threshold vs recall
- confidence threshold vs F1
- seed count vs F1

## 9. Unsupervised Clustering RE

### 9.1 Purpose

GT 없이 relation structure가 얼마나 자연스럽게 군집화되는지 확인한다.

### 9.2 Method

입력:

- entity-marked sentence
- between words
- sentence embedding
- entity type pair

모델:

- TF-IDF + KMeans
- Sentence embedding + Agglomerative clustering
- HDBSCAN

### 9.3 Evaluation

Clustering은 일반 F1만으로 평가하지 않는다.

Gold label이 있을 때:

- Purity
- V-Measure
- Adjusted Rand Index
- Normalized Mutual Information

Gold label 없이:

- cluster sample inspection
- top terms per cluster
- relation discovery quality

### 9.4 Output

- cluster별 대표 문장
- cluster별 top terms
- cluster → relation alignment table

## 10. CNN-Based RE

### 10.1 Purpose

local n-gram pattern을 잘 잡는지 확인한다.

### 10.2 Recommended Model

PCNN, Piecewise CNN.

이유:

- RE에서는 head/tail entity 위치가 중요하다.
- 일반 max-pooling은 위치 정보를 잃는다.
- PCNN은 문장을 entity 기준 3구간으로 나눠 pooling한다.

구간:

```text
before head
between head and tail
after tail
```

### 10.3 Inputs

- token embedding
- position embedding to head
- position embedding to tail
- entity type embedding

### 10.4 Evaluation

- Macro F1
- relation별 F1
- local trigger relation 성능
  - `HAS_FEE`
  - `HAS_CONTACT_EMAIL`
  - `requires_document`

## 11. LSTM-Based RE

### 11.1 Purpose

긴 문맥과 순차 정보를 CNN보다 잘 잡는지 확인한다.

### 11.2 Recommended Model

- BiLSTM
- BiLSTM + entity position embedding

### 11.3 Strength

- 긴 문장
- head와 tail 사이 거리가 먼 경우
- 표 설명문처럼 context가 긴 경우

### 11.4 Evaluation

- distance bucket별 F1

```text
distance 0-5
distance 6-15
distance 16-30
distance 31+
```

CNN과 비교하여 long-distance relation에서 이점이 있는지 본다.

## 12. Attention-Based RE

### 12.1 Purpose

BiLSTM에 attention을 추가하여 relation trigger에 집중하는지 확인한다.

### 12.2 Model

- BiLSTM + word attention
- entity-aware attention

### 12.3 Analysis

Attention heatmap을 생성한다.

확인할 trigger:

- 제출
- 필요
- 신청
- 마감
- 응시료
- 이메일
- 문의
- 다운로드

### 12.4 Evaluation

- Macro F1
- attention visualization
- relation별 top attention token

Attention이 설명 가능성을 제공하는지 확인한다.

## 13. Transformer and LLM Evaluation

### 13.1 Role

Transformer/LLM은 두 가지 관점으로 평가한다.

1. RE 모델로서의 Transformer fine-tuning 성능
2. Silver/GT 생성 도구로서의 LLM 품질

### 13.2 Transformer Fine-Tuning

모델 후보:

- `klue/bert-base`
- `klue/roberta-base`
- multilingual BERT
- KoELECTRA

입력 형식:

```text
[E1] D-4비자 신청 서류 (어학연수 비자) [/E1] ... [E2] 입학원서 [/E2]
```

평가:

- Gold test F1
- relation별 F1
- low-resource setting 성능

### 13.3 LLM as Silver Label Generator

LLM output을 Gold set과 비교한다.

평가:

- LLM Precision
- LLM Recall
- LLM F1
- hallucination rate
- wrong head/tail rate
- wrong relation rate
- missing relation rate

오류 유형:

| Error Type | Description |
|---|---|
| wrong_head | head entity가 잘못 잡힘 |
| wrong_tail | tail entity가 잘못 잡힘 |
| wrong_relation | relation label이 틀림 |
| missing_relation | 추출해야 할 relation 누락 |
| hallucinated_relation | 원문에 없는 relation 생성 |
| over_split | 하나의 문서를 너무 잘게 쪼갬 |
| under_split | 여러 문서를 하나로 합침 |

## 14. Evaluation Metrics

### 14.1 Supervised / Semi-supervised / Deep Learning

필수 지표:

- Precision
- Recall
- Micro F1
- Macro F1
- relation별 F1

추가 지표:

- PR AUC
- confusion matrix
- label efficiency
- inference time

### 14.2 Label Efficiency

semi-supervised의 핵심 지표:

```text
label_reduction_percent
= (1 - required_seed_count / required_supervised_label_count) * 100
```

### 14.3 Snowball Improvement

```text
snowball_improvement_percent
= (Snowball_F1 - DIPRE_F1) / DIPRE_F1 * 100
```

### 14.4 Clustering

필수:

- Purity
- V-Measure
- ARI
- NMI

## 15. Experimental Controls

공정 비교를 위해 다음을 고정한다.

- 동일한 Gold test set
- 동일한 relation label set
- 동일한 candidate pair generation
- 동일한 train/dev/test split
- 동일한 negative sampling policy
- 동일한 evaluation script

모델별 차이는 feature/modeling 방식에서만 발생해야 한다.

## 16. Experiment Matrix

| Method | Train Data | Test Data | Main Metric | Key Analysis |
|---|---|---|---|---|
| LLM Silver | raw documents | Gold set | F1, hallucination rate | GT 생성 품질 |
| Feature-based | Gold/Silver train | Gold test | Macro F1 | feature ablation |
| Kernel-based | Gold/Silver train | Gold test | Macro F1 | sequence vs tree |
| DIPRE | seed tuples | Gold test | F1, label reduction | seed efficiency |
| Snowball | seed + unlabeled | Gold test | F1, confidence curve | noise control |
| Clustering | unlabeled | Gold test | V-measure, purity | relation discovery |
| CNN/PCNN | Gold/Silver train | Gold test | Macro F1 | local pattern |
| BiLSTM | Gold/Silver train | Gold test | Macro F1 | long dependency |
| BiLSTM+Attention | Gold/Silver train | Gold test | Macro F1 + heatmap | interpretability |
| Transformer | Gold/Silver train | Gold test | Macro F1 | upper neural baseline |

## 17. Required Output Artifacts

최종 산출물:

```text
data/re_gold/
data/re_silver/
reports/re_comparison.md
reports/label_efficiency.md
reports/error_analysis.md
reports/confusion_matrices/
reports/attention_heatmaps/
reports/clustering/
```

모델별 산출물:

```text
reports/feature_based/
reports/kernel_based/
reports/dipre/
reports/snowball/
reports/unsupervised/
reports/cnn/
reports/lstm/
reports/attention/
reports/transformer/
reports/llm_silver_eval/
```

## 18. Immediate Execution Order

시간 단위 계획은 사용하지 않는다. 오늘 안에 모든 실험을 진행한다는 전제에서 실행 순서만 정의한다.

1. `data/re_fixed_v6`를 최신 후보 기준으로 확정한다.
2. LLM으로 `data/re_silver/silver.jsonl`을 생성한다.
3. `data/re_fixed_v6/labeling_by_relation/*.csv` 일부를 사람이 검수하여 `data/re_gold/gold.jsonl`을 만든다.
4. Gold test set을 고정한다.
5. Feature-based baseline을 실행한다.
6. Kernel-based sequence kernel을 실행한다.
7. Dependency parser를 붙여 dependency/tree feature를 생성한다.
8. Kernel-based composite kernel을 실행한다.
9. relation별 seed file을 만든다.
10. DIPRE를 실행한다.
11. Snowball confidence score를 추가하고 threshold별 실험을 실행한다.
12. Clustering unsupervised 실험을 실행한다.
13. CNN/PCNN을 실행한다.
14. BiLSTM을 실행한다.
15. BiLSTM+Attention을 실행하고 heatmap을 생성한다.
16. Transformer fine-tuning을 실행한다.
17. LLM Silver output을 Gold set과 비교한다.
18. 모든 결과를 단일 비교표와 그래프로 정리한다.

## 19. Main Risks and Mitigation

| Risk | Impact | Mitigation |
|---|---|---|
| LLM label hallucination | 평가 왜곡 | Gold test set 별도 구축 |
| candidate pair 오류 | 모든 모델 성능 하락 | 후보 생성 규칙 개선, manual inspection |
| relation imbalance | Macro F1 저하 | relation별 sampling, class weight |
| semi-supervised drift | DIPRE/Snowball precision 하락 | confidence threshold, entity type constraint |
| dependency parser 오류 | kernel feature 품질 하락 | dependency feature ablation 포함 |
| neural model data 부족 | overfitting | Silver pretraining + Gold fine-tuning |

## 20. Conclusion

현재 데이터는 GT가 아니라 후보/weak label이다.

정상적인 실험 설계는 다음 구조가 되어야 한다.

```text
Weak candidate generation
→ LLM Silver labeling
→ Human-verified Gold test set
→ 동일 Gold test set으로 모든 방법론 평가
```

이 구조를 지키면 각 방법론의 성능 차이, 라벨링 비용 차이, LLM 생성 데이터의 실제 품질을 방어 가능하게 비교할 수 있다.
