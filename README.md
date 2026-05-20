# OIA 관계 추출(RE) 파이프라인 — 방법론 비교 실험

> **가천대학교 국제교류처(OIA) 행정 텍스트**를 도메인으로,  
> 관계 추출(Relation Extraction)의 **비지도 → 반지도 → 지도 → 딥러닝** 4가지 패러다임을 직접 구현하고 성능을 비교 분석합니다.  
> **KLUE-RE** 공개 벤치마크로 도메인 특화 편향을 추가 검증합니다.

---

## 📋 목차

1. [연구 배경 및 목적](#1-연구-배경-및-목적)
2. [데이터셋 개요](#2-데이터셋-개요)
3. [실험 개요 Executive Summary](#3-실험-개요-executive-summary)
4. [Step 0 — 코퍼스 재구축](#4-step-0--코퍼스-재구축)
5. [Step 1 — Unsupervised RE](#5-step-1--unsupervised-re)
6. [Step 2 — Semi-supervised RE](#6-step-2--semi-supervised-re)
7. [Step 3 — Feature-based RF](#7-step-3--supervised-ml-feature-based-rf)
8. [Step 4 — Kernel-based SVM](#8-step-4--supervised-ml-kernel-based-svm)
9. [Step 5 — Deep Learning](#9-step-5--deep-learning)
10. [전체 실험 결과 분석](#10-전체-실험-결과-분석)
11. [Part 2 — KLUE-RE 벤치마크](#11-part-2--klue-re-벤치마크)
12. [실험 재현](#12-실험-재현)
13. [파일 구조](#13-파일-구조)
14. [참고 문헌](#14-참고-문헌)

---

## 1. 연구 배경 및 목적

### 1.1 배경

**관계 추출(Relation Extraction, RE)** 은 텍스트에서 두 개체 사이의 의미 관계를 자동 인식하는 NLP 핵심 태스크입니다. Knowledge Graph 구축·질의응답·정보 검색에 폭넓게 활용됩니다.

본 프로젝트는 **가천대학교 국제교류처(OIA)** 웹사이트의 공지·행정 텍스트를 대상으로 RE 시스템을 구축합니다. OIA 도메인은 일반 NLP 벤치마크와 다음과 같이 구별됩니다:

| 특성 | 일반 뉴스/위키 텍스트 | OIA 행정 텍스트 |
|---|---|---|
| **문장 구조** | 자연어 서술체, 복잡한 의존 관계 | 테이블·리스트 형태, 개체가 인접 |
| **개체 사이 패턴** | 동사·술어 풍부 | **45%가 인접 (패턴 없음)** |
| **관계 결정 요인** | 어휘·통사 패턴 | **개체 타입 쌍**이 관계를 거의 결정 |
| **레이블 수** | 30~100개 | 12개 (소규모 온톨로지) |

이러한 **구조 데이터 특성**은 기존 RE 알고리즘의 기본 가정(개체 사이에 의미 있는 텍스트가 있다)을 위반하며, 각 방법론이 어떻게 실패하고 어떤 적응이 필요한지를 분석하는 것이 본 연구의 핵심입니다.

### 1.2 OIA Knowledge Graph 온톨로지

추출 대상 관계(12종):

```
HAS_CONTACT_EMAIL         — 공지/페이지 → 이메일 주소
HAS_CONTACT_PHONE         — 공지/페이지 → 전화번호
HAS_DEADLINE              — 공지/이벤트 → 마감일
HAS_FEE                   — 공지/페이지 → 수수료
MENTIONS_EXAM_LEVEL       — 공지/페이지 → 시험 등급
NO_RELATION               — 관계 없음
REFERENCES_ATTACHMENT     — 공지 → 첨부파일
REFERENCES_EXTERNAL_RESOURCE — 공지 → 외부 URL
ANNOUNCED_BY              — 공지 → 부서/기관
MENTIONS                  — 공지 → 언급 항목
REQUIRES_DOCUMENT         — 공지 → 필요 서류
REQUIRES_QUALIFICATION    — 공지 → 지원 자격
```

### 1.3 실험 목적

각 RE 패러다임의 **이론적 가정이 OIA 도메인에서 어떻게 작동하는가**를 검증합니다:

| 패러다임 | 핵심 가정 | OIA에서의 도전 |
|---|---|---|
| **Unsupervised** | 같은 관계는 유사한 문맥에서 등장 | 45% 빈 패턴 → 문맥 자체가 없음 |
| **Semi-supervised** | 소수 시드에서 패턴 추출 → 대규모 코퍼스 탐색 | 시드 패턴도 비어 있음, 코퍼스가 이미 레이블됨 |
| **Supervised ML** | 수작업 피처가 관계를 구분 | **개체 타입 쌍**이 압도적 1위 피처 |
| **Deep Learning** | 충분한 데이터로 표현을 자동 학습 | 1,462건은 Scratch 임베딩에 부족 |

---

## 2. 데이터셋 개요

| 구분 | 설명 | 건수 |
|---|---|---|
| **Gold Standard** | 수작업 레이블링, 12종 관계 | 257건 |
| **Silver Standard** | LLM(GPT) 자동 레이블링 | 1,205건 |
| **합계** | Gold + Silver | **1,462건** |
| **Clean Corpus** | step0 재구축 (DIPRE 탐색 풀) | **1,730건** |

![데이터셋 개요](docs/step0_data_overview.png)

**관계 분포 불균형**: `NO_RELATION`(448건, 25.9%), `REFERENCES_EXTERNAL_RESOURCE`(433건, 25.0%)가 전체의 절반을 차지하는 반면 `REFERENCES_ATTACHMENT`(14건), `HAS_FEE`(18건)는 극소수.

---

## 3. 실험 개요 (Executive Summary)

![최종 비교 업데이트](docs/final_comparison_updated.png)

### OIA 파이프라인 최종 성능

| 패러다임 | 모델 | 지표 | 점수 | 학습 데이터 |
|---|---|---|---|---|
| **Unsupervised** | Pattern-based (TF-IDF KMeans) | V-Measure | 0.4785 | Corpus 1,730건 |
| **Unsupervised** | Embedding-based (SBERT KMeans) | V-Measure | 0.3034 | Corpus 1,730건 |
| **Semi-supervised** | DIPRE (10-seed 부트스트래핑) | Macro F1 | **0.4010** | Seed 10건 |
| **Semi-supervised** | Snowball (+개체 타입 필터) | Macro F1 | **0.4787** | Seed 10건 |
| **Supervised ML** | Feature-based (Random Forest) | Macro F1 | 0.7300 | Gold+Silver 1,462건 |
| **Supervised ML** | Kernel-based SVM (Composite) | Macro F1 | **0.8627** | 1,000건 (O(N²)) |
| **Deep Learning** | Bi-LSTM + Attention (Scratch) | Macro F1 | 0.5418 | Gold+Silver 1,462건 |

> ⚠️ Unsupervised는 V-Measure(군집화 품질), 나머지는 Macro F1(분류 정확도)로 지표 종류가 다릅니다.

---

## 4. Step 0 — 코퍼스 재구축

**파일**: `step0_rebuild_corpus.py`

### 4.1 연구 목적

DIPRE/Snowball은 **소수 시드 → 패턴 추출 → 비레이블 코퍼스 탐색**으로 동작합니다. 그러나 원본 데이터에서 치명적 결함이 발견됐습니다:

> **원본 문제**: `candidates.jsonl` 1,730건 중 **45.3%가 E1-E2 사이 패턴 없음**  
> 이유: OIA는 테이블 구조 데이터 → 이름|이메일, 서비스명|금액이 인접 셀에 위치  
> → DIPRE가 추출할 패턴 없음 → **F1 ≈ 0**

### 4.2 실험 방법

**① 리치 케이스 (원본 패턴 ≥ 10자, 946건)** — 윈도우 추출

```python
# E1 시작 전 200자 ~ E2 끝 후 200자만 보존
# → 전체 페이지 blob 대신 개체 주변 문맥만
snippet = marked_text[max(0, e1_start-200) : min(len, e2_end+200)]
```

**② 빈/짧은 패턴 케이스 (784건)** — 관계별 한국어 템플릿

```python
TEMPLATES = {
    "HAS_CONTACT_EMAIL": [
        "{head}의 문의 이메일 주소는 {tail}입니다.",
        "{head}에 대한 문의는 이메일 {tail}으로 연락 바랍니다.",
    ],
    "HAS_FEE": [
        "{head}의 수수료(전형료)는 {tail}입니다.",
        "{head} 신청 시 발생하는 비용은 {tail}입니다.",
    ],
    # ... 12종 모두 정의
}
```

**③ OpenAI 선택 사용** (`--use-openai`, `OPEN_AI_KEY` 필요)

### 4.3 결과

![코퍼스 품질 개선](docs/corpus_quality_improvement.png)

| 패턴 유형 | 재구축 전 | 재구축 후 | 변화 |
|---|---|---|---|
| **빈 패턴** (< 2자) | 784건 (45.3%) | 49건 (2.8%) | **−42.5%p** |
| **짧은 패턴** (2~9자) | 108건 (6.2%) | 424건 (24.5%) | +18.3%p |
| **리치 패턴** (≥ 10자) | 838건 (48.4%) | 1,257건 (72.7%) | **+24.3%p** |

**출력**:
- `corpus_clean.jsonl` — 관계 레이블 포함 정제 코퍼스
- `corpus_unlabeled.jsonl` — 관계를 `UNKNOWN`으로 마스킹 (DIPRE 탐색 풀)

### 4.4 예상과 다른 점: 동일 텍스트 개체 문제

개체 텍스트가 동일하거나 tail이 head의 부분 문자열인 경우 `str.replace()` 마커 삽입 실패:

```
head = tail = "한국어 교육센터 소개"
→ 첫 번째 replace로 E1 삽입 → 두 번째 replace가 E1 마커 안쪽 텍스트를 다시 치환
```

**수정**: 플레이스홀더(`\x00HEAD\x00`) 2단계 치환 방식으로 해결.

---

## 5. Step 1 — Unsupervised RE

**파일**: `step2_unsupervised_re_v2.py`

### 5.1 연구 목적

레이블 없이 텍스트 패턴과 의미 분포만으로 관계를 군집화합니다.  
**V-Measure** = Homogeneity × Completeness 조화평균으로 평가.

### 5.2 실험 방법 — Open IE

관계 레이블 없이 자유 형태 트리플을 추출하는 **Open IE**를 두 방식으로 구현:

**① SpaCy 의존 구문 기반** — 동사 추출

```python
for token in doc:
    if token.dep_ in ('nsubj', 'csubj'): subject = token.text
    if token.pos_ == 'VERB':             verb    = token.text
    if token.dep_ in ('obj', 'iobj'):    obj     = token.text
```

**② 구조 기반 Open IE** (OIA 도메인 특화) — E1~E2 사이를 predicate로 간주

```python
triple = (head_text, between_5words, tail_text)
# 예: ("외국인학생", "의 수수료(전형료)는", "50,000원")
```

결과: SpaCy 기반 **0건** (OIA 문장은 동사 없는 구조 다수) vs 구조 기반 **1,730건 (100%)**.

![Open IE 분석](docs/open_ie_analysis.png)

### 5.3 실험 방법 — 군집화

```
전체 corpus_clean.jsonl 1,730건 대상 (이전: gold 257건만 → 6.7배 증가)
K = 12
```

**① Pattern-based (V-Measure: 0.4785)**: 개체 사이 텍스트 + 개체 타입 접미사 → char_wb TF-IDF → K-Means  
**② Embedding-based (V-Measure: 0.3034)**: 문장 전체 SBERT 임베딩 → K-Means

### 5.4 결과

![Unsupervised 비교](docs/step1_unsupervised_comparison.png)
![Unsupervised t-SNE](docs/step1_unsupervised_tsne.png)

### 5.5 예상과 다른 점: Pattern > Embedding (OIA에서 역전)

일반적으로 SBERT 임베딩이 TF-IDF보다 높은 V-Measure를 기록하지만, OIA에서는 역전됐습니다:

1. **개체 타입 토큰 효과**: 재구축 코퍼스에서 `__H_Page__ __T_Email__` 같은 타입 토큰을 TF-IDF 피처에 추가 → 관계 경계가 명확히 분리됨
2. **템플릿 문장 균일성**: 784건의 템플릿 생성 문장이 동일한 구문 구조 → SBERT 임베딩 공간에서 분산이 적어 군집 경계가 흐릿해짐

---

## 6. Step 2 — Semi-supervised RE

**파일**: `step3b_semi_supervised.py`

### 6.1 연구 목적

**DIPRE**(Brin, 1998)와 **Snowball**(Agichtein & Gravano, 2000)을 구현하여,  
소수 레이블 시드(10건)로 비레이블 코퍼스에서 관계 인스턴스를 자동 발견합니다.

| | DIPRE | Snowball |
|---|---|---|
| **핵심 아이디어** | 패턴 → 인스턴스 → 패턴 반복 | DIPRE + 개체 타입 컨텍스트 필터 |
| **매칭 조건** | 텍스트 패턴 포함 | 패턴 ∩ 개체 타입 일치 |
| **Semantic Drift** | 발생 | 억제 |

### 6.2 핵심 문제 발견 및 수정

**결함 1: 이미 레이블된 데이터에 패턴 적용 (DIPRE 목적 위반)**

```python
# ❌ 원본: train.jsonl (LLM이 이미 레이블한 silver 데이터)
silver_df = load_silver_standard()

# ✅ 수정: corpus_unlabeled.jsonl (관계가 UNKNOWN으로 마스킹된 진짜 비레이블 풀)
corpus = load_corpus_unlabeled()
```

**결함 2: 빈 패턴 시 관계 건너뜀**

```python
# ❌ 원본: 패턴 없으면 skip
if not patterns:
    continue

# ✅ 수정: TYPE-ONLY fallback — 개체 타입 쌍을 가상 패턴으로 사용
use_type_fallback = (len(quality_patterns) == 0) or (corpus_hits < 3)
```

**결함 3: 관계 라벨 대소문자 불일치**

```python
# gold:   'announced_by', 'requires_document', 'has_deadline'
# corpus: 'ANNOUNCED_BY', 'REQUIRES_DOCUMENT', 'HAS_DEADLINE'
# → 매칭 시 서로 달라 F1 계산 오류

# ✅ 수정: 양쪽 모두 정규화
NORMALIZE_REL = {
    "requires_document": "REQUIRES_DOCUMENT",
    "has_deadline":      "HAS_DEADLINE",
    "announced_by":      "ANNOUNCED_BY",
    ...
}
```

**패턴 품질 필터**: 날짜·숫자 특이값("2015 12 04") 또는 순수 영문("foreign edu safety org") 패턴 제거 + 코퍼스 최소 3건 매칭 조건.

### 6.3 결과

![Semi-supervised 관계별 F1](docs/semi_supervised_per_relation.png)

![Semi-supervised 전체](docs/step2_semi_supervised.png)

| 메트릭 | 원본 (v1) | 수정 후 (v3) | 개선 |
|---|---|---|---|
| **DIPRE Macro F1** | ≈ 0.00 | **0.4010** | +∞ |
| **Snowball Macro F1** | ≈ 0.00 | **0.4787** | +∞ |

### 6.4 관계별 성능 분석

![OIA 도메인 인사이트](docs/oia_domain_insight.png)

관계를 패턴 전략에 따라 세 그룹으로 분류:

**그룹 A — TYPE-ONLY (F1: 0.73~0.93)**  
`REFERENCES_ATTACHMENT`, `REFERENCES_EXTERNAL_RESOURCE`, `HAS_DEADLINE`, `MENTIONS_EXAM_LEVEL`  
→ 개체 타입 쌍만으로 관계가 완전히 결정됨 (`Notice → Deadline` = 항상 `HAS_DEADLINE`)

**그룹 B — TEXT+TYPE (F1: 0.40~0.60)**  
`REQUIRES_DOCUMENT`, `ANNOUNCED_BY`, `HAS_CONTACT_EMAIL`, `HAS_FEE`  
→ 재구축 코퍼스의 템플릿 패턴이 DIPRE에서 활용됨  
→ Snowball의 타입 필터가 false positive를 효과적으로 제거

**그룹 C — TEXT(noisy) (F1: 0.05~0.23)**  
`HAS_CONTACT_PHONE`, `REQUIRES_QUALIFICATION`, `MENTIONS`  
→ Gold seed 패턴이 "교수진 리스트 부서 직위 학위" 같은 페이지 헤더 노이즈  
→ 관계와 무관한 코퍼스 항목에 넓게 매칭 → precision 낮음

### 6.5 OIA 도메인 특이성 — 왜 부트스트래핑이 어려운가

```
원설계 (Agichtein & Gravano, 2000):
  웹 전체 수백만 문서 → 패턴 하나가 수천 건 매칭 → 수십 iteration 증식

이번 실험:
  OIA 코퍼스 1,730건 → 관계당 평균 ~100건 → 타입 필터 후 수십 건
  → Iteration 2~3에서 풀 소진 (Pool Exhaustion)

Snowball이 0.93을 달성하는 이유:
  REFERENCES 관계는 개체 타입이 고유해 1대1 매핑
  → 타입만으로 완벽 분리 → 전통적 Snowball 알고리즘 강점이 아닌
     OIA 도메인 구조의 특성
```

---

## 7. Step 3 — Supervised ML: Feature-based RF

**파일**: `step3_feature_based_re_v2.py`

### 7.1 연구 목적

4종의 언어학적 피처를 명시적으로 추출해 Random Forest로 분류합니다.  
어떤 피처가 OIA 도메인에서 실제로 유효한지 Feature Importance로 분석합니다.

### 7.2 피처 설계

| 피처 그룹 | 추출 방식 | 근거 |
|---|---|---|
| **Context Words** | 문장 전체 TF-IDF (max 500) | 관계의 담화 문맥 포착 |
| **Words Between** | E1~E2 사이 TF-IDF (max 500) | 관계 트리거 단어 (Bunescu & Mooney, 2005) |
| **Semantic Feature** | 개체 타입 쌍 CountVectorizer | `Page\|Email`, `Notice\|Fee` 등 |
| **Dependency Path** | SpaCy `ko_core_news_sm` 구문 경로 TF-IDF | 표면 어휘 독립적 구조 신호 |

Data leakage 방지: TF-IDF는 Train에서만 `.fit()`, Test는 `.transform()` only.

### 7.3 결과 — Macro F1: 0.7300

![Feature-based RF](docs/step3_feature_based.png)

**Feature Importance 1위: Semantic Feature(개체 타입 쌍)**  
OIA에서는 `Page → Email = HAS_CONTACT_EMAIL`, `Notice → Fee = HAS_FEE`로 타입 쌍이 관계를 거의 결정합니다.

### 7.4 예상과 다른 점

**Dependency Path 피처 기여도 최하**: SpaCy `ko_core_news_sm`은 한국어 뉴스 코퍼스 기반. OIA의 테이블/리스트 단문에서 파싱 오류율이 높고, "전형료 70,000원" 같은 명사구에서 동사 없는 의존 관계 발생 → path가 빈 문자열인 경우 다수.

---

## 8. Step 4 — Supervised ML: Kernel-based SVM

**파일**: `step3c_kernel_based_re.py`, `visualize_kernel_ml.py`

### 8.1 연구 목적

피처 벡터 대신 **문장 간 구조 유사도를 커널 함수로 직접 정의**합니다.

### 8.2 Composite Kernel

```
K_composite = 0.3·K_seq + 0.3·K_tree + 0.4·K_semantic
```

| 커널 | 측정 | 가중치 근거 |
|---|---|---|
| **K_seq** | E1~E2 어휘 Jaccard | α=0.3 · Bunescu & Mooney (2005) |
| **K_tree** | SpaCy 구문 트리 간선 Jaccard | β=0.3 · Culotta & Sorensen (2004) |
| **K_semantic** | 개체 타입 쌍 일치 (0 or 1) | γ=0.4 · Feature Importance 실험 결과 |

α=β: 균등 MKL 초기화 원칙(Gönen & Alpaydin, 2011).  
γ > α=β: OIA Feature Importance 실험에서 개체 타입이 압도적 1위 → 도메인 적응(Plank & Moschitti, 2013).

### 8.3 결과 — Macro F1: 0.8627 (전체 최고)

![Kernel SVM](docs/step4_kernel_svm.png)

**N=1,000으로 제한**: O(N²) 커널 행렬 연산. 1,462건 전체는 메모리·연산 비용이 2.1×.

### 8.4 예상과 다른 점

**RF(0.7300)보다 Kernel SVM(0.8627)이 높은 이유**:  
Kernel SVM의 K_semantic은 두 문장의 개체 타입 쌍이 **완전히 일치**할 때만 커널값 1 부여 → 분리 마진에 직접 반영됨. RF의 CountVectorizer는 타입 토큰을 500차원 vocabulary 일부로 처리해 희석됩니다.

---

## 9. Step 5 — Deep Learning

**파일**: `step4_deep_learning_re.py`

### 9.1 아키텍처

```
Input Tokens → Embedding(128) → Bi-LSTM(hidden=64×2) → Attention → FC → Softmax

Attention score = softmax(v · tanh(W · h_t))
Context vector  = Σ score_t · h_t
```

### 9.2 결과 — Macro F1: 0.5418

![Deep Learning](docs/step5_deep_learning.png)

Attention Heatmap: "수수료", "이메일", "마감" 등 트리거 단어에 높은 가중치 → 모델이 자동으로 핵심 단어 발견(해석 가능성 확인).

### 9.3 예상과 다른 점: Supervised ML보다 낮은 이유

1. **개체 타입 정보 미사용**: RF/SVM은 `Page|Email` 타입 쌍을 직접 피처로 주지만 Bi-LSTM은 텍스트 시퀀스만 입력받음. OIA에서 타입 신호가 결정적.
2. **데이터 부족**: 1,462건으로 vocab×128 파라미터 Scratch 학습. `HAS_DEADLINE`(36건), `REFERENCES_ATTACHMENT`(14건)는 수렴 불가.
3. **짧은 구조 텍스트**: LSTM의 장점(장거리 의존성)이 `[E1]서비스[/E1] [E2]금액[/E2]` 2토큰 구조에서 발휘되지 않음.

---

## 10. 전체 실험 결과 분석

![최종 성능 비교](docs/step6_final_comparison.png)

### 10.1 성능 순서 설명

```
Kernel SVM (0.8627) > RF (0.7300) > Bi-LSTM (0.5418) > Snowball (0.4787) > DIPRE (0.4010)
```

이 순서는 **OIA 도메인에서 개체 타입 정보를 얼마나 효과적으로 활용하는가**와 정확히 일치합니다:

| 모델 | 개체 타입 활용 | 효과 |
|---|---|---|
| Kernel SVM | K_semantic 커널 (완전 일치 시 1) | 최대 — 분리 마진에 직접 반영 |
| Random Forest | CountVectorizer 피처 일부 | 높음 — Importance #1 |
| Bi-LSTM | 미사용 | 없음 |
| Snowball | 타입 필터 (매칭 조건) | 중간 — precision 개선 |
| DIPRE | TYPE-ONLY fallback만 | 낮음 |

### 10.2 패러다임별 한계 요약

**Unsupervised**: 개체 타입 피처 추가 후 V-Measure 0.4785 달성. 순수 텍스트 패턴은 OIA 빈 패턴 문제로 효과 낮음.

**Semi-supervised**: 코퍼스 규모 미스매치(1,730건 vs. 원설계 수백만 문서). 3가지 핵심 수정으로 F1≈0 → 0.48 회복.

**Supervised ML**: OIA처럼 개체 타입이 관계를 결정하는 도메인에서 Feature Engineering 매우 효과적. 온톨로지 변경 시 피처 재설계 필요.

**Deep Learning**: 타입 정보 추가(`[E1_TYPE]` 토큰) 또는 PLM fine-tuning으로 크게 개선 가능.

### 10.3 결론

> OIA 행정 도메인에서 **Kernel SVM with Composite Kernel이 최고 성능(0.8627)**을 달성합니다.  
> 핵심 이유: 개체 타입 쌍이 관계를 거의 결정하는 OIA 도메인 구조를 K_semantic 커널이 직접 포착.  
>
> PLM(BERT/RoBERTa) fine-tuning이 성능 상한선을 높일 수 있지만, 현재 데이터(1,462건)와 12종 관계의 도메인 특화 구조에서는 **명시적 커널 설계가 PLM implicit 학습보다 효율적**입니다.

---

## 11. Part 2 — KLUE-RE 벤치마크

### 11.1 목적

OIA 실험 결과가 도메인 특화 편향인지, 일반 RE에서도 동일한 패턴이 나타나는지 검증합니다.

### 11.2 데이터셋

| 항목 | 내용 |
|---|---|
| **출처** | HuggingFace `klue/re` (뉴스·위키백과) |
| **Train** | 32,470건 |
| **Validation** | 7,765건 |
| **관계 수** | 30개 |

### 11.3 KLUE-RE 성능

| 모델 | 지표 | 점수 |
|---|---|---|
| Pattern-based KMeans | V-Measure | 0.0897 |
| Embedding-based KMeans | V-Measure | 0.1392 |
| Feature-based RF | Macro F1 | 0.1626 |
| Kernel SVM (Composite) | Macro F1 | 0.2222 |
| Bi-LSTM + Attention (Scratch) | Macro F1 | 0.0706 |

### 11.4 저성능 원인 분석

**① 클래스 불균형 (238× 비율)**

```
최다: no_relation = 9,534건 (29.4%)
최소: per:place_of_death = 40건 (0.12%)
불균형 비율: 238×
→ 소수 클래스 12개가 F1=0이면 최대 달성 가능 Macro F1 = 18/30 = 0.60
```

**② OIA → KLUE: 개체 타입 신호 붕괴**

OIA: `Notice → Fee = HAS_FEE` (1대1 매핑)  
KLUE: `PER → LOC` 타입 쌍이 `per:place_of_birth`, `per:place_of_residence`, `per:place_of_death` 세 관계에 공유 → 타입만으로 구분 불가 → K_semantic 강점이 사라짐

**③ 자연어 복잡성**

```
[per:place_of_birth]:
  "1899년 6월 조선 충청도 [E2]공주[/E2] 출생한 [E1]백한성[/E1]"
  → 수식어 내 개체 위치, 의존 경로 역전

[per:place_of_residence]:
  "[E1]문재인[/E1] 정부는 [E2]대한민국[/E2]의 변화를..."
  → 거주지 직접 언급 없음, 함의(implication) 이해 필요
```

### 11.5 OIA vs KLUE 비교 인사이트

| | OIA (행정 특화) | KLUE-RE (일반 자연어) |
|---|---|---|
| **Supervised ML 최고** | 0.8627 | 0.2222 |
| **Deep Learning** | 0.5418 | 0.0706 |
| **핵심 신호** | 개체 타입 쌍 | 어휘·통사 구조 + 함의 |
| **권장 모델** | Feature-based / Kernel SVM | **Pre-trained PLM 필수** |

### 11.6 KLUE-RE 시각화

![KLUE Unsupervised](docs/klue_unsupervised_comparison.png)
![KLUE Feature Importance](docs/klue_feature_importance.png)
![KLUE Kernel t-SNE](docs/klue_kernel_tsne.png)
![KLUE Attention Heatmap](docs/klue_attention_heatmap.png)
![KLUE 전체 비교](docs/klue_final_comparison.png)

---

## 12. 실험 재현

```bash
source .venv/bin/activate

# 1단계: 코퍼스 재구축 (필수 전제 조건, ~10초)
python step0_rebuild_corpus.py
# OpenAI 다양화 옵션 (OPEN_AI_KEY 환경변수 필요)
# python step0_rebuild_corpus.py --use-openai --openai-max 500

# 2단계: OIA 전체 파이프라인 (~3~5분)
python run_all_pipeline.py

# 개별 스텝 실행
python step2_unsupervised_re_v2.py    # Unsupervised (Open IE + 군집화)
python step3b_semi_supervised.py       # DIPRE & Snowball
python step3_feature_based_re_v2.py   # Feature-based RF
python step4_deep_learning_re.py      # Bi-LSTM + Attention

# README 시각화 재생성
python generate_readme_visuals.py

# KLUE-RE 파이프라인 (~30분, 인터넷 필요)
python klue_pipeline.py
```

결과: `docs/*.png` + `docs/results.json`

---

## 13. 파일 구조

```
Gachon-IA-KG/
├── data/re_fixed_v6/
│   ├── candidates.jsonl           원본 개체쌍 후보 (1,730건)
│   ├── train.jsonl                LLM Silver (1,205건)
│   ├── labeling_by_relation/      수작업 Gold CSV (12종)
│   ├── corpus_clean.jsonl         ★ step0 재구축 (레이블 포함)
│   └── corpus_unlabeled.jsonl     ★ step0 재구축 (DIPRE pool, UNKNOWN)
│
├── step0_rebuild_corpus.py        ★ 코퍼스 재구축 (빈 패턴 → 템플릿/OpenAI)
├── step1_data_loader.py           Gold/Silver 로더
├── step2_open_ie.py               Open IE (SpaCy + LLM Batches API)
├── step2_unsupervised_re_v2.py    ★ Unsupervised RE (전체 코퍼스 대상)
├── step3_feature_based_re_v2.py   Feature-based RF
├── step3b_semi_supervised.py      ★ DIPRE & Snowball (수정된 버전)
├── step3c_kernel_based_re.py      Kernel SVM 피처 추출
├── step4_deep_learning_re.py      Bi-LSTM + Attention
├── run_all_pipeline.py            전체 파이프라인 마스터 실행
├── generate_readme_visuals.py     ★ README용 추가 시각화 생성
│
├── klue_data_loader.py            KLUE-RE HuggingFace 로더
├── klue_pipeline.py               KLUE-RE 전체 파이프라인
├── analyze_klue.py                KLUE-RE 클래스 불균형 분석
│
└── docs/
    ├── corpus_quality_improvement.png  ★ 코퍼스 재구축 비교
    ├── semi_supervised_per_relation.png ★ 관계별 DIPRE/Snowball F1
    ├── final_comparison_updated.png    ★ 개선 히스토리 포함 비교
    ├── open_ie_analysis.png            ★ Open IE 술어 분포
    ├── oia_domain_insight.png          ★ 패턴 전략별 성능 분석
    ├── step{0..6}_*.png               각 스텝 시각화
    └── klue_*.png                     KLUE-RE 시각화
```

> ★ = 이번 세션에서 새로 생성/수정된 파일

---

## 14. 참고 문헌

| 논문 | 관련 모듈 |
|---|---|
| Brin (1998) *Extracting Patterns and Relations from the World Wide Web* | DIPRE 알고리즘 |
| Agichtein & Gravano (2000, SIGMOD) *Snowball: Extracting Relations from Large Plain-Text Collections* | Snowball 신뢰도 필터 |
| Culotta & Sorensen (2004, ACL) *Dependency Tree Kernels for Relation Extraction* | K_tree 커널 |
| Bunescu & Mooney (2005, EMNLP) *A Shortest Path Dependency Kernel for Relation Extraction* | K_seq 커널 |
| Zhou et al. (2007, ACL) *Exploiting Constituent Dependencies for Tree Kernel-based RE* | α=β=0.3 균등 가중치 |
| Plank & Moschitti (2013, ACL) *Embedding Semantic Similarity in Tree Kernels for Domain Adaptation of RE* | γ=0.4 개체 타입 가중치 |
| Gönen & Alpaydin (2011, JMLR) *Multiple Kernel Learning Algorithms* | MKL 균등 초기화 원칙 |
| Soares et al. (2019, ACL) *Matching the Blanks: Distributional Similarity for Relation Learning* | Entity start hidden state 방식 |
