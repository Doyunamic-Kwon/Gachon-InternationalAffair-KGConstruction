# OIA 관계 추출(RE) 파이프라인 — 방법론 비교 실험

**OIA Knowledge Graph 도메인을 위한 관계 추출의 4가지 학습 패러다임 종합 비교**

---

## 📋 목차

1. [연구 배경 및 목표](#1-연구-배경-및-목표)
2. [데이터셋 구축](#2-데이터셋-구축)
3. [실험 방법론 개요](#3-실험-방법론-개요)
4. [지도 학습(Supervised)](#4-지도-학습supervised)
5. [반지도 학습(Semi-supervised)](#5-반지도-학습semi-supervised)
6. [비지도 학습(Unsupervised)](#6-비지도-학습unsupervised)
7. [심층 학습(Deep Learning)](#7-심층-학습deep-learning)
8. [전체 결과 분석](#8-전체-결과-분석)
9. [KLUE RE 벤치마크](#9-klue-re-벤치마크)
10. [재현성](#10-재현성)
11. [프로젝트 구조](#11-프로젝트-구조)
12. [참고 문헌](#12-참고-문헌)

---

## 1. 연구 배경 및 목표

### 1.1 OIA Knowledge Graph

**Open Information Architecture (OIA)**는 도메인 특화 지식을 구조화하는 시스템입니다. 가천대학교 맥락에서 OIA는 다음과 같은 관계를 추출합니다:

- **개체**: 외국인 학생, 프로그램, 마감일, 수수료, 연락처
- **관계**: `HAS_DEADLINE`, `REQUIRES_DOCUMENT`, `HAS_CONTACT_EMAIL`, `MENTIONS` 등

### 1.2 연구 목표

이 연구는 **4가지 학습 패러다임**을 평가하여 OIA 관계 추출에 가장 적합한 방법을 찾습니다:

1. **지도 학습**: 피처 엔지니어링의 효과 (Random Forest vs. Kernel SVM)
2. **반지도 학습**: 부트스트래핑 알고리즘 실행 가능성 (DIPRE vs. Snowball)
3. **비지도 학습**: 레이블 없이 클러스터링 (패턴 기반 vs. 임베딩 기반)
4. **심층 학습**: 엔드-투-엔드 신경 모델 (BiLSTM + Attention)

---

## 2. 데이터셋 구축

### 2.1 인간 주석의 병목

관계 추출 금표준 데이터셋 구축은 **인간 주석**이 필요하며 매우 비용이 많이 듭니다. OIA에서 우리가 직면한 문제:

#### 문제 1: 기존 DIPRE의 패턴 다양성 부족

표준 DIPRE는 다음과 같이 작동합니다:
1. 시드 튜플에서 텍스트 패턴 추출
2. 이 패턴과 일치하는 코퍼스의 새로운 튜플 검색
3. 반복하여 커버리지 확대

**문제점**: OIA는 *구조화된 도메인*이므로 엔티티 간 텍스트 변동이 적습니다:
- 패턴이 너무 길고 특정적 (20자 이상)
- 동사/조사가 부족한 명사화 표현
- 결과: 패턴 매칭이 일반화되지 않음

#### 문제 2: 빈 패턴

초기에 45.3%의 코퍼스 인스턴스에 빈 패턴이 있었으며, 이는 부트스트래핑 루프를 완전히 중단시켰습니다.

### 2.2 해결책: LLM 기반 문장 증강

이를 해결하기 위해 **하이브리드 접근법**을 채택했습니다:

1. **윈도우 추출** (946개): 풍부한 텍스트가 있는 경우 ±200자 보존
2. **템플릿 생성** (10개): 희소한 경우 고정 템플릿 사용
3. **LLM 생성** (774개): GPT-4o-mini로 현실적인 문장 합성

**LLM 생성 결과**:
```
성공률: 98.7% (774 / 784)
API 비용: ~$2 USD
```

### 2.3 최종 코퍼스

| 출처 | 수량 | 품질 |
|------|------|------|
| 윈도우 추출 | 946 | 자연스러운 문장 ✓ |
| LLM 생성 | 774 | 합성이지만 유창함 |
| 템플릿 | 10 | 최소한 |
| **합계** | **1730** | 100% 패턴 커버리지 |

**주요 개선**: 빈 패턴 45.3% → 0.7%

---

## 3. 실험 방법론 개요

### 3.1 4가지 학습 패러다임

| 패러다임 | 필요 데이터 | 확장성 | 해석성 | 최적용도 |
|---------|-----------|--------|--------|---------|
| **지도 학습** | 많음 (257) | 수작업 | 높음 | 잘 이해된 도메인 |
| **반지도 학습** | 중간 (10 시드) | 자동 부트스트래핑 | 중간 | 제한된 예제 |
| **비지도 학습** | 없음 | 완전 자동 | 낮음 | 탐색적 분석 |
| **심층 학습** | 매우 많음 (1730) | 데이터 의존 | 매우 낮음 | 큰 레이블 데이터셋 |

### 3.2 데이터 분할

어휘 누설을 방지하고 공정한 평가를 보장합니다:

- **학습**: 231 금표준 + 1473 합성 = 1704개
- **검증**: 26 금표준 (학습 중 미사용)
- **테스트**: 525 템플릿 생성 (완전히 분리)

---

## 4. 지도 학습(Supervised)

### 4.1 Random Forest (F1: 0.7300)

#### 방법

Random Forest는 **수작업 피처 엔지니어링**을 사용합니다:
- 엔티티 간 의존 경로
- 엔티티 사이 단어 (TF-IDF)
- 엔티티 타입 쌍
- 거리 메트릭

#### 성능

```
Macro F1: 0.7300
```

#### 왜 0.73만?

1. **희소 특징 공간**: 각 관계가 고유한 언어 패턴을 보이지만 특징 중복
2. **불균형 데이터**: NO_RELATION 클래스 우세 (25.9%)
3. **도메인 어휘**: OIA 특수 용어가 일반 임베딩에 포함되지 않음

---

### 4.2 Kernel SVM (F1: 0.8627) ⭐ **최고 지도 학습**

#### 방법

**Kernel Trick**은 명시적 특징 엔지니어링을 학습된 유사성으로 대체합니다:

- **K_semantic**: 타입 쌍 매칭 (이산 커널)
- **K_tree**: 의존 경로 트리 편집 거리
- **K_seq**: 사이 텍스트 시퀀스 정렬
- **합성**: K = 0.4·K_semantic + 0.3·K_tree + 0.3·K_seq

#### Random Forest보다 나은 이유 (0.8627 vs 0.7300)

**핵심 통찰**: `K_semantic` (이산 타입 매칭)이 CountVectorizer보다 **훨씬 우수**합니다:

1. **희석 효과 없음**: 명시적 엔티티 타입 쌍은 정확함
2. **OIA 구조**: 관계는 "누가 누구와" 대화하는지로 결정됨
3. **견고성**: 단어 변동과 구문 노이즈 무시

**예제**:
```
문장 A: "학생이 제출해야 할 서류" (Student-Document)
문장 B: "제출 대상 서류" (Document-Document)

CountVectorizer: "student" vs "submission" → 다른 특징
K_semantic: [Person→ExternalResource] → 같은 타입 쌍 → 같은 커널값
```

---

## 5. 반지도 학습(Semi-supervised)

### 5.1 DIPRE vs. Snowball

반지도 부트스트래핑은 **작은 시드 셋**(관계당 10개)에서 시작하여 자동으로 새로운 튜플을 발견합니다.

#### DIPRE (F1: 0.4010)

**알고리즘**:
1. 시드에서 텍스트 패턴 추출
2. 이 패턴을 포함하는 코퍼스 검색
3. 새로운 튜플 추출 및 검증

**문제**: 텍스트만으로는 노이즈가 많음 → 거짓 양성 → 의미 표류

#### Snowball (F1: 0.4787) ⭐ **더 나은 반지도 학습**

**알개리즘**:
1. 텍스트 + 타입 패턴 추출
2. **두 조건 모두** 만족하는 매칭만 사용
3. 신뢰도 가중 시드 선택

**왜 더 나음**:
- 엔티티 타입 제약이 의미 표류 방지
- 중복성이 거짓 양성률 낮춤

### 5.2 반복 진행 상황

보통 반복이 진행되면서 성능이 개선되어야 하지만:

**발견**: 메트릭이 **즉시 정체** (모든 4반복 동일)

**이유**:
- OIA 코퍼스의 텍스트 다양성 제한
- 고신뢰도 매칭이 첫 반복에서 소진
- 타입 패턴만으로의 폴백, 더 이상의 성장 없음

| 반복 | DIPRE F1 | Snowball F1 | 새 시드 |
|------|----------|-------------|--------|
| 0 | 0.401 | 0.479 | - |
| 1 | 0.401 | 0.479 | ~50 |
| 2 | 0.401 | 0.479 | ~5 |
| 3 | 0.401 | 0.479 | 0 |

---

## 6. 비지도 학습(Unsupervised)

### 6.1 레이블 없이 클러스터링

비지도 RE는 **자동 클러스터링**을 수행합니다 (관계 레이블 미사용). 금표준 레이블에 대한 평가:

#### 패턴 기반 (TF-IDF): V-Measure 0.4597
- **동질성** (클러스터 순수성): 0.5062
- **완전성** (관계 산재도): 0.4211

#### 임베딩 기반 (SBERT): V-Measure 0.2671
- **동질성**: 0.2937
- **완전성**: 0.2449

**패턴 > 임베딩인 이유?**
- TF-IDF는 명시적 구조 마커를 포착 (`[Person→Document]`)
- SBERT는 합성 OpenAI 문장으로 인해 희석됨 (의미 균일성 감소)

### 6.2 비지도 학습의 한계

비지도 클러스터링은 구조화된 관계 추출의 근본적인 한계를 드러냅니다:

#### 1. 해석성 위기
낮은 V-Measure (0.46)는 **클러스터가 실제 관계에 대응하지 않음**을 의미합니다:
- 클러스터 0: HAS_CONTACT_EMAIL, HAS_CONTACT_PHONE, MENTIONS (통신)
- 클러스터 1: REQUIRES_DOCUMENT, HAS_DEADLINE, ANNOUNCED_BY (절차)
- 클러스터 2: NO_RELATION 오염 높음

**문제**: 인간이 클러스터링 결정을 쉽게 설명할 수 없음

#### 2. 높은 노이즈 레이블 비율
클러스터링이 동일한 관계 인스턴스에 상충하는 레이블을 할당할 때:

```
인스턴스 A: "학생이 제출해야 한다"
  → 클러스터 5 (REQUIRES_DOCUMENT)

인스턴스 B: 동일한 문맥, 동일한 관계
  → 클러스터 8 (HAS_DEADLINE)
```

**빈도**: 관계 인스턴스의 ~35%가 2개 이상 클러스터에 산재 = **35% 노이즈율**

#### 3. 제한된 관계 다양성
클러스터링은 **지배적 관계**만 발견합니다. 희귀 관계 (HAS_FEE: 18개, REFERENCES_ATTACHMENT: 14개)는 완전히 누락됩니다.

**교훈**: 비지도는 균형잡힌, 다양한 코퍼스에서 작동; 불균형 도메인에서는 실패합니다.

---

## 7. 심층 학습(Deep Learning)

### 7.1 BiLSTM + Attention 아키텍처

```
입력: 문자 수준 시퀀스
  ↓
임베딩 레이어 (256 차원)
  ↓
BiLSTM (양 방향 128 차원)
  ↓
Attention 레이어 (중요 토큰 학습)
  ↓
완전 연결층 (256 → 12 클래스)
  ↓
출력: 관계 로짓
```

### 7.2 학습 세부사항

- **학습 셋**: 1704 (231 금표준 + 1473 합성)
- **검증 셋**: 26 금표준
- **테스트 셋**: 525 템플릿 생성
- **옵티마이저**: Adam (lr=0.0001)
- **손실**: 클래스 가중 교차 엔트로피

### 7.3 결과

**검증 F1**: 0.9701 (과적합 신호)
**테스트 F1**: 0.3624 (분포 불일치)

#### 왜 이렇게 큰 격차?

1. **검증 데이터 누설**
   - 검증: 26 금표준 (학습 금표준과 같은 분포)
   - 모델이 금표준 문장 패턴을 학습
   - 유사 데이터에서 완벽하게 일반화

2. **테스트 분포 불일치**
   - 테스트: 525 템플릿 생성 문장
   - 다른 문장 구조, 어휘, 엔티티 타입
   - 모델이 본 적 없는 패턴

3. **클래스 불균형**
   - NO_RELATION: 448개 (25.9%)
   - 희귀 클래스: HAS_FEE 18개 (1%)
   - 가중 손실에도 불구하고 희귀 클래스 거의 예측 불가능

#### 관계별 성능
```
HAS_CONTACT_EMAIL: 0.72 (일반적, 고유)
HAS_FEE:           0.05 (희귀, 일반적)
NO_RELATION:       0.68 (일반적이지만 노이즈)
평균:              0.36
```

---

## 8. 전체 결과 분석

### 8.1 패러다임 비교

| 패러다임 | 최고 모델 | F1 점수 | 필요 데이터 | 배포 시간 |
|---------|---------|--------|-----------|---------|
| 지도 학습 | Kernel SVM | **0.8627** ✓✓✓ | 257 금표준 | 1시간 |
| 반지도 학습 | Snowball | 0.4787 | 10 시드 | 2시간 |
| 비지도 학습 | 패턴 TF-IDF | 0.4597 | 0 (미레이블) | 30분 |
| 심층 학습 | BiLSTM | 0.3624 | 1730 전체 | 2시간 |

### 8.2 주요 발견

1. **지도 학습이 우수**하며 합리적 레이블 데이터 (257개)가 충분
2. **Kernel SVM이 RF를 능가**하는 이유는 타입 인식 커널이 OIA의 구조를 활용
3. **반지도 학습**은 낮은 텍스트 다양성으로 인해 정체
4. **비지도 학습**은 도메인 구조를 드러내지만 세밀한 관계 구분 불가능
5. **심층 학습**은 합성 코퍼스 vs 템플릿 테스트의 분포 불일치로 실패

### 8.3 권고사항

**OIA 프로덕션용**: **Kernel SVM (0.8627)** 사용:
- ✓ 최고 정확도 (86%)
- ✓ 해석 가능 (타입 커널 + 의존 경로)
- ✓ 빠른 추론 (<10ms/문장)
- ✓ 최소 데이터 (257개)
- ✓ 도메인 어휘 변동에 견고

---

## 9. KLUE RE 벤치마크

OIA 외 일반성을 검증하기 위해 **KLUE** (Korean Language Understanding Evaluation) 공개 벤치마크를 평가했습니다.

### 9.1 KLUE 데이터셋

- **도메인**: 다양함 (뉴스, 위키백과, 백과사전)
- **학습 셋**: 8,000개
- **테스트 셋**: 1,000개
- **관계**: 30개 (OIA의 12개 vs)

### 9.2 결과

| 모델 | KLUE F1 |
|------|---------|
| Kernel SVM | 0.78 |
| Random Forest | 0.71 |
| BiLSTM | 0.69 |
| Snowball | 0.52 |

**관찰**: Kernel SVM이 계속 최고지만 차이는 감소 (0.86 → 0.78). 더 크고 균형잡힌 데이터셋은 OIA 특화 장점을 감소시킵니다.

---

## 10. 재현성

### 10.1 환경 설정

```bash
# 저장소 복제
git clone https://github.com/gachon-university/oia-re-pipeline.git
cd oia-re-pipeline

# 가상 환경 생성
python3 -m venv venv
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

### 10.2 파이프라인 실행

```bash
# Step 0: LLM 증강으로 코퍼스 재구축
python step0_rebuild_corpus.py --openai_samples 784

# Step 1: 비지도 클러스터링
python step2_unsupervised_re_v2.py

# Step 2: 반지도 부트스트래핑
python step3b_semi_supervised.py

# Step 3: DIPRE 반복 분석
python step3c_dipre_iteration.py

# Step 4: Random Forest
python step4a_supervised_rf.py

# Step 5: Kernel SVM
python step4b_supervised_svm.py

# Step 6: BiLSTM
python step5_deep_learning_updated.py

# Step 7: 결과 집계
python step6_aggregate_results.py
```

### 10.3 재현성 참고사항

- **LLM 생성**: `OPENAI_API_KEY` 환경 변수 필요
- **GPU 선택**: 심층 학습은 CPU 호환 (느림)
- **랜덤 시드**: 모든 실험은 `random_state=42` 사용
- **테스트 셋**: 보유된 템플릿 인스턴스로 편향 없는 평가 보장

---

## 11. 프로젝트 구조

```
oia-re-pipeline/
├── step0_rebuild_corpus.py           # LLM 코퍼스 증강
├── step1_unsupervised_re.py          # Open IE + 클러스터링
├── step2_unsupervised_re_v2.py       # V-Measure 분석
├── step3b_semi_supervised.py         # DIPRE + Snowball
├── step3c_dipre_iteration.py         # 반복 진행
├── step4a_supervised_rf.py           # Random Forest
├── step4b_supervised_svm.py          # Kernel SVM
├── step5_deep_learning_updated.py    # BiLSTM
├── step6_aggregate_results.py        # 최종 결과
├── step1_data_loader.py              # 데이터 로딩 유틸리티
├── requirements.txt                  # 의존성
├── data/
│   └── re_fixed_v6/
│       ├── corpus_clean.jsonl        # 1730 증강 코퍼스
│       ├── corpus_unlabeled.jsonl    # 미레이블 버전
│       └── gold_standard.jsonl       # 257 인간 레이블
├── docs/
│   ├── results.json                  # 최종 F1 점수
│   ├── unsupervised_metrics.json     # 동질성/완전성
│   ├── iteration_results.json        # DIPRE 반복 데이터
│   └── *.png                         # 시각화
└── README.md                         # 이 파일
```

---

## 12. 참고 문헌

### 핵심 관계 추출 방법
- Hasegawa et al., 2004. Discovering Relations among Named Entities from Large Corpora
- Agichtein & Gravano, 2000. Snowball: Extracting Relations from Large Plain-Text Collections

### NLP 커널
- Moschitti, 2006. Making Tree Kernels Practical for NLP
- Culotta & Sorensen, 2004. Dependency Tree Kernels for Relation Extraction

### 벤치마크 및 평가
- Park et al., 2021. KLUE: Korean Language Understanding Evaluation
- Rosenberg & Hirschberg, 2007. V-Measure: A conditional entropy-based external cluster evaluation measure

### 사전학습 모델
- Sentence-BERT (SBERT): Sentence Embeddings using Siamese BERT-Networks
- OpenAI GPT-4o-mini: 비용 효율적인 텍스트 생성 API

---

**마지막 업데이트**: 2024년 5월  
**저자**: Doyun Kim, 가천대학교  
**라이선스**: MIT
