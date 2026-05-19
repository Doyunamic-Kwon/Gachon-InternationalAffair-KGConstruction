# RE Labeling Guide

`data/re/labeling_sample.csv` 또는 `data/re/labeling_by_relation/*.csv`에서 `gold_relation` 칸만 채웁니다.

헷갈리거나 잘못 뽑힌 후보는 `NO_RELATION`으로 표시합니다.

## Allowed Labels

| Label | When To Use |
|---|---|
| `announced_by` | 공지(`Notice`)를 발표/안내한 부서나 담당자가 tail일 때 |
| `mentions` | 공지가 장학금, 행사, 비자/체류 정보를 주제로 언급할 때 |
| `requires_qualification` | 장학금/행사/비자 등이 특정 지원 자격이나 대상자를 요구할 때 |
| `requires_document` | 장학금/행사/비자 등이 제출 서류를 요구할 때 |
| `has_deadline` | 장학금/행사/비자 등이 마감일, 신청기간, 제출기간을 가질 때 |
| `HAS_FEE` | 프로그램/시험/절차에 비용이 연결될 때 |
| `HAS_DEADLINE` | 일반적인 주체가 날짜/기간을 가질 때. `has_deadline`과 헷갈리면 `has_deadline` 우선 |
| `REQUIRES_DOCUMENT` | 일반적인 주체가 서류를 요구할 때. 업무형 relation이면 `requires_document` 우선 |
| `HAS_CONTACT_EMAIL` | 공지/부서/담당자/페이지에 이메일 연락처가 연결될 때 |
| `HAS_CONTACT_PHONE` | 공지/부서/담당자/페이지에 전화번호가 연결될 때 |
| `REFERENCES_ATTACHMENT` | PDF/HWP 등 첨부파일을 참조할 때 |
| `REFERENCES_EXTERNAL_RESOURCE` | 외부 사이트, 신청 링크, 공식 포털을 참조할 때 |
| `MENTIONS_EXAM_LEVEL` | TOPIK 급수/시험 레벨이 언급될 때 |
| `NO_RELATION` | head와 tail 사이에 의미 있는 관계가 없거나 후보 추출이 이상할 때 |

## Recommended First Batch

처음에는 아래 파일만 먼저 라벨링하면 됩니다.

1. `data/re/labeling_by_relation/has_deadline.csv`
2. `data/re/labeling_by_relation/requires_document.csv`
3. `data/re/labeling_by_relation/requires_qualification.csv`
4. `data/re/labeling_by_relation/HAS_FEE.csv`
5. `data/re/labeling_by_relation/HAS_CONTACT_EMAIL.csv`

각 파일에서 10~20개만 먼저 채워도 됩니다. 첫 배치는 품질 확인용입니다.

## How To Fill

예시:

```csv
gold_relation,suggested_relation,head_text,tail_text
has_deadline,has_deadline,한국어능력 졸업인증 대체시험,2026.05.30
NO_RELATION,requires_document,회원가입 유의사항,재학증명서
HAS_CONTACT_EMAIL,HAS_CONTACT_EMAIL,국제교류처,rachel39@gachon.ac.kr
```

`suggested_relation`이 맞으면 그대로 `gold_relation`에 복사합니다.
