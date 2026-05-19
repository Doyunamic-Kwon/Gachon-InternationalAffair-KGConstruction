"""
KLUE-RE 데이터 로더
HuggingFace klue/re 데이터셋을 로드하고
기존 파이프라인과 호환되는 형태로 변환합니다.

변환 형식:
  marked_text: "[E1]subject[/E1] context [E2]object[/E2]"
  head_type: subject entity type
  tail_type: object entity type
  final_relation: relation label string
"""

from datasets import load_dataset
import pandas as pd

# KLUE-RE 전체 라벨 30개
KLUE_LABEL_NAMES = [
    'no_relation', 'org:dissolved', 'org:founded', 'org:place_of_headquarters',
    'org:alternate_names', 'org:member_of', 'org:members',
    'org:political/religious_affiliation', 'org:product', 'org:founded_by',
    'org:top_members/employees', 'org:number_of_employees/members',
    'per:date_of_birth', 'per:date_of_death', 'per:place_of_birth',
    'per:place_of_death', 'per:place_of_residence', 'per:origin',
    'per:employee_of', 'per:schools_attended', 'per:alternate_names',
    'per:parents', 'per:children', 'per:siblings', 'per:spouse',
    'per:other_family', 'per:colleagues', 'per:product', 'per:religion',
    'per:title'
]

def build_marked_text(sentence, subj, obj):
    """
    원문 + subject/object 위치 정보를 이용해 marked_text를 생성합니다.
    subject가 앞에 있으면 E1, object가 뒤에 있으면 E2로 마킹합니다.
    """
    s_start = subj['start_idx']
    s_end   = subj['end_idx']
    o_start = obj['start_idx']
    o_end   = obj['end_idx']

    # 두 엔티티의 상대 위치에 따라 마킹 순서 결정
    if s_start <= o_start:
        # subject가 앞
        marked = (
            sentence[:s_start]
            + "[E1]" + sentence[s_start:s_end+1] + "[/E1]"
            + sentence[s_end+1:o_start]
            + "[E2]" + sentence[o_start:o_end+1] + "[/E2]"
            + sentence[o_end+1:]
        )
    else:
        # object가 앞
        marked = (
            sentence[:o_start]
            + "[E2]" + sentence[o_start:o_end+1] + "[/E2]"
            + sentence[o_end+1:s_start]
            + "[E1]" + sentence[s_start:s_end+1] + "[/E1]"
            + sentence[s_end+1:]
        )
    return marked


def load_klue_re(split='train', max_samples=None):
    """
    KLUE-RE 데이터셋을 로드하여 기존 파이프라인 호환 DataFrame으로 반환합니다.

    Returns:
        DataFrame with columns:
            - sentence      : 원문
            - marked_text   : [E1]/[E2] 마킹된 문장
            - head_type     : subject entity type (ORG, PER, LOC, ...)
            - tail_type     : object entity type
            - final_relation: 관계 라벨 문자열
    """
    ds = load_dataset("klue", "re")
    data = ds[split]

    label_names = ds['train'].features['label'].names

    rows = []
    for item in data:
        sentence  = item['sentence']
        subj      = item['subject_entity']
        obj       = item['object_entity']
        label_idx = item['label']
        label_str = label_names[label_idx]

        marked = build_marked_text(sentence, subj, obj)

        rows.append({
            'sentence':       sentence,
            'marked_text':    marked,
            'head_type':      subj['type'],
            'tail_type':      obj['type'],
            'final_relation': label_str,
        })

        if max_samples and len(rows) >= max_samples:
            break

    df = pd.DataFrame(rows)
    print(f"✅ KLUE-RE ({split}) 로드 완료: 총 {len(df)}건, 관계 수 {df['final_relation'].nunique()}개")
    return df


if __name__ == "__main__":
    train_df = load_klue_re('train')
    val_df   = load_klue_re('validation')

    print("\n[샘플 확인]")
    row = train_df.iloc[0]
    print(f"원문       : {row['sentence']}")
    print(f"Marked     : {row['marked_text']}")
    print(f"Head Type  : {row['head_type']}")
    print(f"Tail Type  : {row['tail_type']}")
    print(f"관계 라벨  : {row['final_relation']}")
    print(f"\n관계 분포 (상위 10):\n{train_df['final_relation'].value_counts().head(10)}")
