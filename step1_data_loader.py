import os
import glob
import pandas as pd
import json

def load_gold_standard(data_dir="data/re_fixed_v6/labeling_by_relation"):
    """
    Load human-annotated Gold Standard data from specific relations.
    """
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return pd.DataFrame()
        
    df_list = []
    for file in csv_files:
        df = pd.read_csv(file)
        # Filter only items where 'gold_relation' is not empty if possible,
        # but usually all data in this folder is treated as gold.
        # Alternatively, suggested_relation might be considered the label if gold_relation is NaN.
        # For this setup, we'll assume gold_relation or suggested_relation holds the ground truth.
        df['final_relation'] = df['gold_relation'].fillna(df['suggested_relation'])
        df_list.append(df)
        
    gold_df = pd.concat(df_list, ignore_index=True)
    
    # We only care about rows that have some valid relation
    gold_df = gold_df.dropna(subset=['final_relation'])
    print(f"✅ Gold Standard 데이터 로드 완료: 총 {len(gold_df)} 문장")
    return gold_df

def load_silver_standard(jsonl_dir="data/re_fixed_v6"):
    """
    Load LLM-generated Silver Standard data (Train set).
    """
    train_path = os.path.join(jsonl_dir, "train.jsonl")
    
    if not os.path.exists(train_path):
        print(f"File not found: {train_path}")
        return pd.DataFrame()
        
    # Read jsonl
    data = []
    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
            
    silver_df = pd.DataFrame(data)
    print(f"✅ Silver Standard (Train) 데이터 로드 완료: 총 {len(silver_df)} 문장")
    return silver_df

if __name__ == "__main__":
    print("--- 🚀 Step 1. 데이터 파이프라인 셋업 ---")
    gold_df = load_gold_standard()
    silver_df = load_silver_standard()
    
    if not gold_df.empty:
        print("\n[Gold Standard 샘플]")
        print(gold_df[['final_relation', 'head_text', 'tail_text']].head(3))
        
    if not silver_df.empty:
        print("\n[Silver Standard 샘플]")
        # jsonl might have different keys, let's print columns to verify
        print(f"Columns: {list(silver_df.columns)}")
        print(silver_df.head(1))
