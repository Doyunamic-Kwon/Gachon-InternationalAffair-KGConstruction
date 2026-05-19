"""
KLUE-RE Class Imbalance Analysis
- Class distribution visualization
- Example sentences per relation
- Why Bi-LSTM fails analysis
"""
import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

os.makedirs("docs", exist_ok=True)
plt.rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False


def run_klue_analysis():
    print("KLUE-RE 클래스 불균형 분석 중...")
    from klue_data_loader import load_klue_re

    train_df = load_klue_re('train')
    val_df   = load_klue_re('validation')

    # ── 1. Class Distribution ──────────────────────────────────────
    train_counts = train_df['final_relation'].value_counts()
    val_counts   = val_df['final_relation'].value_counts().reindex(train_counts.index, fill_value=0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # (a) Train distribution — bar chart
    colors_list = ['#d62728' if rel == 'no_relation' else '#4c72b0'
                   for rel in train_counts.index]
    bars = axes[0].barh(train_counts.index, train_counts.values,
                        color=colors_list, edgecolor='white', linewidth=0.5)
    for bar in bars:
        v = bar.get_width()
        axes[0].text(v + 30, bar.get_y() + bar.get_height()/2,
                     str(v), va='center', fontsize=8, fontweight='bold')
    axes[0].axvline(train_counts.mean(), color='orange', linestyle='--',
                    linewidth=1.5, label=f'Mean = {train_counts.mean():.0f}')
    axes[0].set_title("KLUE-RE Train — Class Distribution (30 Relations)",
                      fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Count", fontsize=10)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, train_counts.max() * 1.22)
    axes[0].legend(fontsize=9)
    sns.despine(ax=axes[0])

    # (b) Imbalance ratio
    ratio = train_counts.max() / train_counts
    colors_ratio = ['#d62728' if r > 10 else '#ff7f0e' if r > 3 else '#4c72b0'
                    for r in ratio.values]
    axes[1].barh(ratio.index, ratio.values, color=colors_ratio,
                 edgecolor='white', linewidth=0.5)
    axes[1].axvline(1, color='green', linestyle='-', linewidth=1, alpha=0.5)
    axes[1].axvline(10, color='red', linestyle='--', linewidth=1.5,
                    label='10× imbalance threshold')
    axes[1].set_title("KLUE-RE — Imbalance Ratio\n(max_count / class_count)",
                      fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Imbalance Ratio (higher = more imbalanced)", fontsize=10)
    axes[1].invert_yaxis()
    axes[1].legend(fontsize=9)
    sns.despine(ax=axes[1])

    plt.suptitle("KLUE-RE Class Imbalance Analysis\n"
                 f"Train={len(train_df):,} | Val={len(val_df):,} | 30 Relations",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = "docs/klue_class_distribution.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {path}")

    # ── 2. Micro vs Macro F1 comparison (conceptual) ───────────────
    klue_results = json.load(open('docs/klue_results.json')) if os.path.exists('docs/klue_results.json') else None

    # ── 3. Example analysis ────────────────────────────────────────
    print("\n=== KLUE-RE 예시 분석 ===")
    rare_relations = train_counts[train_counts < 200].index.tolist()
    common_relations = train_counts[train_counts > 1000].index.tolist()

    print(f"\n다수 클래스 (>1000건): {common_relations}")
    print(f"소수 클래스 (<200건): {rare_relations}")

    examples = {}
    for rel in rare_relations[:5]:
        sample = train_df[train_df['final_relation'] == rel].head(2)
        examples[rel] = []
        for _, row in sample.iterrows():
            examples[rel].append({
                'sentence': row['sentence'][:120],
                'marked_text': row['marked_text'][:120],
                'head_type': row['head_type'],
                'tail_type': row['tail_type'],
                'relation': rel,
                'count': int(train_counts[rel])
            })

    # Add no_relation example
    no_rel_sample = train_df[train_df['final_relation'] == 'no_relation'].head(2)
    examples['no_relation'] = []
    for _, row in no_rel_sample.iterrows():
        examples['no_relation'].append({
            'sentence': row['sentence'][:120],
            'head_type': row['head_type'],
            'tail_type': row['tail_type'],
            'relation': 'no_relation',
            'count': int(train_counts['no_relation'])
        })

    with open('docs/klue_examples.json', 'w', encoding='utf-8') as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    print("  ✅ docs/klue_examples.json")

    # Print stats for README
    print("\n=== README용 통계 ===")
    print(f"최다 클래스: {train_counts.index[0]} = {train_counts.iloc[0]}건 "
          f"({train_counts.iloc[0]/len(train_df)*100:.1f}%)")
    print(f"최소 클래스: {train_counts.index[-1]} = {train_counts.iloc[-1]}건 "
          f"({train_counts.iloc[-1]/len(train_df)*100:.2f}%)")
    print(f"클래스간 불균형 비율: {train_counts.max()/train_counts.min():.0f}×")
    print(f"no_relation 비율: {train_counts['no_relation']/len(train_df)*100:.1f}%")
    print(f"<100건 클래스 수: {(train_counts < 100).sum()}개")
    print(f"<200건 클래스 수: {(train_counts < 200).sum()}개")

    return train_df, val_df, train_counts, examples


if __name__ == "__main__":
    train_df, val_df, train_counts, examples = run_klue_analysis()

    print("\n=== 소수 클래스 예시 (README용) ===")
    for rel, exs in list(examples.items())[:3]:
        print(f"\n[ {rel} ] (n={exs[0]['count']}건)")
        for ex in exs[:1]:
            print(f"  문장: {ex['sentence']}")
            if 'marked_text' in ex:
                print(f"  Marked: {ex['marked_text']}")
            print(f"  Entity Types: {ex['head_type']} → {ex['tail_type']}")
