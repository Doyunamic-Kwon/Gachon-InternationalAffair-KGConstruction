import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from step3b_semi_supervised import run_dipre_and_snowball

# Mac 한글 폰트 설정
plt.rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False


def create_final_summary_chart(
    # Unsupervised: step2_unsupervised_re_v2.py 실행 결과
    unsup_pattern=0.2325,
    unsup_embed=0.3695,
    # Supervised: step3_feature_based_re_v2.py / visualize_kernel_ml.py 실행 결과
    sup_rf=0.7300,
    sup_kernel=0.8627,
    # Deep Learning: 10 epochs, Macro F1 실측값
    dl_bilstm=0.5418,
    # Semi-supervised: 10-seed + HTML 정제 후 실측값 (2026-05-19)
    semi_dipre=0.1215,
    semi_snowball=0.3010,
):
    if semi_dipre is None or semi_snowball is None:
        print("Semi-supervised F1 실측 계산 중 (step3b)...")
        semi_dipre, semi_snowball = run_dipre_and_snowball()
    else:
        print(f"Semi-supervised 실측값 사용: DIPRE={semi_dipre:.4f}, Snowball={semi_snowball:.4f}")

    categories = [
        'Unsupervised\n(V-Measure)',
        'Semi-supervised\n(Macro F1)',
        'Supervised ML\n(Macro F1)',
        'Deep Learning\n(Macro F1)',
    ]
    models = [
        ['Pattern-based', 'Embedding-based'],
        ['DIPRE', 'Snowball'],
        ['Feature-based (RF)', 'Kernel-based (SVM)'],
        ['Bi-LSTM + Attention', ''],
    ]
    scores = [
        [unsup_pattern, unsup_embed],
        [semi_dipre,    semi_snowball],
        [sup_rf,        sup_kernel],
        [dl_bilstm,     0.0],
    ]

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.set_style("whitegrid")

    x     = np.arange(len(categories))
    width = 0.35

    bar1_scores = [s[0] for s in scores]
    bar2_scores = [s[1] for s in scores]
    rects1 = ax.bar(x - width / 2, bar1_scores, width, color='#4c72b0')
    rects2 = ax.bar(x + width / 2, bar2_scores, width, color='#dd8452')

    ax.set_ylabel('Performance Score (0 ~ 1)', fontsize=12, fontweight='bold')
    ax.set_title('Relation Extraction 파이프라인 방법론별 최종 성능 비교',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.legend().set_visible(False)

    def autolabel(rects, model_names, idx):
        for i, rect in enumerate(rects):
            height = rect.get_height()
            if height > 0:
                ax.annotate(
                    f'{model_names[i][idx]}\n({height:.2f})',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords='offset points',
                    ha='center', va='bottom', fontsize=10, fontweight='bold',
                )

    autolabel(rects1, models, 0)
    autolabel(rects2, models, 1)

    plt.tight_layout()
    plt.savefig('final_pipeline_comparison.png', dpi=300)
    print("✅ final_pipeline_comparison.png 저장 완료")
    print(f"\n[사용된 Semi-supervised 실측값]")
    print(f"  DIPRE    Macro F1: {semi_dipre:.4f}")
    print(f"  Snowball Macro F1: {semi_snowball:.4f}")


if __name__ == "__main__":
    create_final_summary_chart()
