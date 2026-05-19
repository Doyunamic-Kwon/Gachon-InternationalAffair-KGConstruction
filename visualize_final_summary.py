import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Mac 한글 폰트 설정
plt.rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

def create_final_summary_chart():
    categories = ['Unsupervised\n(V-Measure)', 'Semi-supervised\n(Macro F1)', 'Supervised ML\n(Macro F1)', 'Deep Learning\n(Macro F1)']
    
    # 모델명
    models = [
        ['Pattern-based', 'Embedding-based'],
        ['DIPRE', 'Snowball'],
        ['Feature-based (RF)', 'Kernel-based (SVM)'],
        ['Bi-LSTM + Attention', '']
    ]
    
    # 점수
    scores = [
        [0.2325, 0.3695],
        [0.2622, 0.4795],
        [0.7300, 0.8350],
        [0.6526, 0.0]
    ]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    x = np.arange(len(categories))
    width = 0.35
    
    # 첫 번째 바 (왼쪽)
    bar1_scores = [s[0] for s in scores]
    rects1 = ax.bar(x - width/2, bar1_scores, width, label='Model 1', color='#4c72b0')
    
    # 두 번째 바 (오른쪽)
    bar2_scores = [s[1] for s in scores]
    rects2 = ax.bar(x + width/2, bar2_scores, width, label='Model 2', color='#dd8452')
    
    # X축 Y축 설정
    ax.set_ylabel('Performance Score (0 ~ 1)', fontsize=12, fontweight='bold')
    ax.set_title('Relation Extraction 파이프라인 방법론별 최종 성능 비교', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.0)
    
    # 범례 숨기기 (텍스트로 대체)
    ax.legend().set_visible(False)
    
    # 막대 위에 점수와 모델명 텍스트 표시
    def autolabel(rects, model_names, idx):
        for i, rect in enumerate(rects):
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{model_names[i][idx]}\n({height:.2f})',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 5),  # 5 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=10, fontweight='bold')

    autolabel(rects1, models, 0)
    autolabel(rects2, models, 1)
    
    plt.tight_layout()
    plt.savefig('final_pipeline_comparison.png', dpi=300)
    print("✅ final_pipeline_comparison.png 저장 완료")

if __name__ == "__main__":
    create_final_summary_chart()
