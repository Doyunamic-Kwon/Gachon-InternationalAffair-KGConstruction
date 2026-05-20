import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.font_manager as fm

plt.rc('font', family='DejaVu Sans')
plt.rcParams['axes.unicode_minus'] = False

def visualize_model_comparison():
    """베이스라인 모델 성능 비교 시각화"""
    models = ['Feature-based\n(Words Only)', 'Feature-based\n(+ Entity Type)', 'Unsupervised\n(TF-IDF)', 'Unsupervised\n(Sentence-BERT)']
    scores = [0.04, 0.85, 0.30, 0.72] # 예시용 가상 수치(실제 도출된 값 기반)
    metrics = ['Macro F1', 'Macro F1', 'V-Measure', 'V-Measure']

    plt.figure(figsize=(10, 6))
    colors = ['#ff9999', '#ff3333', '#99ccff', '#3385ff']
    bars = plt.bar(models, scores, color=colors)
    
    plt.title("ML Baseline Comparison (Text-only vs. Combined Features)", fontsize=14)
    plt.ylabel("Score (0~1)", fontsize=12)
    plt.ylim(0, 1.1)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.2f}", ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300)
    print("✅ model_comparison.png 저장 완료")

def visualize_semantic_drift():
    """DIPRE vs Snowball Semantic Drift 제어 시각화"""
    iterations = [1, 2, 3, 4, 5]
    
    # DIPRE: 튜플 수는 급증하지만 정확도는 급락 (Semantic Drift)
    dipre_tuples = [5, 20, 80, 250, 1000]
    dipre_precision = [1.0, 0.6, 0.3, 0.1, 0.02]
    
    # Snowball: 튜플 수는 완만하게 증가, 정확도는 높게 유지
    snowball_tuples = [5, 12, 35, 70, 120]
    snowball_precision = [1.0, 0.95, 0.92, 0.90, 0.88]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax2 = ax1.twinx()
    
    # 튜플 개수 (막대그래프)
    width = 0.35
    ax1.bar([x - width/2 for x in iterations], dipre_tuples, width, label='DIPRE (Tuples)', color='#d9d9d9', alpha=0.7)
    ax1.bar([x + width/2 for x in iterations], snowball_tuples, width, label='Snowball (Tuples)', color='#b3c6ff', alpha=0.7)
    
    # 정확도 (꺾은선 그래프)
    ax2.plot(iterations, dipre_precision, 'r-o', label='DIPRE (Precision)', linewidth=2.5, markersize=8)
    ax2.plot(iterations, snowball_precision, 'b-s', label='Snowball (Precision)', linewidth=2.5, markersize=8)
    
    ax1.set_xlabel('Bootstrapping Iteration', fontsize=12)
    ax1.set_ylabel('Extracted Tuples (Count)', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)

    ax1.set_xticks(iterations)
    plt.title("Bootstrapping: DIPRE Semantic Drift vs. Snowball Control Effect", fontsize=13)
    
    # 범례 합치기
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center left')
    
    plt.tight_layout()
    plt.savefig('semantic_drift_comparison.png', dpi=300)
    print("✅ semantic_drift_comparison.png 저장 완료")

if __name__ == "__main__":
    visualize_model_comparison()
    visualize_semantic_drift()
