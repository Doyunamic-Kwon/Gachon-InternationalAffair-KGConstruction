"""
README 업데이트용 추가 시각화 생성
"""
import json, re, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

os.makedirs("docs", exist_ok=True)
plt.rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

# ─────────────────────────────────────────────────────────────
# 1. 코퍼스 품질 개선 Before / After
# ─────────────────────────────────────────────────────────────
def plot_corpus_quality():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Before (candidates.jsonl 기준)
    before = {'리치 패턴\n(≥10자)': 838, '짧은 패턴\n(2-9자)': 108, '빈 패턴\n(<2자)': 784}
    # After (corpus_clean.jsonl 기준)
    after  = {'리치 패턴\n(≥10자)': 1257, '짧은 패턴\n(2-9자)': 424, '빈 패턴\n(<2자)': 49}

    colors = ['#4CAF50', '#FFC107', '#F44336']

    for ax, data, title in zip(
        [axes[0], axes[1]],
        [before, after],
        ['BEFORE\ncandidates.jsonl (원본)', 'AFTER\ncorpus_clean.jsonl (재구축)']
    ):
        wedges, texts, autotexts = ax.pie(
            list(data.values()), labels=list(data.keys()),
            autopct='%1.1f%%', colors=colors,
            startangle=90, pctdistance=0.75,
            wedgeprops=dict(edgecolor='white', linewidth=2),
            textprops={'fontsize': 10}
        )
        for at in autotexts:
            at.set_fontsize(10)
            at.set_fontweight('bold')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=12)
        total = sum(data.values())
        ax.text(0, -1.35, f"총 {total:,}건", ha='center', fontsize=11, fontweight='bold')

    # Bar chart: 변화량 비교
    categories = ['빈 패턴\n(<2자)', '짧은 패턴\n(2-9자)', '리치 패턴\n(≥10자)']
    b_vals = [784/1730*100, 108/1730*100, 838/1730*100]
    a_vals = [49/1730*100,  424/1730*100, 1257/1730*100]
    x = np.arange(len(categories))
    w = 0.35
    ax = axes[2]
    bars1 = ax.bar(x - w/2, b_vals, w, label='Before', color=['#F44336','#FFC107','#4CAF50'],
                   alpha=0.6, edgecolor='gray')
    bars2 = ax.bar(x + w/2, a_vals, w, label='After',  color=['#F44336','#FFC107','#4CAF50'],
                   alpha=1.0, edgecolor='gray')

    for bar, val in zip(bars1, b_vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f'{val:.1f}%', ha='center', fontsize=9)
    for bar, val in zip(bars2, a_vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylabel('비율 (%)', fontsize=11)
    ax.set_title('패턴 품질 개선 비교\n(Before vs After)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 80)
    ax.grid(axis='y', alpha=0.3)
    sns.despine(ax=ax)

    # 큰 화살표 주석
    fig.text(0.36, 0.5, '→\nstep0\n재구축', ha='center', va='center',
             fontsize=16, fontweight='bold', color='#2196F3')

    plt.suptitle('코퍼스 품질 개선: 빈 패턴 45.3% → 2.8% (−42.5%p)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('docs/corpus_quality_improvement.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ docs/corpus_quality_improvement.png")


# ─────────────────────────────────────────────────────────────
# 2. DIPRE / Snowball per-relation F1 비교
# ─────────────────────────────────────────────────────────────
def plot_dipre_per_relation():
    relations = [
        'HAS_CONTACT_PHONE', 'REQUIRES_DOCUMENT', 'ANNOUNCED_BY',
        'NO_RELATION', 'HAS_DEADLINE', 'REFERENCES_ATTACHMENT',
        'REFERENCES_EXTERNAL_RESOURCE', 'HAS_CONTACT_EMAIL',
        'HAS_FEE', 'MENTIONS_EXAM_LEVEL', 'REQUIRES_QUALIFICATION', 'MENTIONS'
    ]
    dipre_f1    = [0.0862, 0.5340, 0.0998, 0.1888, 0.7273, 0.9333,
                   0.9272, 0.3520, 0.0779, 0.8000, 0.0588, 0.0271]
    snowball_f1 = [0.2326, 0.5969, 0.5641, 0.1623, 0.7273, 0.9333,
                   0.9272, 0.4018, 0.2857, 0.8000, 0.0615, 0.0517]
    type_only = [False, False, False, False, True, True,
                 True, False, False, True, False, False]

    fig, ax = plt.subplots(figsize=(13, 7))
    y = np.arange(len(relations))
    h = 0.35

    # 배경 교차 줄
    for i in range(len(relations)):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color='#f5f5f5', zorder=0)

    bars1 = ax.barh(y + h/2, dipre_f1,    h, label='DIPRE',    color='#ff9999', edgecolor='#cc4444')
    bars2 = ax.barh(y - h/2, snowball_f1, h, label='Snowball', color='#6699cc', edgecolor='#334d80')

    # 값 레이블
    for bar, val in zip(bars1, dipre_f1):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}', va='center', fontsize=9)
    for bar, val in zip(bars2, snowball_f1):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}', va='center', fontsize=9, fontweight='bold')

    # TYPE-ONLY 마크
    for i, (rel, is_type) in enumerate(zip(relations, type_only)):
        if is_type:
            ax.text(0.97, i, '★ TYPE-ONLY', transform=ax.get_yaxis_transform(),
                    ha='right', va='center', fontsize=8,
                    color='#2e7d32', fontweight='bold')

    ax.set_yticks(y)
    ax.set_yticklabels(relations, fontsize=10)
    ax.set_xlabel('Binary F1-Score', fontsize=12)
    ax.set_title(
        'DIPRE vs Snowball — 관계별 Binary F1 (OIA Corpus, 10-seed)\n'
        '★ TYPE-ONLY: 텍스트 패턴 없음 → 개체 타입 매칭으로 fallback',
        fontsize=13, fontweight='bold'
    )
    ax.set_xlim(0, 1.1)
    ax.axvline(x=np.mean(dipre_f1),    color='#cc4444', linestyle='--', alpha=0.6, linewidth=1.5,
               label=f'DIPRE avg={np.mean(dipre_f1):.3f}')
    ax.axvline(x=np.mean(snowball_f1), color='#334d80', linestyle='--', alpha=0.6, linewidth=1.5,
               label=f'Snowball avg={np.mean(snowball_f1):.3f}')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    sns.despine(ax=ax)

    plt.tight_layout()
    plt.savefig('docs/semi_supervised_per_relation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ docs/semi_supervised_per_relation.png")


# ─────────────────────────────────────────────────────────────
# 3. 전체 파이프라인 최종 비교 (업데이트된 수치 반영)
# ─────────────────────────────────────────────────────────────
def plot_final_comparison_updated():
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ── 왼쪽: 패러다임별 최고 성능 ──
    paradigms = [
        'Unsupervised\n(V-Measure)',
        'Semi-supervised\n(Macro F1)',
        'Supervised ML\n(Macro F1)',
        'Deep Learning\n(Macro F1)',
    ]
    model1 = ['Pattern-based\n(TF-IDF)', 'DIPRE\n(10-seed)', 'Random Forest', 'Bi-LSTM+Attn']
    model2 = ['Embedding-based\n(SBERT)', 'Snowball\n(+Type Filter)', 'Kernel SVM', '']
    scores1 = [0.2466, 0.4010, 0.7300, 0.5418]
    scores2 = [0.3534, 0.4787, 0.8627, 0.0]
    colors1 = ['#ffb3b3', '#ff9999', '#90EE90', '#9999ff']
    colors2 = ['#99ccff', '#6699cc', '#2e7d32', 'none']

    x = np.arange(len(paradigms))
    w = 0.38
    ax = axes[0]

    bars1 = ax.bar(x - w/2, scores1, w, color=colors1, edgecolor='gray', linewidth=0.8, label='Method 1')
    bars2 = ax.bar(x + w/2, scores2, w, color=colors2, edgecolor='gray', linewidth=0.8, label='Method 2')

    for bar, s, lbl in zip(bars1, scores1, model1):
        if s > 0:
            ax.text(bar.get_x()+bar.get_width()/2, s+0.012,
                    f'{lbl}\n{s:.4f}', ha='center', fontsize=7.5, fontweight='bold')
    for bar, s, lbl in zip(bars2, scores2, model2):
        if s > 0:
            ax.text(bar.get_x()+bar.get_width()/2, s+0.012,
                    f'{lbl}\n{s:.4f}', ha='center', fontsize=7.5, fontweight='bold')

    # 배경 영역
    bg_colors = ['#fff0f0', '#f0f8ff', '#f0fff0', '#fff8f0']
    for i, bc in enumerate(bg_colors):
        ax.axvspan(i - 0.5, i + 0.5, alpha=0.2, color=bc, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(paradigms, fontsize=10, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('OIA RE Pipeline — 전체 패러다임 성능 비교\n(Updated: DIPRE 0.40 / Snowball 0.48)',
                 fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    sns.despine(ax=ax)

    # ── 오른쪽: DIPRE/Snowball 개선 히스토리 ──
    ax2 = axes[1]
    versions = ['v1\n(원본)', 'v2\n(품질 필터)', 'v3\n(코퍼스\n재구축)']
    dipre_hist    = [0.0,   0.2604, 0.4010]
    snowball_hist = [0.005, 0.3873, 0.4787]

    x2 = np.arange(len(versions))
    w2 = 0.3
    b1 = ax2.bar(x2 - w2/2, dipre_hist,    w2, color='#ff9999', edgecolor='#cc4444',
                 label='DIPRE Macro F1')
    b2 = ax2.bar(x2 + w2/2, snowball_hist, w2, color='#6699cc', edgecolor='#334d80',
                 label='Snowball Macro F1')

    for bar, v in zip(b1, dipre_hist):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                 f'{v:.4f}', ha='center', fontsize=10, fontweight='bold')
    for bar, v in zip(b2, snowball_hist):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                 f'{v:.4f}', ha='center', fontsize=10, fontweight='bold')

    # 개선율 화살표
    for i, (d1, d2) in enumerate(zip(dipre_hist[:-1], dipre_hist[1:])):
        if d1 > 0:
            ax2.annotate('', xy=(i+1-w2/2+w2/2, d2), xytext=(i-w2/2+w2/2, d1),
                         arrowprops=dict(arrowstyle='->', color='#cc4444', lw=1.5))

    ax2.set_xticks(x2)
    ax2.set_xticklabels(versions, fontsize=11)
    ax2.set_ylim(0, 0.65)
    ax2.set_ylabel('Macro F1-Score', fontsize=11)
    ax2.set_title('DIPRE & Snowball 개선 히스토리\n(코퍼스 재구축 전후)',
                  fontsize=11, fontweight='bold')

    # 개선율 텍스트
    ax2.text(2.5, 0.56,
             f'DIPRE:    ≈0 → 0.40  (+∞)\nSnowball: ≈0 → 0.48  (+∞)',
             ha='right', va='top', fontsize=9.5,
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#e8f5e9', alpha=0.8))

    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    sns.despine(ax=ax2)

    plt.suptitle('OIA Relation Extraction — 최종 실험 결과 요약',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('docs/final_comparison_updated.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ docs/final_comparison_updated.png")


# ─────────────────────────────────────────────────────────────
# 4. Open IE 트리플 분포 시각화
# ─────────────────────────────────────────────────────────────
def plot_open_ie_analysis():
    # corpus_clean.jsonl에서 집계
    import json, re
    from collections import Counter
    from pathlib import Path

    path = Path('data/re_fixed_v6/corpus_clean.jsonl')
    if not path.exists():
        print("  corpus_clean.jsonl 없음, Open IE 시각화 스킵")
        return

    rows = [json.loads(l) for l in open(path)]

    def between(marked):
        m = re.search(r'\[/E1\](.*?)\[E2\]', marked, re.DOTALL)
        if not m:
            m = re.search(r'\[/E2\](.*?)\[E1\]', marked, re.DOTALL)
        raw = m.group(1).strip() if m else ''
        cleaned = re.sub(r'[^\w\s가-힣]', ' ', raw)
        return ' '.join(cleaned.split()[:5])

    predicates = Counter()
    type_pairs  = Counter()
    for r in rows:
        pred = between(r.get('marked_text',''))
        if pred:
            predicates[pred] += 1
        head_type = (r.get('head') or {}).get('type','?')
        tail_type = (r.get('tail') or {}).get('type','?')
        type_pairs[f"{head_type} → {tail_type}"] += 1

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 왼쪽: 상위 술어
    top_preds = predicates.most_common(12)
    labels_p = [p[:20] for p, _ in top_preds]
    counts_p = [c for _, c in top_preds]
    colors_p  = sns.color_palette('Blues_d', len(labels_p))[::-1]
    axes[0].barh(range(len(labels_p)), counts_p, color=colors_p, edgecolor='gray')
    axes[0].set_yticks(range(len(labels_p)))
    axes[0].set_yticklabels(labels_p, fontsize=9)
    axes[0].invert_yaxis()
    for i, c in enumerate(counts_p):
        axes[0].text(c + 1, i, str(c), va='center', fontsize=9, fontweight='bold')
    axes[0].set_xlabel('빈도', fontsize=11)
    axes[0].set_title('Open IE — 상위 12 술어 (Predicate) 빈도\n(E1~E2 사이 최대 5단어)',
                      fontsize=11, fontweight='bold')
    axes[0].set_xlim(0, max(counts_p)*1.15)
    axes[0].grid(axis='x', alpha=0.3)
    sns.despine(ax=axes[0])

    # 오른쪽: 개체 타입 쌍 분포 (상위 10)
    top_types = type_pairs.most_common(10)
    labels_t = [t for t, _ in top_types]
    counts_t = [c for _, c in top_types]
    colors_t  = sns.color_palette('Set2', len(labels_t))
    wedges, texts, autotexts = axes[1].pie(
        counts_t, labels=labels_t, autopct='%1.1f%%',
        colors=colors_t, startangle=90, pctdistance=0.78,
        wedgeprops=dict(edgecolor='white', linewidth=1.5),
        textprops={'fontsize': 8}
    )
    for at in autotexts:
        at.set_fontsize(8)
        at.set_fontweight('bold')
    axes[1].set_title('Open IE — 개체 타입 쌍 분포 (상위 10)\n(Head 타입 → Tail 타입)',
                      fontsize=11, fontweight='bold')

    plt.suptitle(f'Open IE 구조 분석 — OIA 코퍼스 {len(rows):,}건',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('docs/open_ie_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ docs/open_ie_analysis.png")


# ─────────────────────────────────────────────────────────────
# 5. OIA 도메인 특성 — 관계별 패턴 타입 매핑 (연구 인사이트)
# ─────────────────────────────────────────────────────────────
def plot_oia_domain_insight():
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 왼쪽: 패턴 전략 (Type-only vs Text) per relation
    relations = [
        'REFERENCES_ATTACHMENT', 'REFERENCES_EXTERNAL_RESOURCE',
        'HAS_DEADLINE', 'MENTIONS_EXAM_LEVEL',
        'REQUIRES_DOCUMENT', 'ANNOUNCED_BY',
        'HAS_CONTACT_EMAIL', 'HAS_FEE',
        'HAS_CONTACT_PHONE', 'REQUIRES_QUALIFICATION',
        'MENTIONS', 'NO_RELATION'
    ]
    strategy = ['TYPE-ONLY','TYPE-ONLY','TYPE-ONLY','TYPE-ONLY',
                'TEXT+TYPE','TEXT+TYPE','TEXT+TYPE','TEXT+TYPE',
                'TEXT(noisy)','TEXT(noisy)','TEXT(noisy)','TEXT(noisy)']
    snowball_f1 = [0.9333, 0.9272, 0.7273, 0.8000,
                   0.5969, 0.5641, 0.4018, 0.2857,
                   0.2326, 0.0615, 0.0517, 0.1623]

    color_map = {'TYPE-ONLY': '#4CAF50', 'TEXT+TYPE': '#2196F3', 'TEXT(noisy)': '#FF5722'}
    bar_colors = [color_map[s] for s in strategy]

    y = np.arange(len(relations))
    ax = axes[0]
    bars = ax.barh(y, snowball_f1, color=bar_colors, edgecolor='white', linewidth=0.5)
    for bar, v in zip(bars, snowball_f1):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{v:.3f}', va='center', fontsize=9, fontweight='bold')
    ax.set_yticks(y)
    ax.set_yticklabels(relations, fontsize=9)
    ax.set_xlabel('Snowball F1-Score', fontsize=11)
    ax.set_title('Snowball F1 by 패턴 전략\n(관계 타입에 따른 성능 차이)',
                 fontsize=11, fontweight='bold')
    ax.set_xlim(0, 1.1)
    ax.grid(axis='x', alpha=0.3)

    # 범례
    patches = [mpatches.Patch(color=v, label=k) for k, v in color_map.items()]
    ax.legend(handles=patches, fontsize=9, loc='lower right')
    sns.despine(ax=ax)

    # 오른쪽: OIA 구조 데이터 vs 자연어 비교
    categories = ['개체 인접\n(패턴 없음)', 'HTML 오염\n패턴', '유효 텍스트\n패턴', '타입 전용\n패턴']
    before_vals = [45.3, 27.1, 13.3, 14.3]  # 원본
    after_vals  = [2.8,  5.2, 59.7, 32.3]   # 재구축 후

    x = np.arange(len(categories))
    w = 0.35
    ax2 = axes[1]
    b1 = ax2.bar(x-w/2, before_vals, w, label='Before (원본)', color='#ffb3b3', edgecolor='gray')
    b2 = ax2.bar(x+w/2, after_vals,  w, label='After (재구축)', color='#99ccff', edgecolor='gray')

    for bar, v in zip(b1, before_vals):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                 f'{v:.1f}%', ha='center', fontsize=9)
    for bar, v in zip(b2, after_vals):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                 f'{v:.1f}%', ha='center', fontsize=9, fontweight='bold')

    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=10)
    ax2.set_ylabel('비율 (%)', fontsize=11)
    ax2.set_title('OIA 구조 데이터 특성:\n패턴 유형별 비율 변화',
                  fontsize=11, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.set_ylim(0, 75)
    ax2.grid(axis='y', alpha=0.3)
    sns.despine(ax=ax2)

    plt.suptitle('OIA 도메인 인사이트: 구조 데이터에서의 관계 추출 전략',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('docs/oia_domain_insight.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ docs/oia_domain_insight.png")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    plot_corpus_quality()
    plot_dipre_per_relation()
    plot_final_comparison_updated()
    plot_open_ie_analysis()
    plot_oia_domain_insight()
    print("\n✅ 모든 시각화 생성 완료 → docs/")
