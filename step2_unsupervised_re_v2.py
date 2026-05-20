"""
Step 2: Unsupervised RE (Open IE / Pattern / Distributional / Embedding)
─────────────────────────────────────────────────────────────────────────
기존 문제: gold_df(257건)만 사용 → 너무 작은 데이터셋, 군집화 의미 없음

수정 사항:
  1. corpus_clean.jsonl(1730건) 사용 → 전체 OIA 코퍼스 대상 군집화
  2. 금(gold) 레이블은 V-Measure 평가용으로만 사용
  3. K = 관계 수 (12개, corpus 관계 분포 기반)
  4. Open IE: SpaCy 패턴 기반 + 엔티티 타입 기반 (structured domain 특성 반영)
"""

import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import v_measure_score

from step1_data_loader import load_gold_standard

# Mac 한글 폰트 설정
plt.rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

# ── 관계 정규화 ─────────────────────────────────────────
NORMALIZE_REL = {
    "requires_document":      "REQUIRES_DOCUMENT",
    "has_deadline":            "HAS_DEADLINE",
    "announced_by":            "ANNOUNCED_BY",
    "mentions":                "MENTIONS",
    "requires_qualification":  "REQUIRES_QUALIFICATION",
}

def norm_rel(r: str) -> str:
    return NORMALIZE_REL.get(str(r).strip(), str(r).strip())


# ── corpus_clean.jsonl 로더 ─────────────────────────────
def load_corpus_clean(path: str = "data/re_fixed_v6/corpus_clean.jsonl") -> list:
    p = Path(path)
    if not p.exists():
        print(f"[WARN] {path} 없음. step0_rebuild_corpus.py를 먼저 실행하세요.")
        return []
    rows = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    print(f"✅ corpus_clean 로드: {len(rows)}건")
    return rows


# ── Open IE: SpaCy 의존 구문 기반 ──────────────────────
def extract_open_ie_tuple_spacy(text: str, nlp) -> str | None:
    if not nlp:
        return None
    clean = re.sub(r'\[/?E[12]\]', '', text)
    doc = nlp(clean)
    subject, verb, obj = "", "", ""
    for token in doc:
        if token.dep_ in ('nsubj', 'csubj') and not subject:
            subject = token.text
        if token.pos_ == 'VERB' and not verb:
            verb = token.text
        if token.dep_ in ('obj', 'iobj', 'pobj') and not obj:
            obj = token.text
    if subject and verb and obj:
        return f"({subject}, {verb}, {obj})"
    return None


# ── Open IE: 엔티티 타입 + 패턴 기반 ──────────────────
def extract_open_ie_structured(row: dict) -> dict | None:
    """
    OIA 구조 데이터 특성에 맞는 Open IE:
    (head_entity, between_text, tail_entity) 트리플 추출
    """
    marked = str(row.get("marked_text", ""))
    head = (row.get("head") or {})
    tail = (row.get("tail") or {})
    head_text = head.get("text", "") if isinstance(head, dict) else str(head)
    tail_text = tail.get("text", "") if isinstance(tail, dict) else str(tail)
    head_type = head.get("type", "?") if isinstance(head, dict) else "?"
    tail_type = tail.get("type", "?") if isinstance(tail, dict) else "?"

    # 개체 사이 텍스트 추출
    m = re.search(r'\[/E1\](.*?)\[E2\]', marked, re.DOTALL)
    if not m:
        m = re.search(r'\[/E2\](.*?)\[E1\]', marked, re.DOTALL)
    between = re.sub(r'[^\w\s가-힣]', ' ', m.group(1)).strip() if m else ""
    between = ' '.join(between.split()[:5])  # 최대 5단어

    if not head_text or not tail_text:
        return None

    return {
        "subject":      head_text,
        "subject_type": head_type,
        "predicate":    between or f"[{head_type}→{tail_type}]",
        "object":       tail_text,
        "object_type":  tail_type,
        "triple":       f"({head_text}, {between or head_type+'→'+tail_type}, {tail_text})",
    }


# ── 패턴 특징 추출 (TF-IDF용) ─────────────────────────
def get_pattern(row: dict) -> str:
    """개체 사이 텍스트 + 개체 타입 조합 특징"""
    marked = str(row.get("marked_text", ""))
    m = re.search(r'\[/E1\](.*?)\[E2\]', marked, re.DOTALL)
    if not m:
        m = re.search(r'\[/E2\](.*?)\[E1\]', marked, re.DOTALL)
    between = m.group(1).strip() if m else ""

    head = row.get("head") or {}
    tail = row.get("tail") or {}
    h_type = (head.get("type", "") if isinstance(head, dict) else "")
    t_type = (tail.get("type", "") if isinstance(tail, dict) else "")

    # 특징 = 패턴 + 타입 접미사
    return f"{between} __H_{h_type}__ __T_{t_type}__"


# ── 메인 ───────────────────────────────────────────────
def run_unsupervised_re():
    print("--- 🚀 Step 2. Unsupervised RE (Open IE / Pattern / Embedding) ---")
    print("    [수정] 전체 corpus_clean.jsonl(1730건) 대상 군집화\n")

    corpus = load_corpus_clean()
    if not corpus:
        return

    gold_df = load_gold_standard()
    gold_df["final_relation"] = gold_df["final_relation"].apply(norm_rel)

    # corpus에서 텍스트/레이블 추출
    texts      = [r.get("marked_text", r.get("sentence", "")) for r in corpus]
    true_rels  = [norm_rel(r.get("relation", "UNKNOWN")) for r in corpus]
    k          = len(set(true_rels))

    print(f"데이터: {len(corpus)}건 | 관계 종류: {k}개")
    rel_dist = Counter(true_rels)
    print("관계 분포:")
    for rel, cnt in rel_dist.most_common():
        print(f"  {rel:35s}: {cnt}")

    # ── 1. Open IE ────────────────────────────────────
    print("\n▶ 1. Open IE (두 가지 방식)")

    # 1a. SpaCy 의존 구문 기반
    try:
        import spacy
        nlp = spacy.load("ko_core_news_sm")
    except Exception:
        nlp = None

    if nlp:
        print("  1a. SpaCy 의존 구문 기반 (ko_core_news_sm):")
        spacy_triples = []
        for r in corpus[:20]:  # 예시 20건
            t = extract_open_ie_tuple_spacy(r.get("sentence", ""), nlp)
            if t:
                spacy_triples.append(t)
        print(f"  추출 예시 ({len(spacy_triples)}건):")
        for t in spacy_triples[:5]:
            print(f"    {t}")
    else:
        print("  1a. SpaCy 모델 없음 (ko_core_news_sm 미설치)")

    # 1b. 구조 기반 Open IE (엔티티 타입 + 패턴)
    print(f"\n  1b. 구조 기반 Open IE (엔티티 타입 + 개체 사이 패턴):")
    structured_triples = []
    for r in corpus:
        t = extract_open_ie_structured(r)
        if t:
            structured_triples.append(t)

    print(f"  전체 {len(structured_triples)}/{len(corpus)}건 트리플 추출 ({len(structured_triples)/len(corpus)*100:.1f}%)")
    print(f"  예시:")
    for t in structured_triples[:5]:
        print(f"    {t['triple']}")

    # 술어(predicate) 분포
    pred_dist = Counter(t["predicate"] for t in structured_triples)
    print(f"\n  상위 10 술어:")
    for pred, cnt in pred_dist.most_common(10):
        print(f"    {cnt:4d}  [{pred}]")

    # ── 2. Pattern-based Clustering (TF-IDF) ─────────
    print("\n▶ 2. Pattern-based Clustering (TF-IDF on between-entity text + type)")
    patterns = [get_pattern(r) for r in corpus]

    vec_pattern = TfidfVectorizer(max_features=1000, analyzer="char_wb", ngram_range=(2, 4))
    X_pattern = vec_pattern.fit_transform(patterns)

    kmeans_pattern = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_pattern = kmeans_pattern.fit_predict(X_pattern)
    v_score_pattern = v_measure_score(true_rels, labels_pattern)
    print(f"Pattern-based V-Measure Score: {v_score_pattern:.4f}")
    print(f"(Gold-only 대비 데이터 {len(corpus)/len(gold_df):.1f}배 증가)")

    # ── 3. Embedding-based Clustering ────────────────
    print("\n▶ 3. Embedding-based Clustering (SBERT Distributional)")
    try:
        from sentence_transformers import SentenceTransformer
        print("  Sentence-BERT 임베딩 중...")
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

        # 클린 텍스트 (마커 제거)
        clean_texts = [re.sub(r'\[/?E[12]\]', '', t).strip() for t in texts]
        X_embed = model.encode(clean_texts, show_progress_bar=True, batch_size=64)

        kmeans_embed = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_embed = kmeans_embed.fit_predict(X_embed)
        v_score_embed = v_measure_score(true_rels, labels_embed)
        print(f"Embedding-based V-Measure Score: {v_score_embed:.4f}")
        embed_available = True
    except Exception as e:
        print(f"  SBERT 실패: {e}")
        v_score_embed  = 0.0
        embed_available = False

    # ── 시각화 ────────────────────────────────────────
    if embed_available:
        methods = [
            "Pattern-based\n(TF-IDF + Type)",
            "Embedding-based\n(SBERT)",
        ]
        scores = [v_score_pattern, v_score_embed]
    else:
        methods = ["Pattern-based\n(TF-IDF + Type)"]
        scores  = [v_score_pattern]

    plt.figure(figsize=(9, 6))
    colors = ["#ffb3b3", "#99ccff", "#b3d9b3"]
    bars = plt.bar(methods, scores, color=colors[:len(scores)], width=0.5)
    plt.title(
        f"Unsupervised RE 방법론별 군집화 성능 비교\n(V-Measure, 전체 코퍼스 {len(corpus)}건)",
        fontsize=13,
    )
    plt.ylabel("V-Measure Score (0~1)", fontsize=12)
    plt.ylim(0, 1.0)

    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2, yval + 0.02,
            f"{yval:.4f}", ha="center", fontsize=13, fontweight="bold",
        )

    plt.tight_layout()
    out_path = "unsupervised_comparison.png"
    plt.savefig(out_path, dpi=300)
    print(f"\n✅ {out_path} 저장 완료")

    print(f"\n요약:")
    print(f"  데이터: Gold 257건 → Full Corpus {len(corpus)}건 ({len(corpus)/257*100:.0f}% 증가)")
    print(f"  Pattern V-Measure: {v_score_pattern:.4f}")
    if embed_available:
        print(f"  SBERT V-Measure:   {v_score_embed:.4f}")
    print(f"  Open IE 트리플:    {len(structured_triples)}건 추출")


if __name__ == "__main__":
    run_unsupervised_re()
