"""
Step 3-C: DIPRE Iteration Analysis
──────────────────────────────────
각 iteration (0~3)에서 관계별 precision/recall/F1 변화 추적
Snowball의 노이즈 제거 효과 정량화
"""

import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
from sklearn.metrics import f1_score, precision_score, recall_score

from step1_data_loader import load_gold_standard
from step3b_semi_supervised import (
    extract_text_pattern, _extract_context_before_e1, _parse_type,
    load_corpus_unlabeled, norm_rel, make_type_pattern
)

def _is_generalizable(pat: str) -> bool:
    """패턴 품질 필터: 숫자/영문만 많은 패턴 제거"""
    words = pat.split()
    numeric = sum(1 for w in words if re.match(r'^\d[\d.\-:/]*$', w))
    if numeric / max(len(words), 1) > 0.5:
        return False
    korean = sum(1 for c in pat if '가' <= c <= '힣')
    if len(words) >= 2 and korean == 0:
        return False
    return True

# ═══════════════════════════════════════════════════════
# ITERATION TRACKING
# ═══════════════════════════════════════════════════════

class DIREIterationTracker:
    def __init__(self):
        self.iterations = defaultdict(lambda: {
            "dipre_f1": [], "snowball_f1": [],
            "dipre_discovered": 0, "snowball_discovered": 0,
            "patterns_per_rel": {},
        })

    def run(self, max_iterations=4, n_seeds=10):
        print("--- 🚀 DIPRE Iteration Analysis (0~3) ---\n")

        gold_df = load_gold_standard()
        corpus = load_corpus_unlabeled()

        gold_df["final_relation"] = gold_df["final_relation"].apply(norm_rel)
        relations = sorted(gold_df["final_relation"].unique())

        # Iteration 0: Initial seeds
        iteration_results = []
        current_seeds = {rel: gold_df[gold_df["final_relation"] == rel].head(n_seeds)
                        for rel in relations}

        for iteration in range(max_iterations):
            print(f"\n▶ Iteration {iteration}")
            iter_data = {
                "iteration": iteration,
                "relations": {}
            }

            for target_relation in relations:
                seeds = current_seeds[target_relation]
                if len(seeds) == 0:
                    continue

                # Pattern extraction
                text_patterns = set()
                type_patterns = set()

                for _, row in seeds.iterrows():
                    p = extract_text_pattern(row.get("marked_text", ""))
                    if p and len(p) >= 2:
                        text_patterns.add(p)

                    ctx = _extract_context_before_e1(row.get("marked_text", ""))
                    if ctx and len(ctx) >= 2:
                        text_patterns.add(ctx)

                    h_type = row.get("head_type") or _parse_type(row.get("head", {}))
                    t_type = row.get("tail_type") or _parse_type(row.get("tail", {}))
                    if h_type and t_type:
                        type_patterns.add(make_type_pattern(h_type, t_type))

                quality_patterns = [p for p in text_patterns if _is_generalizable(p)]

                # DIPRE: Text patterns
                dipre_preds = []
                snowball_preds = []
                actuals = []

                seed_head_types = set()
                seed_tail_types = set()
                for _, row in seeds.iterrows():
                    h = row.get("head_type") or _parse_type(row.get("head", {}))
                    t = row.get("tail_type") or _parse_type(row.get("tail", {}))
                    seed_head_types.add(h)
                    seed_tail_types.add(t)

                for row in corpus:
                    marked = str(row.get("marked_text", ""))
                    actual = norm_rel(row.get("true_relation", ""))

                    matched_text = any(pat in marked for pat in quality_patterns)
                    h_type = _parse_type(row.get("head", {}))
                    t_type = _parse_type(row.get("tail", {}))
                    matched_type = (h_type in seed_head_types and t_type in seed_tail_types)

                    # DIPRE: text only
                    dipre_match = matched_text or matched_type
                    # Snowball: text AND type
                    snowball_match = matched_text and matched_type

                    dipre_preds.append(target_relation if dipre_match else "OTHER")
                    snowball_preds.append(target_relation if snowball_match else "OTHER")
                    actuals.append(target_relation if actual == target_relation else "OTHER")

                # Calculate metrics
                dipre_f1 = f1_score(actuals, dipre_preds, pos_label=target_relation,
                                    average="binary", zero_division=0)
                snowball_f1 = f1_score(actuals, snowball_preds, pos_label=target_relation,
                                       average="binary", zero_division=0)

                dipre_discovered = sum(1 for p in dipre_preds if p == target_relation)
                snowball_discovered = sum(1 for p in snowball_preds if p == target_relation)

                iter_data["relations"][target_relation] = {
                    "dipre_f1": float(dipre_f1),
                    "snowball_f1": float(snowball_f1),
                    "dipre_discovered": int(dipre_discovered),
                    "snowball_discovered": int(snowball_discovered),
                    "text_patterns": len(quality_patterns),
                    "type_patterns": len(type_patterns),
                }

                print(f"  {target_relation:30s}: "
                      f"DIPRE={dipre_f1:.3f} Snowball={snowball_f1:.3f} "
                      f"(txt={len(quality_patterns)} typ={len(type_patterns)})")

                # Update seeds for next iteration (Snowball: high-confidence predictions)
                high_conf = [(row, p) for row, p in zip(corpus, snowball_preds)
                            if p == target_relation and snowball_match][:min(5, n_seeds)]

                if high_conf:
                    # Convert corpus row to gold-like format
                    new_seeds = []
                    for row, _ in high_conf:
                        new_seed = {
                            "marked_text": row.get("marked_text"),
                            "head": row.get("head"),
                            "tail": row.get("tail"),
                            "head_type": _parse_type(row.get("head")),
                            "tail_type": _parse_type(row.get("tail")),
                            "final_relation": target_relation,
                        }
                        new_seeds.append(pd.Series(new_seed))

                    if new_seeds:
                        current_seeds[target_relation] = pd.concat(
                            [current_seeds[target_relation], pd.DataFrame(new_seeds)],
                            ignore_index=True
                        ).drop_duplicates(subset=["marked_text"]).head(n_seeds)

            iteration_results.append(iter_data)

        # Save results
        with open("iteration_results.json", "w") as f:
            json.dump(iteration_results, f, indent=2, ensure_ascii=False)

        print(f"\n✅ iteration_results.json 저장 완료")
        return iteration_results

if __name__ == "__main__":
    tracker = DIREIterationTracker()
    tracker.run(max_iterations=4, n_seeds=10)
