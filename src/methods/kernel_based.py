import argparse
import math
import random
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, f1_score
from sklearn.svm import SVC

from src.io_utils import read_jsonl, write_json, write_jsonl


def tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9가-힣_./:-]+|\[/?E[12]\]", text.lower())


def entity_between_tokens(row: dict) -> list[str]:
    toks = tokens(row["marked_text"])
    try:
        start = toks.index("[e1]")
        end = toks.index("[e2]")
    except ValueError:
        return toks[:80]
    if start > end:
        start, end = end, start
    return toks[start + 1 : end][:80]


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def lcs_ratio(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 0.0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, token_a in enumerate(a, start=1):
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1] / math.sqrt(len(a) * len(b))


def entity_type_kernel(a: dict, b: dict) -> float:
    score = 0.0
    score += 0.5 if a["head"]["type"] == b["head"]["type"] else 0.0
    score += 0.5 if a["tail"]["type"] == b["tail"]["type"] else 0.0
    return score


def composite_kernel(a: dict, b: dict) -> float:
    a_all, b_all = tokens(a["marked_text"])[:120], tokens(b["marked_text"])[:120]
    a_between, b_between = entity_between_tokens(a), entity_between_tokens(b)
    lexical = jaccard(a_all, b_all)
    between = 0.5 * jaccard(a_between, b_between) + 0.5 * lcs_ratio(a_between, b_between)
    entity_type = entity_type_kernel(a, b)
    return 0.35 * lexical + 0.4 * between + 0.25 * entity_type


def kernel_matrix(left: list[dict], right: list[dict]) -> np.ndarray:
    matrix = np.zeros((len(left), len(right)), dtype=float)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            matrix[i, j] = composite_kernel(a, b)
    return matrix


def sample_rows(rows: list[dict], max_rows: int, seed: int) -> list[dict]:
    if max_rows <= 0 or len(rows) <= max_rows:
        return rows
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = {}
    for row in rows:
        by_label.setdefault(row["relation"], []).append(row)

    sampled: list[dict] = []
    per_label = max(1, max_rows // max(1, len(by_label)))
    leftovers: list[dict] = []
    for label_rows in by_label.values():
        rng.shuffle(label_rows)
        sampled.extend(label_rows[:per_label])
        leftovers.extend(label_rows[per_label:])
    rng.shuffle(leftovers)
    sampled.extend(leftovers[: max(0, max_rows - len(sampled))])
    rng.shuffle(sampled)
    return sampled


def main() -> None:
    parser = argparse.ArgumentParser(description="Kernel-based supervised RE with a composite custom kernel.")
    parser.add_argument("--train", default="data/re/train.jsonl")
    parser.add_argument("--test", default="data/re/test.jsonl")
    parser.add_argument("--output-dir", default="reports/kernel_based")
    parser.add_argument("--c", type=float, default=2.0)
    parser.add_argument("--max-train", type=int, default=700)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_rows = sample_rows(read_jsonl(args.train), args.max_train, args.seed)
    test_rows = read_jsonl(args.test)
    y_train = [row["relation"] for row in train_rows]
    y_test = [row["relation"] for row in test_rows]

    train_kernel = kernel_matrix(train_rows, train_rows)
    test_kernel = kernel_matrix(test_rows, train_rows)

    model = SVC(kernel="precomputed", C=args.c, class_weight="balanced")
    model.fit(train_kernel, y_train)
    y_pred = model.predict(test_kernel)

    predictions = []
    for row, pred, gold in zip(test_rows, y_pred, y_test, strict=False):
        predictions.append({"id": row["id"], "gold": gold, "pred": pred, "text": row["marked_text"]})

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "predictions.jsonl", predictions)
    metrics = {
        "micro_f1": f1_score(y_test, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
    }
    write_json(output_dir / "metrics.json", metrics)
    print(metrics["classification_report"])
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
