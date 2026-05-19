import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.metrics import classification_report, f1_score

from src.io_utils import read_jsonl, write_json, write_jsonl


def normalize_pattern(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\[e1\].*?\[/e1\]", "<E1>", text)
    text = re.sub(r"\[e2\].*?\[/e2\]", "<E2>", text)
    text = re.sub(r"\d[\d,./:-]*", "<NUM>", text)
    return re.sub(r"\s+", " ", text).strip()


def train_patterns(rows: list[dict], min_count: int) -> dict[str, str]:
    counts = defaultdict(Counter)
    for row in rows:
        counts[normalize_pattern(row["marked_text"])][row["relation"]] += 1

    patterns = {}
    for pattern, label_counts in counts.items():
        label, count = label_counts.most_common(1)[0]
        if count >= min_count and label != "NO_RELATION":
            patterns[pattern] = label
    return patterns


def predict(row: dict, patterns: dict[str, str]) -> str:
    pattern = normalize_pattern(row["marked_text"])
    if pattern in patterns:
        return patterns[pattern]
    for known_pattern, label in patterns.items():
        if known_pattern in pattern or pattern in known_pattern:
            return label
    return "NO_RELATION"


def main() -> None:
    parser = argparse.ArgumentParser(description="DIPRE-style pattern bootstrapping baseline.")
    parser.add_argument("--train", default="data/re/train.jsonl")
    parser.add_argument("--test", default="data/re/test.jsonl")
    parser.add_argument("--output-dir", default="reports/dipre")
    parser.add_argument("--min-count", type=int, default=1)
    args = parser.parse_args()

    train_rows = read_jsonl(args.train)
    test_rows = read_jsonl(args.test)
    patterns = train_patterns(train_rows, args.min_count)
    y_test = [row["relation"] for row in test_rows]
    y_pred = [predict(row, patterns) for row in test_rows]

    predictions = [
        {"id": row["id"], "gold": gold, "pred": pred, "text": row["marked_text"]}
        for row, gold, pred in zip(test_rows, y_test, y_pred, strict=False)
    ]
    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "predictions.jsonl", predictions)
    write_json(
        output_dir / "metrics.json",
        {
            "pattern_count": len(patterns),
            "micro_f1": f1_score(y_test, y_pred, average="micro", zero_division=0),
            "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
            "classification_report": classification_report(y_test, y_pred, zero_division=0),
        },
    )
    print(f"patterns={len(patterns)}")
    print(classification_report(y_test, y_pred, zero_division=0))


if __name__ == "__main__":
    main()
