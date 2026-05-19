import argparse
from pathlib import Path

from sklearn.metrics import classification_report, f1_score

from src.io_utils import read_jsonl, write_json, write_jsonl
from src.methods.dipre import normalize_pattern, train_patterns


def bootstrap_patterns(train_rows: list[dict], unlabeled_rows: list[dict], rounds: int, min_count: int) -> dict[str, str]:
    labeled = list(train_rows)
    patterns = train_patterns(labeled, min_count)

    for _ in range(rounds):
        newly_labeled = []
        for row in unlabeled_rows:
            pattern = normalize_pattern(row["marked_text"])
            if pattern in patterns:
                pseudo = dict(row)
                pseudo["relation"] = patterns[pattern]
                newly_labeled.append(pseudo)
        if not newly_labeled:
            break
        labeled.extend(newly_labeled)
        patterns = train_patterns(labeled, min_count)
    return patterns


def predict(row: dict, patterns: dict[str, str]) -> str:
    return patterns.get(normalize_pattern(row["marked_text"]), "NO_RELATION")


def main() -> None:
    parser = argparse.ArgumentParser(description="Snowball-style semi-supervised RE baseline.")
    parser.add_argument("--train", default="data/re/train.jsonl")
    parser.add_argument("--dev", default="data/re/dev.jsonl")
    parser.add_argument("--test", default="data/re/test.jsonl")
    parser.add_argument("--output-dir", default="reports/snowball")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--min-count", type=int, default=1)
    args = parser.parse_args()

    train_rows = read_jsonl(args.train)
    dev_rows = read_jsonl(args.dev)
    test_rows = read_jsonl(args.test)
    patterns = bootstrap_patterns(train_rows, dev_rows, args.rounds, args.min_count)
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
