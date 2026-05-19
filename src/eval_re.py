import argparse
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_recall_fscore_support

from src.io_utils import read_jsonl, write_json


def evaluate_rows(rows: list[dict[str, str]]) -> dict:
    y_true = [row["gold"] for row in rows]
    y_pred = [row["pred"] for row in rows]
    labels = sorted(set(y_true) | set(y_pred))
    per_label = {}
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    for label, p, r, f, s in zip(labels, precision, recall, f1, support, strict=False):
        per_label[label] = {"precision": p, "recall": r, "f1": f, "support": int(s)}

    confusion = defaultdict(Counter)
    for gold, pred in zip(y_true, y_pred, strict=False):
        confusion[gold][pred] += 1

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "per_label": per_label,
        "confusion": {gold: dict(preds) for gold, preds in confusion.items()},
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate relation extraction predictions.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", default="reports/metrics.json")
    args = parser.parse_args()

    metrics = evaluate_rows(read_jsonl(args.predictions))
    write_json(args.output, metrics)
    print(metrics["classification_report"])
    print(f"Wrote {Path(args.output)}")


if __name__ == "__main__":
    main()
