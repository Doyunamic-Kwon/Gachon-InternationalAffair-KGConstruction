import argparse
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

from src.io_utils import read_jsonl, write_json, write_jsonl


def load_xy(path: str) -> tuple[list[dict], list[str]]:
    rows = read_jsonl(path)
    return rows, [row["relation"] for row in rows]


def as_text(rows: list[dict]) -> list[str]:
    return [row["marked_text"] for row in rows]


def as_metadata(rows: list[dict]) -> list[str]:
    values = []
    for row in rows:
        values.append(
            " ".join(
                [
                    f"HEAD_TYPE={row['head']['type']}",
                    f"TAIL_TYPE={row['tail']['type']}",
                ]
            )
        )
    return values


def build_model() -> Pipeline:
    return Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "text",
                            Pipeline(
                                [
                                    ("selector", FunctionTransformer(as_text, validate=False)),
                                    ("tfidf", TfidfVectorizer(ngram_range=(1, 3), min_df=1, sublinear_tf=True)),
                                ]
                            ),
                        ),
                        (
                            "metadata",
                            Pipeline(
                                [
                                    ("selector", FunctionTransformer(as_metadata, validate=False)),
                                    ("tfidf", TfidfVectorizer(token_pattern=r"(?u)\b\S+\b")),
                                ]
                            ),
                        ),
                    ]
                ),
            ),
            ("clf", LinearSVC(class_weight="balanced", random_state=42)),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature-based supervised RE baseline.")
    parser.add_argument("--train", default="data/re/train.jsonl")
    parser.add_argument("--test", default="data/re/test.jsonl")
    parser.add_argument("--output-dir", default="reports/feature_based")
    args = parser.parse_args()

    train_rows, y_train = load_xy(args.train)
    test_rows, y_test = load_xy(args.test)
    model = build_model()
    model.fit(train_rows, y_train)
    y_pred = model.predict(test_rows)

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
