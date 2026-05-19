import argparse
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score

from src.io_utils import read_jsonl, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Unsupervised RE by context clustering and majority-label mapping.")
    parser.add_argument("--train", default="data/re/train.jsonl")
    parser.add_argument("--test", default="data/re/test.jsonl")
    parser.add_argument("--output-dir", default="reports/unsupervised")
    parser.add_argument("--clusters", type=int, default=12)
    args = parser.parse_args()

    train_rows = read_jsonl(args.train)
    test_rows = read_jsonl(args.test)
    all_rows = train_rows + test_rows
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=1)
    matrix = vectorizer.fit_transform([row["marked_text"] for row in all_rows])
    n_clusters = min(args.clusters, len(all_rows))
    clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    clusters = clusterer.fit_predict(matrix)

    cluster_labels = defaultdict(Counter)
    for row, cluster in zip(train_rows, clusters[: len(train_rows)], strict=False):
        cluster_labels[int(cluster)][row["relation"]] += 1
    cluster_to_label = {
        cluster: counts.most_common(1)[0][0]
        for cluster, counts in cluster_labels.items()
    }

    test_clusters = clusters[len(train_rows) :]
    y_test = [row["relation"] for row in test_rows]
    y_pred = [cluster_to_label.get(int(cluster), "NO_RELATION") for cluster in test_clusters]
    predictions = [
        {"id": row["id"], "gold": gold, "pred": pred, "cluster": int(cluster), "text": row["marked_text"]}
        for row, gold, pred, cluster in zip(test_rows, y_test, y_pred, test_clusters, strict=False)
    ]

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "predictions.jsonl", predictions)
    write_json(
        output_dir / "metrics.json",
        {
            "cluster_count": n_clusters,
            "cluster_to_label": cluster_to_label,
            "micro_f1": f1_score(y_test, y_pred, average="micro", zero_division=0),
            "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
            "classification_report": classification_report(y_test, y_pred, zero_division=0),
        },
    )
    print(classification_report(y_test, y_pred, zero_division=0))


if __name__ == "__main__":
    main()
