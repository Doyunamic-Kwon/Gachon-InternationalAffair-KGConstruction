import argparse
from pathlib import Path

from src.io_utils import read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect RE method metrics into one comparison report.")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--output", default="reports/re_comparison.md")
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    rows = []
    for metrics_path in sorted(reports_dir.glob("*/metrics.json")):
        method = metrics_path.parent.name
        metrics = read_json(metrics_path)
        rows.append(
            {
                "method": method,
                "micro_f1": metrics.get("micro_f1", 0.0),
                "macro_f1": metrics.get("macro_f1", 0.0),
            }
        )

    lines = [
        "# RE Method Comparison",
        "",
        "| method | micro_f1 | macro_f1 |",
        "|---|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: item["macro_f1"], reverse=True):
        lines.append(f"| {row['method']} | {row['micro_f1']:.4f} | {row['macro_f1']:.4f} |")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(reports_dir / "metrics_index.json", rows)
    print(output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
