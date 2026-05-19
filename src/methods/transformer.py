import argparse

from src.methods._torch_required import require_transformers


def main() -> None:
    parser = argparse.ArgumentParser(description="Transformer RE placeholder using data/re JSONL format.")
    parser.add_argument("--train", default="data/re/train.jsonl")
    parser.add_argument("--dev", default="data/re/dev.jsonl")
    parser.add_argument("--test", default="data/re/test.jsonl")
    parser.add_argument("--model-name", default="klue/bert-base")
    parser.add_argument("--output-dir", default="reports/transformer")
    parser.parse_args()
    require_transformers()
    raise SystemExit("Transformers are available, but fine-tuning is intentionally left as the next implementation step.")


if __name__ == "__main__":
    main()
