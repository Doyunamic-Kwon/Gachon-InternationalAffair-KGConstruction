import argparse

from src.methods._torch_required import require_torch


def main() -> None:
    parser = argparse.ArgumentParser(description="BiLSTM+Attention RE placeholder using data/re JSONL format.")
    parser.add_argument("--train", default="data/re/train.jsonl")
    parser.add_argument("--dev", default="data/re/dev.jsonl")
    parser.add_argument("--test", default="data/re/test.jsonl")
    parser.add_argument("--output-dir", default="reports/attention")
    parser.parse_args()
    require_torch()
    raise SystemExit("PyTorch is available, but the attention architecture has not been trained in this lightweight scaffold yet.")


if __name__ == "__main__":
    main()
