def require_torch() -> None:
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "This neural RE method requires PyTorch. Install a Python-version-compatible torch build, "
            "then run this script again. The input format is data/re/{train,dev,test}.jsonl."
        ) from exc


def require_transformers() -> None:
    require_torch()
    try:
        import transformers  # noqa: F401
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Transformer RE requires transformers in addition to PyTorch. "
            "Install transformers after PyTorch, then run this script again."
        ) from exc
