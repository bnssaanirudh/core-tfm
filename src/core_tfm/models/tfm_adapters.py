from __future__ import annotations

from .sklearn_like import SklearnLikeAdapter


def tabpfn3_adapter(**kwargs) -> SklearnLikeAdapter:
    try:
        from tabpfn import TabPFNClassifier
    except ImportError as exc:
        raise ImportError("Install the optional dependency with `pip install tabpfn`.") from exc
    return SklearnLikeAdapter(lambda: TabPFNClassifier(**kwargs))


def tabiclv2_adapter(**kwargs) -> SklearnLikeAdapter:
    try:
        from tabicl import TabICLClassifier
    except ImportError as exc:
        raise ImportError("Install the optional dependency with `pip install tabicl`.") from exc
    return SklearnLikeAdapter(lambda: TabICLClassifier(**kwargs))


def tabfm_adapter(*, backend: str = "pytorch", **kwargs) -> SklearnLikeAdapter:
    try:
        from tabfm import TabFMClassifier
        if backend == "pytorch":
            from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0
        elif backend == "jax":
            from tabfm import tabfm_v1_0_0_jax as tabfm_v1_0_0
        else:
            raise ValueError("backend must be 'pytorch' or 'jax'.")
    except ImportError as exc:
        raise ImportError("Install TabFM from its official google-research/tabfm repository with a backend extra.") from exc
    model = tabfm_v1_0_0.load()
    return SklearnLikeAdapter(lambda: TabFMClassifier(model=model, **kwargs))
