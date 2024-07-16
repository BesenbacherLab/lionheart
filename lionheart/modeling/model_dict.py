from typing import Optional


def create_model_dict(
    name: str,
    model_class,
    settings: dict,
    grid: Optional[dict] = None,
    expected_ndim: int = 2,
    requires_channel_dim: bool = False,
    uses_max_iter_arg: bool = True,
):
    return {
        "name": name,
        "model": model_class,
        "settings": settings,
        "grid": grid if grid is not None else {},
        "expected_ndim": expected_ndim,
        "requires_channel_dim": requires_channel_dim,
        "uses_max_iter_arg": uses_max_iter_arg,
    }
