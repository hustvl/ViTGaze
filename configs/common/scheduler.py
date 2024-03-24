from typing import Literal
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from fvcore.common.param_scheduler import MultiStepParamScheduler, CosineParamScheduler


def get_scheduler(typ: Literal["multistep", "cosine"] = "multistep", **kwargs):
    if typ == "multistep":
        return MultiStepParamScheduler(**kwargs)
    elif typ == "cosine":
        return CosineParamScheduler(**kwargs)


lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(get_scheduler)(),
    warmup_length=0,
    warmup_factor=0.001,
)
