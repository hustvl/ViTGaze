import time
import torch
from detectron2.engine import SimpleTrainer
from typing import Iterable, Generator


def cycle(iterable: Iterable) -> Generator:
    while True:
        for item in iterable:
            yield item


class CycleTrainer(SimpleTrainer):
    def __init__(
        self,
        model,
        data_loader,
        optimizer,
        gather_metric_period=1,
        zero_grad_before_forward=False,
        async_write_metrics=False,
    ):
        super().__init__(
            model,
            data_loader,
            optimizer,
            gather_metric_period,
            zero_grad_before_forward,
            async_write_metrics,
        )

    @property
    def _data_loader_iter(self):
        # only create the data loader iterator when it is used
        if self._data_loader_iter_obj is None:
            self._data_loader_iter_obj = cycle(self.data_loader)
        return self._data_loader_iter_obj
