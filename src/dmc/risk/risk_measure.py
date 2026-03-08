import torch
import abc
from ..hedging.hedge_result import HedgeResult


class RiskMeasure(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def evaluate(self, hedge_result: HedgeResult) -> torch.Tensor:
        raise NotImplementedError