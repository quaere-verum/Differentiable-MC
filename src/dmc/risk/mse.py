from .risk_measure import RiskMeasure
from torch import Tensor
from ..hedging.hedge_result import HedgeResult

class MSERisk(RiskMeasure):
    def __init__(self):
        super().__init__()

    def evaluate(self, hedge_result: HedgeResult) -> Tensor:
        return hedge_result.pnl.square().mean()