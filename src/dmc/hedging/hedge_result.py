from torch import Tensor
from dataclasses import dataclass

@dataclass(frozen=True)
class HedgeResult:
    pnl: Tensor