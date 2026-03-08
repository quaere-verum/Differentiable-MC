from .risk_measure import RiskMeasure
import torch
from enum import IntEnum
from ..hedging.hedge_result import HedgeResult


class VaREstimateType(IntEnum):
    BATCH = 1
    TRAINABLE = 2
    EXTERNAL = 3


class CVaRRisk(RiskMeasure):
    def __init__(
        self,
        alpha: float,
        var_estimate_type: VaREstimateType = VaREstimateType.BATCH,
        softplus_beta: float | None = None,
    ):
        super().__init__()

        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")

        self.alpha = alpha
        self.var_estimate_type = var_estimate_type
        self.softplus_beta = softplus_beta
        self._external_z_initialized = False

        if self.var_estimate_type == VaREstimateType.TRAINABLE:
            self.register_parameter(
                "z",
                torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
            )
        elif self.var_estimate_type == VaREstimateType.EXTERNAL:
            self.register_buffer("z", torch.tensor(0.0, dtype=torch.float32))

    def set_z(self, value: float | torch.Tensor) -> None:
        if self.var_estimate_type != VaREstimateType.EXTERNAL:
            raise RuntimeError("set_z is only valid when var_estimate_type == EXTERNAL")

        z = self.get_buffer("z")
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=z.dtype, device=z.device)
        else:
            value = value.detach().to(dtype=z.dtype, device=z.device)

        if value.numel() != 1:
            raise ValueError("z must be scalar")

        z.copy_(value.reshape(()))
        self._external_z_initialized = True

    def evaluate(self, hedge_result: HedgeResult) -> torch.Tensor:
        losses = -hedge_result.pnl

        if self.var_estimate_type == VaREstimateType.BATCH:
            z = torch.quantile(losses.detach(), q=self.alpha, interpolation="linear")
        elif self.var_estimate_type == VaREstimateType.TRAINABLE:
            z = self.get_parameter("z")
        else:
            if not self._external_z_initialized:
                raise RuntimeError("External z has not been initialized")
            z = self.get_buffer("z")
        if self.softplus_beta:
            return z + torch.nn.functional.softplus(losses - z, self.softplus_beta).mean() / (1.0 - self.alpha)
        else:
            return z + torch.relu(losses - z).mean() / (1.0 - self.alpha)