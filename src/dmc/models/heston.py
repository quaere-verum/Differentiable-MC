import torch
from dataclasses import dataclass
from ..simulation.simulation_grid import SimulationGrid
from ..simulation.simulation_result import SimulationResult
from ..simulation.measure import Measure
from .model import StochasticModel

@dataclass(frozen=True)
class HestonParameters:
    s0: float
    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float

    def parameter_dict(self) -> dict:
        return {
            "s0": self.s0,
            "v0": self.v0,
            "kappa": self.kappa,
            "theta": self.theta,
            "xi": self.xi,
            "rho": self.rho,
        }


class HestonModel(StochasticModel):
    def __init__(
        self,
        model_parameters: HestonParameters,
        *,
        measure: Measure = Measure.FORWARD,
        use_logit_params: bool = False,
    ):
        super().__init__()
        self._dt: torch.Tensor | None = None
        self._measure = measure
        self._model_parameters = model_parameters
        self._use_logit_params = use_logit_params
        if self._use_logit_params:
            raise NotImplementedError
        else:
            for key, val in self._model_parameters.parameter_dict().items():
                self.register_parameter(key, torch.nn.Parameter(
                    torch.tensor(val, dtype=torch.float32)
                ))

    def bind(self, sim_grid: SimulationGrid) -> None:
        self._dt = (sim_grid.time_grid[1:] - sim_grid.time_grid[:-1]).to(device=self.get_parameter("rho").device, dtype=torch.float32)
    
    def simulate(self, batch_size: int) -> SimulationResult:
        assert self._dt is not None, ("Call self.bind(sim_grid) before simulating.")
        num_steps = self._dt.numel()
        

        rho = self.get_parameter("rho")
        z1 = torch.randn((batch_size, num_steps), device=rho.device)
        z2 = torch.randn((batch_size, num_steps), device=rho.device)
        z_v = z1
        z_s = rho * z1 + (1.0 - rho.square()).clamp_min(0.0).sqrt() * z2

        kappa = self.get_parameter("kappa")
        theta = self.get_parameter("theta")
        xi = self.get_parameter("xi")

        logS = torch.empty((batch_size, num_steps + 1), dtype=torch.float32, device=rho.device)
        variance = torch.empty((batch_size, num_steps + 1), dtype=torch.float32, device=rho.device)
        
        logS[:, 0] = self.get_parameter("s0").log()
        variance[:, 0] = self.get_parameter("v0")

        for t in range(num_steps):
            dt = self._dt[t]
            v_prev = variance[:, t].clamp_min(0.0)
            vol = (v_prev * dt).sqrt()
            dv = (
                kappa * (theta - v_prev) * dt
                + xi * vol * z_v[:, t]
            )
            variance[:, t + 1] = v_prev + dv
            if self._measure == Measure.FORWARD:
                drift =  -0.5 * v_prev * dt
            elif self._measure == Measure.RISK_NEUTRAL:
                # Need to add risk free rate - add later using TermStructure
                raise NotImplementedError
            dlogS = drift + vol * z_s[:, t]
            logS[:, t + 1] = logS[:, t] + dlogS

        spot = logS.exp()
        return SimulationResult(spot=spot, variance=variance, short_rate=None)