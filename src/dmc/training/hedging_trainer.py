from dmc.models.model import StochasticModel
from dmc.simulation.simulation_grid import SimulationGrid
from dmc.risk.risk_measure import RiskMeasure
from dmc.products.product import Product
from dmc.control.controller import Controller
from dmc.feature_extraction.feature_extractor import FeatureExtractor
from dmc.hedging.hedging import HedgingEngine, ControlIntervals
import torch
import time
from dataclasses import dataclass


@dataclass
class TrainingSummary:
    train_losses: list[float]
    val_losses: list[float]
    learning_rates: list[float]
    best_val_loss: float
    best_iteration: int
    stopped_early: bool
    stop_reason: str

def _get_lr(optim: torch.optim.Optimizer) -> float:
    return float(optim.param_groups[0]["lr"])


class HedgingTrainer:
    def __init__(
        self,
        model: StochasticModel,
        product: Product,
        feature_extractor: FeatureExtractor,
        controller: Controller,
        risk: RiskMeasure,
        initial_cash: float,
        optim: torch.optim.Optimizer,
        transaction_cost_rate: float = 1e-3,
        use_smooth_payoff: bool = True,
    ):
        self.model = model
        self.product = product
        self.feature_extractor = feature_extractor
        self.controller = controller
        self.risk = risk
        self.initial_cash = initial_cash
        self.optim = optim
        self.transaction_cost_rate = transaction_cost_rate
        self.use_smooth_payoff = use_smooth_payoff

        self.control_grid: SimulationGrid | None = None
        self._master_grid: SimulationGrid | None = None
        self.hedging_engine: HedgingEngine | None = None

    def set_control_grid(self, grid: SimulationGrid) -> None:
        if grid.time_grid[-1].item() > self.product.final_maturity():
            raise ValueError("Control grid cannot extend beyond product maturity")

        self.control_grid = grid
        self._master_grid = self.control_grid.merge(self.product.get_timegrid())

        self.hedging_engine = HedgingEngine(
            product=self.product,
            controller=self.controller,
            feature_extractor=self.feature_extractor,
            use_smooth_payoff=self.use_smooth_payoff,
            initial_cash=self.initial_cash,
            transaction_cost_rate=self.transaction_cost_rate,
            control_intervals=ControlIntervals(
                start_times=self.control_grid.time_grid[:-1],
                end_times=self.control_grid.time_grid[1:],
            ),
        )

        self.model.bind(self._master_grid)
        self.product.bind(self._master_grid)
        self.hedging_engine.bind(self._master_grid)

    def train(
        self,
        n_iters: int,
        batch_size: int,
        *,
        val_batch_size: int = 2**15,
        validate_every: int = 5,
        early_stopping_patience: int = 15,
        early_stopping_min_delta: float = 1e-2,
        early_stopping_warmup: int = 10,
        lr_decay_factor: float = 0.5,
        lr_patience: int = 5,
        min_learning_rate: float = 1e-3,
        common_random_numbers: bool = True,
    ) -> TrainingSummary:
        if self.hedging_engine is None:
            raise RuntimeError("Call set_control_grid(grid) before train()")

        train_losses: list[float] = []
        val_losses: list[float] = []
        learning_rates: list[float] = []

        best_val_loss = float("inf")
        best_iteration = -1
        best_state_dict_controller: dict[str, object] | None = None
        best_state_dict_fe: dict[str, object] | None = None
        no_improvement_count = 0
        stopped_early = False
        stop_reason = "reached_max_iterations"

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optim,
            mode="min",
            factor=lr_decay_factor,
            patience=lr_patience,
            threshold=early_stopping_min_delta,
            threshold_mode="abs",
            min_lr=min_learning_rate,
        )

        # Fixed validation batch for stable model selection / stopping
        with torch.no_grad():
            val_simulated = self.model.simulate(batch_size=val_batch_size)
            if common_random_numbers:
                simulated = self.model.simulate(batch_size=batch_size)
        start = time.perf_counter()

        for iter_idx in range(n_iters):
            self.optim.zero_grad(set_to_none=True)

            if not common_random_numbers:
                with torch.no_grad():
                    simulated = self.model.simulate(batch_size=batch_size)

            hedge_result = self.hedging_engine.run(simulated)
            loss = self.risk.evaluate(hedge_result)
            loss.backward()
            self.optim.step()

            train_loss_value = float(loss.item())
            train_losses.append(train_loss_value)
            learning_rates.append(_get_lr(self.optim))

            msg = f"iter={iter_idx}, train_loss={train_loss_value:.5f}, lr={_get_lr(self.optim):.6g}"

            if iter_idx % validate_every == 0:
                with torch.no_grad():
                    val_hedge_result = self.hedging_engine.run(val_simulated)
                    val_loss = self.risk.evaluate(val_hedge_result)

                val_loss_value = float(val_loss.item())
                val_losses.append(val_loss_value)

                scheduler.step(val_loss_value)

                improved = val_loss_value < (best_val_loss - early_stopping_min_delta)
                if improved:
                    best_val_loss = val_loss_value
                    best_iteration = iter_idx
                    no_improvement_count = 0
                    best_state_dict_controller = {
                        k: v.detach().cpu().clone()
                        for k, v in self.controller.state_dict().items()
                    }
                    best_state_dict_fe = {
                        k: v.detach().cpu().clone()
                        for k, v in self.feature_extractor.state_dict().items()
                    }
                else:
                    if iter_idx >= early_stopping_warmup:
                        no_improvement_count += 1

                msg += f", val_loss={val_loss_value:.5f}, best_val={best_val_loss:.5f}"

                if (
                    iter_idx >= early_stopping_warmup
                    and no_improvement_count >= early_stopping_patience
                ):
                    stopped_early = True
                    stop_reason = "validation_plateau"
                    print(msg)
                    print(f"Early stopping at iter={iter_idx}")
                    break

                if _get_lr(self.optim) <= min_learning_rate + 1e-12:
                    if iter_idx >= early_stopping_warmup and no_improvement_count > 0:
                        stopped_early = True
                        stop_reason = "min_learning_rate_reached"
                        print(msg)
                        print(f"Stopping at iter={iter_idx} because min learning rate was reached")
                        break

            print(msg)

        if best_state_dict_controller is not None:
            self.controller.load_state_dict(best_state_dict_controller)
            self.feature_extractor.load_state_dict(best_state_dict_fe)

        elapsed = time.perf_counter() - start
        print(f"time={elapsed:.4f}s")

        return TrainingSummary(
            train_losses=train_losses,
            val_losses=val_losses,
            learning_rates=learning_rates,
            best_val_loss=best_val_loss,
            best_iteration=best_iteration,
            stopped_early=stopped_early,
            stop_reason=stop_reason,
        )
