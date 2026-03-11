from dmc.models.heston import HestonModel, HestonParameters
from dmc.simulation.simulation_grid import SimulationGrid
from dmc.risk.mse import MSERisk
from dmc.risk.cvar import CVaRRisk, VaREstimateType
from dmc.products.barrier_option import DownAndOutBarrierOption
from dmc.control.mlp_controller import MlpController
from dmc.feature_extraction.barrier_feature_extractor import BarrierOptionFeatureExtractor, VarianceFeatureType
from dmc.training.hedging_trainer import HedgingTrainer, TrainingSummary
from dmc.hedging.hedging import ControlIntervals
from itertools import chain
import torch
from dmc.products.product import Product
from dmc.control.controller import Controller
from dmc.feature_extraction.feature_extractor import FeatureExtractor
from dmc.simulation.simulation_grid import SimulationGrid
from dmc.simulation.simulation_result import SimulationResult
from dmc.hedging.hedging import ControlIntervals, ControlIndices
from dmc.simulation.simulation_state import SimulationState
import argparse
import pprint
import hashlib
import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any
from enum import Enum

@dataclass
class ExperimentConfig:
    log_dir: str = "logs"
    seed: int = 0
    device: str = "cuda"

    learning_rate: float = 0.05
    min_learning_rate: float = 1e-5
    cvar_alpha: float = 0.95
    cvar_softplus_beta: float | None = None
    risk_name: str = "cvar"

    barrier: float = 80.0
    strike: float = 100.0
    maturity: float = 1.0
    transaction_cost_rate: float = 1e-3

    observation_freq: int = 252
    hedging_freq: int = 252

    hidden_sizes: tuple[int, ...] = (16, 16)
    variance_feature_type: VarianceFeatureType = VarianceFeatureType.LEARNED_FILTER

    n_eval_paths: int = 2**19
    n_train_iterations: int = 1000
    batch_size: int = 2**16

    s0: float = 100.0
    v0: float = 0.04
    kappa: float = 2.0
    theta: float = 0.04
    xi: float = 1.0
    rho: float = -0.7

def parse_args() -> ExperimentConfig:

    parser = argparse.ArgumentParser(description="Deep Hedging Experiment")

    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--cvar-alpha", type=float, default=0.95)
    parser.add_argument("--cvar-softplus-beta", type=float, default=None)
    parser.add_argument("--risk-name", type=str, default="cvar")

    parser.add_argument("--barrier", type=float, default=80.0)
    parser.add_argument("--strike", type=float, default=100.0)
    parser.add_argument("--maturity", type=float, default=1.0)

    parser.add_argument("--transaction-cost-rate", type=float, default=1e-3)

    parser.add_argument("--observation-freq", type=int, default=252)
    parser.add_argument("--hedging-freq", type=int, default=252)

    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[])
    parser.add_argument("--variance-feature-type", type=str, default="learned")

    parser.add_argument("--n-eval-paths", type=int, default=2**19)
    parser.add_argument("--n-train-iterations", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2**15)

    parser.add_argument("--s0", type=float, default=100.0)
    parser.add_argument("--v0", type=float, default=0.04)
    parser.add_argument("--kappa", type=float, default=2.0)
    parser.add_argument("--theta", type=float, default=0.04)
    parser.add_argument("--xi", type=float, default=1.0)
    parser.add_argument("--rho", type=float, default=-0.7)

    args = parser.parse_args()

    vft_name =  getattr(args, "variance_feature_type").lower()
    if vft_name == "learned":
        variance_feature_type = VarianceFeatureType.LEARNED_FILTER
    elif vft_name == "markov":
        variance_feature_type = VarianceFeatureType.MARKOV_STATE
    elif vft_name == "none":
        variance_feature_type = VarianceFeatureType.NONE
    elif vft_name == "gated":
        variance_feature_type = VarianceFeatureType.LEARNED_GATED_FILTER
    else:
        raise ValueError(f"Unknown variance_feature_type: {vft_name}")
    return ExperimentConfig(
        log_dir=args.log_dir,
        seed=args.seed,
        device=args.device,
        learning_rate=args.learning_rate,
        cvar_alpha=args.cvar_alpha,
        cvar_softplus_beta=args.cvar_softplus_beta,
        risk_name=args.risk_name,
        barrier=args.barrier,
        strike=args.strike,
        maturity=args.maturity,
        transaction_cost_rate=args.transaction_cost_rate,
        observation_freq=args.observation_freq,
        hedging_freq=args.hedging_freq,
        variance_feature_type=variance_feature_type,
        hidden_sizes=tuple(args.hidden_sizes),
        n_eval_paths=args.n_eval_paths,
        n_train_iterations=args.n_train_iterations,
        batch_size=args.batch_size,
        s0=args.s0,
        v0=args.v0,
        kappa=args.kappa,
        theta=args.theta,
        xi=args.xi,
        rho=args.rho,
    )

@dataclass
class LoggedHedgeResult:
    # Terminal quantities
    pnl: torch.Tensor                      # [B]
    payoff: torch.Tensor                   # [B]
    terminal_wealth: torch.Tensor          # [B]
    final_position: torch.Tensor           # [B]
    liquidation_cost: torch.Tensor         # [B]
    total_transaction_cost: torch.Tensor   # [B]
    total_turnover: torch.Tensor           # [B]

    # Pathwise logged quantities
    wealth_path: torch.Tensor              # [B, K+1]
    positions: torch.Tensor                # [B, K]
    trades: torch.Tensor                   # [B, K]
    transaction_costs: torch.Tensor        # [B, K]
    spot_at_hedge_times: torch.Tensor      # [B, K]
    spot_after_interval: torch.Tensor      # [B, K]

    # Optional diagnostics
    hidden_states: torch.Tensor | None = None   # [B, K, H] or None
    hidden_variance: torch.Tensor | None = None # [B, K] or None


@dataclass
class LoggedHedgingEngine:
    product: Product
    controller: Controller
    feature_extractor: FeatureExtractor
    use_smooth_payoff: bool
    initial_cash: float
    transaction_cost_rate: float
    control_intervals: ControlIntervals
    log_hidden_state: bool = False

    def __post_init__(self):
        self.control_indices: ControlIndices | None = None

    def bind(self, sim_grid: SimulationGrid) -> None:
        self.control_indices = ControlIndices(
            start_idx=sim_grid.find_times_in_grid(self.control_intervals.start_times),
            end_idx=sim_grid.find_times_in_grid(self.control_intervals.end_times),
        )

    def run(self, simulated: SimulationResult) -> LoggedHedgeResult:
        if self.control_indices is None:
            raise RuntimeError("Bind to SimulationGrid before calling run.")

        S = simulated.spot
        batch_size = S.size(0)
        n_intervals = self.control_intervals.n_intervals()
        device = S.device
        dtype = S.dtype

        payoff = (
            self.product.compute_smooth_payoff(simulated)
            if self.use_smooth_payoff
            else self.product.compute_payoff(simulated)
        )
        if payoff.ndim == 2 and payoff.shape[1] == 1:
            payoff = payoff.squeeze(1)
        if payoff.ndim != 1 or payoff.shape[0] != batch_size:
            raise ValueError(f"Expected payoff shape ({batch_size},), got {tuple(payoff.shape)}")

        wealth = torch.full((batch_size,), self.initial_cash, dtype=dtype, device=device)
        position = torch.zeros((batch_size,), dtype=dtype, device=device)

        # Logged pathwise tensors
        wealth_path = torch.empty((batch_size, n_intervals + 1), dtype=dtype, device=device)
        wealth_path[:, 0] = wealth

        positions = torch.empty((batch_size, n_intervals), dtype=dtype, device=device)
        trades = torch.empty((batch_size, n_intervals), dtype=dtype, device=device)
        transaction_costs = torch.empty((batch_size, n_intervals), dtype=dtype, device=device)
        spot_at_hedge_times = torch.empty((batch_size, n_intervals), dtype=dtype, device=device)
        spot_after_interval = torch.empty((batch_size, n_intervals), dtype=dtype, device=device)

        total_transaction_cost = torch.zeros((batch_size,), dtype=dtype, device=device)
        total_turnover = torch.zeros((batch_size,), dtype=dtype, device=device)

        # Optional feature / hidden-state logs
        hidden_logs = None
        hidden_variance = None

        hidden_state_dim = self.feature_extractor.hidden_state_dim()

        if self.log_hidden_state and hidden_state_dim is not None:
            hidden_logs = torch.empty(
                (batch_size, n_intervals, hidden_state_dim),
                dtype=dtype,
                device=device,
            )
            hidden_variance = torch.empty(
                (batch_size, n_intervals),
                dtype=dtype,
                device=device,
            )

        state = SimulationState()
        state.t_prev = 0.0
        state.spot_previous = S.select(1, 0)
        state.spot_cumulative_min = S.select(1, 0)
        state.spot_cumulative_max = S.select(1, 0)

        if hidden_state_dim is not None:
            state.hidden_state = torch.zeros(
                (batch_size, hidden_state_dim),
                dtype=dtype,
                device=device,
            )

        for k in range(n_intervals):
            t0 = self.control_indices.start_idx[k]
            t1 = self.control_indices.end_idx[k]

            state.spot = S.select(1, t0)
            state.spot_cumulative_min = torch.minimum(state.spot_cumulative_min, state.spot)
            state.spot_cumulative_max = torch.maximum(state.spot_cumulative_max, state.spot)

            if simulated.variance is not None:
                state.variance = simulated.variance.select(1, t0)
            if simulated.short_rate is not None:
                state.short_rate = simulated.short_rate.select(1, t0)

            state.t = self.control_intervals.start_times[k]
            state.t_next = self.control_intervals.end_times[k]

            fe_output = self.feature_extractor.get_features(state)
            hedge = self.controller.forward(fe_output.features)

            if hedge.ndim == 2 and hedge.shape[1] == 1:
                hedge = hedge.squeeze(1)
            if hedge.ndim != 1 or hedge.shape[0] != batch_size:
                raise ValueError(f"Expected hedge shape ({batch_size},), got {tuple(hedge.shape)}")

            trade = hedge - position
            cost = self.transaction_cost_rate * state.spot * torch.abs(trade)

            next_spot = S.select(1, t1)
            interval_pnl = hedge * (next_spot - state.spot)
            wealth = wealth + interval_pnl - cost

            # Log current step
            positions[:, k] = hedge
            trades[:, k] = trade
            transaction_costs[:, k] = cost
            spot_at_hedge_times[:, k] = state.spot
            spot_after_interval[:, k] = next_spot
            wealth_path[:, k + 1] = wealth

            total_transaction_cost = total_transaction_cost + cost
            total_turnover = total_turnover + torch.abs(trade)

            if self.log_hidden_state and hidden_logs is not None:
                if fe_output.hidden_state is None:
                    raise RuntimeError(
                        "log_hidden_state=True, but feature extractor returned no hidden state."
                    )
                hidden_logs[:, k, :] = fe_output.hidden_state
                hidden_variance[:, k] = state.variance

            # Advance state
            state.hidden_state = fe_output.hidden_state
            state.t_prev = state.t
            state.spot_previous = state.spot
            position = hedge

        final_spot = S.select(1, self.control_indices.end_idx[-1])
        liquidation_cost = self.transaction_cost_rate * final_spot * torch.abs(position)
        wealth = wealth - liquidation_cost
        total_transaction_cost = total_transaction_cost + liquidation_cost

        pnl = wealth - payoff

        return LoggedHedgeResult(
            pnl=pnl,
            payoff=payoff,
            terminal_wealth=wealth,
            final_position=position,
            liquidation_cost=liquidation_cost,
            total_transaction_cost=total_transaction_cost,
            total_turnover=total_turnover,
            wealth_path=wealth_path,
            positions=positions,
            trades=trades,
            transaction_costs=transaction_costs,
            spot_at_hedge_times=spot_at_hedge_times,
            spot_after_interval=spot_after_interval,
            hidden_states=hidden_logs,
            hidden_variance=hidden_variance,
        )

def _to_serializable(obj: Any) -> Any:
    if is_dataclass(obj):
        return _to_serializable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item()
        return obj.detach().cpu().tolist()
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    if isinstance(obj, float):
        if torch.isnan(torch.tensor(obj)):
            return "nan"
        if torch.isinf(torch.tensor(obj)):
            return "inf" if obj > 0 else "-inf"
        return obj
    return str(obj)

def _sanitize_float_str(x: float) -> str:
    """
    Make floats compact and directory-name safe.
    """
    s = f"{x:.12g}"
    s = s.replace("-", "m").replace(".", "p")
    return s


def make_run_name(config: ExperimentConfig) -> str:
    """
    Human-readable run name from the swept parameters.

    Includes all experiment axes and a short hash to avoid collisions
    if additional config fields are added later.
    """

    hidden = (
        "x".join(str(h) for h in config.hidden_sizes)
        if config.hidden_sizes
        else "linear"
    )

    vf = int(config.variance_feature_type)

    payload = json.dumps(_to_serializable(config), sort_keys=True)
    short_hash = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]

    return (
        f"hs_{hidden}"
        f"__bar_{_sanitize_float_str(config.barrier)}"
        f"__tc_{_sanitize_float_str(config.transaction_cost_rate)}"
        f"__xi_{_sanitize_float_str(config.xi)}"
        f"__vf_{vf}"
        f"__loss_{config.risk_name}"
        f"__seed_{config.seed}"
        f"__{short_hash}"
    )


def _flatten_1d(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().to(torch.float64).cpu()
    if x.ndim != 1:
        raise ValueError(f"Expected 1D tensor, got shape {tuple(x.shape)}")
    return x

def _quantiles(x: torch.Tensor, probs: list[float]) -> dict[str, float]:
    q = torch.tensor(probs, dtype=torch.float64)
    vals = torch.quantile(x, q)
    out: dict[str, float] = {}
    for p, v in zip(probs, vals):
        pct = int(round(100 * p))
        out[f"q{pct:02d}"] = float(v.item())
    return out


def _var_cvar_from_pnl(pnl: torch.Tensor, alpha: float) -> dict[str, float]:
    """
    Treat losses as -PnL.
    VaR_alpha(loss) and CVaR_alpha(loss).
    For example alpha=0.95 means worst 5% tail.
    """
    losses = -pnl
    var_level = torch.quantile(losses, alpha)
    tail = losses[losses >= var_level]
    cvar = tail.mean() if tail.numel() > 0 else var_level
    return {
        "loss_var_alpha": float(var_level.item()),
        "loss_cvar_alpha": float(cvar.item()),
    }

def _basic_1d_stats(x: torch.Tensor, *, include_quantiles: bool = True) -> dict[str, Any]:
    x = _flatten_1d(x)
    n = int(x.numel())

    if n == 0:
        return {"count": 0}

    mean = x.mean()
    std = x.std(unbiased=False)
    min_ = x.min()
    max_ = x.max()

    # Population central moments
    centered = x - mean
    var = centered.square().mean()

    if var.item() > 0.0:
        m3 = centered.pow(3).mean()
        m4 = centered.pow(4).mean()

        skewness = m3 / var.pow(1.5)
        kurtosis = m4 / var.square()
        excess_kurtosis = kurtosis - 3.0

        skewness_value = float(skewness.item())
        kurtosis_value = float(kurtosis.item())
        excess_kurtosis_value = float(excess_kurtosis.item())
    else:
        skewness_value = float("nan")
        kurtosis_value = float("nan")
        excess_kurtosis_value = float("nan")

    out: dict[str, Any] = {
        "count": n,
        "mean": float(mean.item()),
        "std": float(std.item()),
        "min": float(min_.item()),
        "max": float(max_.item()),
        "skewness": skewness_value,
        "kurtosis": kurtosis_value,
        "excess_kurtosis": excess_kurtosis_value,
        "positive_fraction": float((x > 0).to(torch.float64).mean().item()),
        "negative_fraction": float((x < 0).to(torch.float64).mean().item()),
        "zero_fraction": float((x == 0).to(torch.float64).mean().item()),
    }

    if include_quantiles:
        out["quantiles"] = _quantiles(x, [0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99])

    return out

def _time_series_summary(x: torch.Tensor, name: str) -> dict[str, Any]:
    """
    Summarize [B, K] or [B, K+1] tensor over paths at each time index.
    Avoid storing all paths; keep per-time cross-sectional summaries.
    """
    x = x.detach().to(torch.float64).cpu()

    if x.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {tuple(x.shape)}")

    mean_t = x.mean(dim=0)
    std_t = x.std(dim=0, unbiased=False)
    p05_t = torch.quantile(x, 0.05, dim=0)
    p50_t = torch.quantile(x, 0.50, dim=0)
    p95_t = torch.quantile(x, 0.95, dim=0)

    return {
        "shape": list(x.shape),
        "time_mean": mean_t.tolist(),
        "time_std": std_t.tolist(),
        "time_q05": p05_t.tolist(),
        "time_q50": p50_t.tolist(),
        "time_q95": p95_t.tolist(),
        "global_mean": float(x.mean().item()),
        "global_std": float(x.std(unbiased=False).item()),
        "global_min": float(x.min().item()),
        "global_max": float(x.max().item()),
    }

def _optional_tensor_summary(x: torch.Tensor | None, name: str) -> dict[str, Any] | None:
    if x is None:
        return None

    x = x.detach().to(torch.float64).cpu()
    out: dict[str, Any] = {
        "shape": list(x.shape),
        "global_mean": float(x.mean().item()),
        "global_std": float(x.std(unbiased=False).item()),
        "global_min": float(x.min().item()),
        "global_max": float(x.max().item()),
    }

    # For [B, K, F] or [B, K, H], also keep mean/std over batch and time per feature dimension.
    if x.ndim == 3:
        mean_last = x.mean(dim=(0, 1))
        std_last = x.std(dim=(0, 1), unbiased=False)
        out["mean_by_last_dim"] = mean_last.tolist()
        out["std_by_last_dim"] = std_last.tolist()

    return out

def summarize_logged_hedge_result(
    result: LoggedHedgeResult,
    config: ExperimentConfig,
) -> dict[str, Any]:
    pnl = _flatten_1d(result.pnl)
    payoff = _flatten_1d(result.payoff)
    terminal_wealth = _flatten_1d(result.terminal_wealth)
    final_position = _flatten_1d(result.final_position)
    liquidation_cost = _flatten_1d(result.liquidation_cost)
    total_transaction_cost = _flatten_1d(result.total_transaction_cost)
    total_turnover = _flatten_1d(result.total_turnover)

    alpha = config.cvar_alpha

    hidden_variance_last = None
    if result.hidden_variance is not None:
        hv = result.hidden_variance.detach().to(torch.float64).cpu()
        if hv.ndim != 2:
            raise ValueError(
                f"hidden_variance must have shape [B, K], got {tuple(hv.shape)}"
            )
        hidden_variance_last = hv[:, -1]

    hidden_state_variance_corr = None
    if result.hidden_states is not None and result.hidden_variance is not None:
        hs = result.hidden_states.detach().to(torch.float64).cpu()
        hv = result.hidden_variance.detach().to(torch.float64).cpu()

        # hidden_states may be [B, K] or [B, K, H]
        if hs.ndim == 2:
            hs_flat = hs.reshape(-1)
            hv_flat = hv.reshape(-1)
            if hs_flat.numel() > 1:
                hidden_state_variance_corr = float(
                    torch.corrcoef(torch.stack([hs_flat, hv_flat]))[0, 1].item()
                )
        elif hs.ndim == 3:
            # correlation of each hidden dimension with latent variance
            hv_flat = hv.reshape(-1)
            corr_by_hidden_dim = []
            for h in range(hs.shape[2]):
                hs_flat = hs[:, :, h].reshape(-1)
                if hs_flat.numel() > 1:
                    corr = torch.corrcoef(torch.stack([hs_flat, hv_flat]))[0, 1].item()
                    corr_by_hidden_dim.append(float(corr))
                else:
                    corr_by_hidden_dim.append(float("nan"))
            hidden_state_variance_corr = {
                "corr_by_hidden_dim": corr_by_hidden_dim,
                "max_abs_corr": float(
                    max(abs(c) for c in corr_by_hidden_dim)
                ) if corr_by_hidden_dim else float("nan"),
            }

    hidden_variance_position_corr = None
    hidden_variance_trade_corr = None
    if result.hidden_variance is not None:
        hv = result.hidden_variance.detach().to(torch.float64).cpu().reshape(-1)
        pos = result.positions.detach().to(torch.float64).cpu().reshape(-1)
        trd = result.trades.detach().to(torch.float64).cpu().reshape(-1)

        if hv.numel() > 1:
            hidden_variance_position_corr = float(
                torch.corrcoef(torch.stack([hv, pos]))[0, 1].item()
            )
            hidden_variance_trade_corr = float(
                torch.corrcoef(torch.stack([hv, trd]))[0, 1].item()
            )

    metrics: dict[str, Any] = {
        "run_summary": {
            "n_eval_paths": int(pnl.numel()),
            "hedging_steps": int(result.positions.shape[1]),
            "wealth_path_steps": int(result.wealth_path.shape[1]),
            "has_hidden_states": result.hidden_states is not None,
            "has_hidden_variance": result.hidden_variance is not None,
        },
        "terminal_metrics": {
            "pnl": {
                **_basic_1d_stats(pnl),
                **_var_cvar_from_pnl(pnl, alpha),
            },
            "payoff": _basic_1d_stats(payoff),
            "terminal_wealth": _basic_1d_stats(terminal_wealth),
            "final_position": _basic_1d_stats(final_position),
            "liquidation_cost": _basic_1d_stats(liquidation_cost),
            "total_transaction_cost": _basic_1d_stats(total_transaction_cost),
            "total_turnover": _basic_1d_stats(total_turnover),
        },
        "derived_metrics": {
            "mean_abs_final_position": float(final_position.abs().mean().item()),
            "mean_squared_final_position": float(final_position.square().mean().item()),
            "mean_cost_per_unit_turnover": float(
                (total_transaction_cost / total_turnover.clamp_min(1e-12)).mean().item()
            ),
            "fraction_nonzero_final_position": float(
                (final_position != 0).to(torch.float64).mean().item()
            ),
            "fraction_positive_pnl": float((pnl > 0).to(torch.float64).mean().item()),
            "fraction_negative_pnl": float((pnl < 0).to(torch.float64).mean().item()),
            "corr_payoff_pnl": float(
                torch.corrcoef(torch.stack([payoff, pnl]))[0, 1].item()
            ) if pnl.numel() > 1 else float("nan"),
            "corr_turnover_pnl": float(
                torch.corrcoef(torch.stack([total_turnover, pnl]))[0, 1].item()
            ) if pnl.numel() > 1 else float("nan"),
            "corr_cost_pnl": float(
                torch.corrcoef(torch.stack([total_transaction_cost, pnl]))[0, 1].item()
            ) if pnl.numel() > 1 else float("nan"),
            "corr_hidden_variance_positions": hidden_variance_position_corr,
            "corr_hidden_variance_trades": hidden_variance_trade_corr,
            "hidden_state_hidden_variance_correlation": hidden_state_variance_corr,
        },
        "path_summaries": {
            "wealth_path": _time_series_summary(result.wealth_path, "wealth_path"),
            "positions": _time_series_summary(result.positions, "positions"),
            "trades": _time_series_summary(result.trades, "trades"),
            "transaction_costs": _time_series_summary(
                result.transaction_costs, "transaction_costs"
            ),
            "hidden_variance": (
                _time_series_summary(result.hidden_variance, "hidden_variance")
                if result.hidden_variance is not None else None
            ),
        },
        "optional_summaries": {
            "hidden_states": _optional_tensor_summary(result.hidden_states, "hidden_states"),
        },
    }

    return metrics

def save_experiment_summary(
    config: ExperimentConfig,
    result: LoggedHedgeResult,
    training_summary: TrainingSummary,
    controller: torch.nn.Module,
    feature_extractor: torch.nn.Module | None = None,
) -> Path:
    output_root = Path(config.log_dir)
    run_dir = output_root / make_run_name(config)
    run_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.json"
    training_path = run_dir / "training.json"
    controller_path = run_dir / "controller.pt"
    feature_extractor_path = run_dir / "feature_extractor.pt"

    config_payload = _to_serializable(config)
    metrics_payload = summarize_logged_hedge_result(result, config)
    training_payload = _to_serializable(training_summary)

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2, sort_keys=True)

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(_to_serializable(metrics_payload), f, indent=2, sort_keys=True)

    with training_path.open("w", encoding="utf-8") as f:
        json.dump(training_payload, f, indent=2, sort_keys=True)

    torch.save(controller.state_dict(), controller_path)

    if feature_extractor is not None:
        torch.save(feature_extractor.state_dict(), feature_extractor_path)

    return run_dir

def _jsonable(x: Any) -> Any:
    if isinstance(x, Enum):
        # Prefer stable string names for cache keys.
        return x.name
    if is_dataclass(x):
        return {k: _jsonable(v) for k, v in asdict(x).items()}
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def make_pricing_payload(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "product": {
            "barrier": float(config.barrier),
            "strike": float(config.strike),
            "maturity": float(config.maturity),
        },
        "model": {
            "s0": float(config.s0),
            "v0": float(config.v0),
            "kappa": float(config.kappa),
            "theta": float(config.theta),
            "xi": float(config.xi),
            "rho": float(config.rho),
        },
        "simulation": {
            "observation_freq": int(config.observation_freq),
        },
    }


def make_pricing_key(config: ExperimentConfig) -> str:
    payload = make_pricing_payload(config)
    return json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"))


def get_or_compute_initial_cash(
    config: ExperimentConfig,
    model: HestonModel,
    product: DownAndOutBarrierOption,
    device: torch.device | str,
    *,
    n_pricing_paths: int = 2**20,
) -> float:
    output_root = Path(config.log_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    premia_path = output_root / "premia.json"
    pricing_key = make_pricing_key(config)

    # Load cache if present
    if premia_path.is_file():
        with premia_path.open("r", encoding="utf-8") as f:
            premia = json.load(f)
    else:
        premia = {}

    # Cache hit
    if pricing_key in premia:
        initial_cash = float(premia[pricing_key]["premium"])
        return initial_cash

    # Cache miss: compute premium
    with torch.no_grad():
        model = model.to(device)
        product = product.to(device)

        time_grid = product.get_timegrid()
        model.bind(time_grid)
        product.bind(time_grid)

        pricing_paths = model.simulate(n_pricing_paths)
        payoff = product.compute_payoff(pricing_paths)
        initial_cash = float(payoff.mean().item())

        # Optional diagnostic: MC standard error
        payoff_std = float(payoff.std(unbiased=False).item())
        mc_se = payoff_std / (n_pricing_paths ** 0.5)

    premia[pricing_key] = {
        "premium": initial_cash,
        "mc_price_n_paths": int(n_pricing_paths),
        "mc_standard_error": mc_se,
        "pricing_payload": make_pricing_payload(config),
    }

    with premia_path.open("w", encoding="utf-8") as f:
        json.dump(premia, f, indent=2, sort_keys=True)

    return initial_cash

def main():
    config = parse_args()
    pprint.pprint(config)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    learning_rate = config.learning_rate
    cvar_alpha = config.cvar_alpha
    cvar_softplus_beta = config.cvar_softplus_beta
    barrier = config.barrier
    strike = config.strike
    maturity = config.maturity
    transaction_cost_rate = config.transaction_cost_rate
    observation_freq = config.observation_freq
    hedging_freq = config.hedging_freq
    hidden_sizes = config.hidden_sizes
    n_eval_paths = config.n_eval_paths
    n_train_iterations = config.n_train_iterations
    batch_size = config.batch_size
    variance_feature_type = config.variance_feature_type
    risk_name = config.risk_name

    parameters = HestonParameters(
        s0=config.s0,
        v0=config.v0,
        kappa=config.kappa,
        theta=config.theta,
        xi=config.xi,
        rho=config.rho,
    )

    model = HestonModel(parameters)
    product_grid = SimulationGrid(torch.linspace(0.0, maturity, int(observation_freq * maturity) + 1))
    control_grid = SimulationGrid(torch.linspace(0.0, maturity, int(hedging_freq * maturity) + 1).to(device))
    product = DownAndOutBarrierOption(
        maturity=maturity,
        barrier=barrier,
        strike=strike,
        observation_grid=product_grid.time_grid,
    )

    initial_cash = get_or_compute_initial_cash(
        config=config,
        model=model,
        product=product,
        device=device,
        n_pricing_paths=2**20,
    )

    feature_extractor = BarrierOptionFeatureExtractor(
        maturity=maturity,
        barrier=barrier,
        strike=strike,
        variance_feature_type=variance_feature_type,
    )
    controller = MlpController(
        feature_dim=feature_extractor.feature_dim(),
        hidden_sizes=hidden_sizes
    )
    
    if risk_name == "cvar":
        risk = CVaRRisk(alpha=cvar_alpha, softplus_beta=cvar_softplus_beta, var_estimate_type=VaREstimateType.BATCH)
    elif risk_name == "mse":
        risk = MSERisk()
    else:
        raise ValueError(f"Unknown risk_name: {risk_name}")
    
    controller.to(device)
    model.to(device)
    product.to(device)
    feature_extractor.to(device)
    risk.to(device)

    optim = torch.optim.Adam(
        chain(
            controller.parameters(),
            feature_extractor.parameters(),
        ), 
        lr=learning_rate
    )

    for item in model.parameters():
        item.requires_grad_(False)
    for item in product.parameters():
        item.requires_grad_(False)
    for item in feature_extractor.parameters():
        item.requires_grad_(True)
    for item in controller.parameters():
        item.requires_grad_(True)

    controller.to(device)
    model.to(device)
    product.to(device)
    feature_extractor.to(device)
    risk.to(device)

    trainer = HedgingTrainer(
        model=model,
        product=product,
        feature_extractor=feature_extractor,
        controller=controller,
        risk=risk,
        initial_cash=initial_cash,
        transaction_cost_rate=transaction_cost_rate,
        optim=optim,
    )

    def eval(batch_size: int) -> None:
        hedging_engine = LoggedHedgingEngine(
            product=product,
            controller=controller,
            feature_extractor=feature_extractor,
            use_smooth_payoff=False,
            initial_cash=initial_cash,
            transaction_cost_rate=transaction_cost_rate,
            control_intervals=ControlIntervals(
                start_times=control_grid.time_grid[:-1],
                end_times=control_grid.time_grid[1:],
            ),
            log_hidden_state=variance_feature_type == VarianceFeatureType.LEARNED_FILTER
        )

        risk = CVaRRisk(alpha=cvar_alpha, softplus_beta=None, var_estimate_type=VaREstimateType.BATCH)
        risk.to(device)
        master_grid = control_grid.merge(product.get_timegrid())

        model.bind(master_grid)
        product.bind(master_grid)
        hedging_engine.bind(master_grid)

        with torch.no_grad():
            simulated = model.simulate(batch_size=batch_size)
            hedge_result = hedging_engine.run(simulated)
        return hedge_result

    trainer.set_control_grid(control_grid)
    training_summary = trainer.train(n_train_iterations, batch_size)
    eval_result = eval(n_eval_paths)
    save_experiment_summary(
        config=config, 
        result=eval_result,
        training_summary=training_summary,
        controller=controller,
        feature_extractor=feature_extractor,
    )

if __name__ == "__main__":
    main()
