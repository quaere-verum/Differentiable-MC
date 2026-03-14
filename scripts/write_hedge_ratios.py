from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from dmc.control.mlp_controller import MlpController
from dmc.feature_extraction.barrier_feature_extractor import (
    DownAndOutCallFeatureExtractor,
    DownAndOutPutFeatureExtractor,
    VarianceFeatureType,
)


def safe_read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def load_terminal_payload(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload in {path}, got {type(payload)}")

    out: dict[str, torch.Tensor] = {}
    for k, v in payload.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu()
        else:
            out[k] = torch.as_tensor(v)
    return out


def instantiate_feature_extractor(config: dict[str, Any]) -> torch.nn.Module:
    variance_feature_type = config["variance_feature_type"]
    put_or_call = str(config["put_or_call"]).lower()

    kwargs = dict(
        maturity=float(config["maturity"]),
        barrier=float(config["barrier"]),
        strike=float(config["strike"]),
        variance_feature_type=variance_feature_type,
    )

    if put_or_call == "call":
        return DownAndOutCallFeatureExtractor(**kwargs)
    elif put_or_call == "put":
        return DownAndOutPutFeatureExtractor(**kwargs)
    else:
        raise ValueError(f"Unknown put_or_call={put_or_call!r}")


def instantiate_controller(config: dict[str, Any], feature_dim: int) -> torch.nn.Module:
    hidden_sizes = tuple(int(x) for x in config.get("hidden_sizes", []))
    return MlpController(feature_dim=feature_dim, hidden_sizes=hidden_sizes)


def get_state_slice_values(
    payload: dict[str, torch.Tensor],
    variance_feature_type: VarianceFeatureType,
    quantiles: list[float],
) -> tuple[list[float], str]:
    """
    Return representative low/med/high state-slice values from saved terminal distributions.
    """
    if variance_feature_type in (
        VarianceFeatureType.LEARNED_FILTER,
        VarianceFeatureType.LEARNED_GATED_FILTER,
    ):
        if "hidden_states" not in payload:
            raise KeyError("hidden_states not found in terminal_distributions.pt")

        hs = payload["hidden_states"].to(torch.float32)
        if hs.ndim == 3:
            if hs.shape[2] != 1:
                raise ValueError(f"Expected hidden_states shape [B, K, 1], got {tuple(hs.shape)}")
            hs = hs.squeeze(-1)
        elif hs.ndim != 2:
            raise ValueError(f"Expected hidden_states shape [B, K] or [B, K, 1], got {tuple(hs.shape)}")

        values = torch.quantile(hs.reshape(-1), torch.tensor(quantiles, dtype=torch.float32)).tolist()
        return values, "hidden state"

    elif variance_feature_type == VarianceFeatureType.MARKOV_STATE:
        if "hidden_variance" not in payload:
            raise KeyError("hidden_variance not found in terminal_distributions.pt")

        hv = payload["hidden_variance"].to(torch.float32)
        if hv.ndim != 2:
            raise ValueError(f"Expected hidden_variance shape [B, K], got {tuple(hv.shape)}")

        values = torch.quantile(hv.reshape(-1), torch.tensor(quantiles, dtype=torch.float32)).tolist()
        return values, "latent variance"

    elif variance_feature_type == VarianceFeatureType.NONE:
        return [0.0], "no variance state"

    else:
        raise ValueError(f"Unsupported variance feature type: {variance_feature_type}")


def build_policy_features(
    *,
    variance_feature_type: VarianceFeatureType,
    tau_grid: torch.Tensor,
    log_moneyness_grid: torch.Tensor,
    state_value: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build policy input features directly from the same conventions used by the feature extractor.

    Returns
    -------
    features : [N_tau * N_m, D]
    tau_mesh : [N_tau, N_m]
    logm_mesh : [N_tau, N_m]
    """
    tau_mesh, logm_mesh = torch.meshgrid(tau_grid, log_moneyness_grid, indexing="ij")
    alive = torch.ones_like(logm_mesh)

    base_features = [
        tau_mesh.reshape(-1, 1),
        logm_mesh.reshape(-1, 1),
        alive.reshape(-1, 1),
    ]

    if variance_feature_type in (
        VarianceFeatureType.LEARNED_FILTER,
        VarianceFeatureType.LEARNED_GATED_FILTER,
        VarianceFeatureType.MARKOV_STATE,
    ):
        state_col = torch.full_like(logm_mesh.reshape(-1, 1), float(state_value))
        base_features.append(state_col)

    features = torch.cat(base_features, dim=1)
    return features, tau_mesh, logm_mesh


def evaluate_controller_surface(
    controller: MlpController,
    features: torch.Tensor,
    n_tau: int,
    n_logm: int,
) -> torch.Tensor:
    with torch.no_grad():
        hedge = controller.forward(features)
        if hedge.ndim == 2 and hedge.shape[1] == 1:
            hedge = hedge.squeeze(1)
        hedge = hedge.reshape(n_tau, n_logm).detach().cpu().to(torch.float32)
    return hedge


def plot_heatmaps(
    *,
    out_path: Path,
    hedge_surfaces: list[torch.Tensor],
    tau_grid: torch.Tensor,
    log_moneyness_grid: torch.Tensor,
    state_values: list[float],
    state_label: str,
    put_or_call: str,
    barrier: float,
    strike: float,
) -> None:
    n_panels = len(hedge_surfaces)
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(6 * n_panels, 5),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    axes = axes[0]

    vmin = min(float(h.min()) for h in hedge_surfaces)
    vmax = max(float(h.max()) for h in hedge_surfaces)

    for ax, hedge, state_value in zip(axes, hedge_surfaces, state_values):
        im = ax.imshow(
            hedge.numpy(),
            origin="lower",
            aspect="auto",
            extent=[
                float(log_moneyness_grid.min()),
                float(log_moneyness_grid.max()),
                float(tau_grid.min()),
                float(tau_grid.max()),
            ],
            vmin=vmin,
            vmax=vmax,
        )

        # Barrier location in log-moneyness coordinates.
        if put_or_call.lower() == "call":
            barrier_logm = np.log(barrier / strike)
        else:
            barrier_logm = np.log(strike / barrier)

        ax.axvline(barrier_logm, color="white", linestyle="--", linewidth=1.5)
        ax.set_xlabel("Log moneyness")
        ax.set_ylabel("Time to maturity")
        ax.set_title(f"{state_label} = {state_value:.3f}")

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.95)
    cbar.set_label("Hedge ratio")

    fig.suptitle("Hedge-ratio heatmaps", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_curves(
    *,
    out_path: Path,
    controller: torch.nn.Module,
    put_or_call: str,
    variance_feature_type: VarianceFeatureType,
    maturity: float,
    barrier: float,
    strike: float,
    tau_slices: list[float],
    log_moneyness_grid: torch.Tensor,
    state_values: list[float],
    state_label: str,
) -> None:
    nrows = len(state_values)
    ncols = len(tau_slices)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5 * ncols, 4 * nrows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    # Barrier location in log-moneyness coordinates.
    if put_or_call.lower() == "call":
        barrier_logm = np.log(barrier / strike)
    else:
        barrier_logm = np.log(strike / barrier)

    for i, state_value in enumerate(state_values):
        for j, tau in enumerate(tau_slices):
            ax = axes[i][j]
            tau_grid = torch.tensor([tau], dtype=torch.float32)

            features, _, logm_mesh = build_policy_features(
                variance_feature_type=variance_feature_type,
                tau_grid=tau_grid,
                log_moneyness_grid=log_moneyness_grid,
                state_value=state_value,
            )
            hedge = evaluate_controller_surface(
                controller,
                features,
                n_tau=1,
                n_logm=log_moneyness_grid.numel(),
            ).squeeze(0)

            ax.plot(logm_mesh.squeeze(0).numpy(), hedge.numpy(), linewidth=2)
            ax.axvline(barrier_logm, color="black", linestyle="--", linewidth=1.0)
            ax.set_title(f"$\\tau$ = {tau:.2f}")
            ax.set_xlabel("Log moneyness")
            ax.set_ylabel("Hedge ratio")
            ax.text(
                0.02,
                0.95,
                f"{state_label} = {state_value:.3f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.75),
            )

    fig.suptitle("Hedge-ratio slices", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Write hedge-ratio visualisations from a logged run.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory containing config.json, controller.pt, feature_extractor.pt, terminal_distributions.pt")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--tensor-filename", type=str, default="terminal_distributions.pt")
    parser.add_argument("--n-logm", type=int, default=121, help="Number of log-moneyness grid points")
    parser.add_argument("--logm-min", type=float, default=-0.35)
    parser.add_argument("--logm-max", type=float, default=0.35)
    parser.add_argument("--n-tau", type=int, default=81, help="Number of tau grid points for heatmaps")
    parser.add_argument("--tau-min", type=float, default=0.02)
    parser.add_argument("--tau-max", type=float, default=1.0)
    parser.add_argument("--state-quantiles", type=float, nargs="*", default=[0.10, 0.50, 0.90])
    parser.add_argument("--tau-slices", type=float, nargs="*", default=[0.75, 0.25, 0.05])
    args = parser.parse_args()

    run_dir = args.run_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    config = safe_read_json(run_dir / "config.json")
    payload = load_terminal_payload(run_dir / args.tensor_filename)

    put_or_call = str(config["put_or_call"]).lower()
    maturity = float(config["maturity"])
    barrier = float(config["barrier"])
    strike = float(config["strike"])
    variance_feature_type = config["variance_feature_type"]

    feature_extractor = instantiate_feature_extractor(config)
    controller = instantiate_controller(config, feature_extractor.feature_dim())

    feature_extractor.load_state_dict(torch.load(run_dir / "feature_extractor.pt", map_location="cpu"))
    controller.load_state_dict(torch.load(run_dir / "controller.pt", map_location="cpu"))

    feature_extractor.eval()
    controller.eval()

    state_values, state_label = get_state_slice_values(
        payload,
        variance_feature_type=variance_feature_type,
        quantiles=list(args.state_quantiles),
    )

    tau_grid = torch.linspace(args.tau_min, args.tau_max, args.n_tau, dtype=torch.float32)
    log_moneyness_grid = torch.linspace(args.logm_min, args.logm_max, args.n_logm, dtype=torch.float32)

    hedge_surfaces: list[torch.Tensor] = []
    for state_value in state_values:
        features, _, _ = build_policy_features(
            variance_feature_type=variance_feature_type,
            tau_grid=tau_grid,
            log_moneyness_grid=log_moneyness_grid,
            state_value=state_value,
        )
        hedge = evaluate_controller_surface(
            controller,
            features,
            n_tau=tau_grid.numel(),
            n_logm=log_moneyness_grid.numel(),
        )
        hedge_surfaces.append(hedge)

    out_heatmap = out_dir / "hedge_ratio_heatmaps.png"
    plot_heatmaps(
        out_path=out_heatmap,
        hedge_surfaces=hedge_surfaces,
        tau_grid=tau_grid,
        log_moneyness_grid=log_moneyness_grid,
        state_values=state_values,
        state_label=state_label,
        put_or_call=put_or_call,
        barrier=barrier,
        strike=strike,
    )
    print(f"Wrote {out_heatmap}")

    out_curves = out_dir / "hedge_ratio_curves.png"
    plot_curves(
        out_path=out_curves,
        controller=controller,
        put_or_call=put_or_call,
        variance_feature_type=variance_feature_type,
        maturity=maturity,
        barrier=barrier,
        strike=strike,
        tau_slices=list(args.tau_slices),
        log_moneyness_grid=log_moneyness_grid,
        state_values=state_values,
        state_label=state_label,
    )
    print(f"Wrote {out_curves}")


if __name__ == "__main__":
    main()