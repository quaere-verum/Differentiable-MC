from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch


METRIC_SPECS: dict[str, list[str]] = {
    "pnl": ["mean", "std", "loss_var_alpha", "loss_cvar_alpha"],
    "total_transaction_cost": ["mean", "std"],
    "total_turnover": ["mean", "std"],
}


FEATURE_LABELS = {
    "none": "None",
    "learned": "Learned Filter",
    "gated": "Gated Filter",
    "markov": "Oracle Variance",
}

FEATURE_ORDER = ["none", "learned", "gated", "markov"]


def safe_read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def flatten_dict(d: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, key, sep=sep))
        else:
            out[key] = v
    return out


def infer_variance_feature_label(x: Any) -> str:
    if x is None:
        return "unknown"

    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"markov", "learned", "none", "gated"}:
            return s
        for token in ("markov", "learned", "none", "gated"):
            if token in s:
                return token
        return s

    try:
        i = int(x)
    except Exception:
        return str(x)

    mapping = {
        1: "none",
        2: "learned",
        3: "markov",
        4: "gated",
    }
    return mapping.get(i, f"vf_{i}")


def normalise_hidden_sizes(x: Any) -> str:
    if x is None:
        return "unknown"

    if isinstance(x, (list, tuple)):
        return "linear" if len(x) == 0 else "x".join(str(int(v)) for v in x)

    s = str(x).strip()
    if s in {"[]", "()", ""}:
        return "linear"

    nums = []
    token = ""
    for ch in s:
        if ch.isdigit():
            token += ch
        else:
            if token:
                nums.append(token)
                token = ""
    if token:
        nums.append(token)

    if nums:
        return "x".join(nums)

    return s


def load_runs(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for p in root.rglob("metrics.json"):
        run_dir = p.parent
        config = safe_read_json(run_dir / "config.json")
        metrics = safe_read_json(run_dir / "metrics.json")
        if config is None or metrics is None:
            continue

        row: dict[str, Any] = {
            "run_dir": str(run_dir),
            "run_name": run_dir.name,
        }
        row.update({f"config.{k}": v for k, v in flatten_dict(config).items()})
        row.update({f"metrics.{k}": v for k, v in flatten_dict(metrics).items()})

        row["transaction_cost_rate"] = float(config["transaction_cost_rate"])
        row["vol_regime"] = config["vol_regime"]
        row["loss_name"] = str(config.get("risk_name", config.get("loss_name", "unknown"))).lower()
        row["variance_feature"] = infer_variance_feature_label(config.get("variance_feature_type"))
        row["hidden_arch"] = normalise_hidden_sizes(config.get("hidden_sizes"))
        row["seed"] = int(config.get("seed", 0))
        row["barrier"] = float(config.get("barrier")) if config.get("barrier") is not None else None
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    for metric_name, fields in METRIC_SPECS.items():
        for field in fields:
            src = f"metrics.terminal_metrics.{metric_name}.{field}"
            dst = f"{metric_name}.{field}"
            if src in df.columns:
                df[dst] = pd.to_numeric(df[src], errors="coerce")

    return df


def load_terminal_payload(run_dir: Path, tensor_filename: str) -> dict[str, torch.Tensor]:
    tensor_path = run_dir / tensor_filename
    if not tensor_path.is_file():
        raise FileNotFoundError(f"Expected tensor file not found: {tensor_path}")

    payload = torch.load(tensor_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload in {tensor_path}, got {type(payload)}")

    out: dict[str, torch.Tensor] = {}
    for key, value in payload.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.detach().cpu()
        else:
            out[key] = torch.as_tensor(value)

    return out


def plot_log_survival_comparison(
    df_base: pd.DataFrame,
    out_path: Path,
    tensor_filename: str,
    quantile: float = 0.95,
) -> None:
    """
    Plot empirical survival functions of losses L = -PnL.

    Colour  -> information regime
    Linestyle -> loss objective (MSE vs CVaR)

    Parameters
    ----------
    quantile:
        tail zoom level. 0.95 means plot worst 5% losses.
    """

    COLOR_MAP = {
        "none": "#d62728",
        "filter": "#1f77b4",
        "gated": "#2ca02c",
        "markov": "#9467bd",
    }

    LINESTYLE_MAP = {
        "mse": "-",
        "cvar": "--",
    }

    fig, ax = plt.subplots(figsize=(15,5), ncols=2, sharex=True, sharey=True)

    for k, vol_regime in enumerate(["normal", "stressed"]):
        series = []
        df_regime = df_base[df_base["vol_regime"] == vol_regime]
        for feature in FEATURE_ORDER:
            for loss in ["mse", "cvar"]:

                sub = df_regime[
                    (df_regime["variance_feature"] == feature) &
                    (df_regime["loss_name"] == loss)
                ]

                if sub.empty:
                    continue

                if len(sub) > 1:
                    sub = sub.sort_values(["seed", "run_name"]).head(1)

                run_dir = Path(sub.iloc[0]["run_dir"])
                pnl = load_terminal_payload(run_dir, tensor_filename)["pnl"]

                losses = -pnl
                series.append((feature, loss, losses))

        if not series:
            raise ValueError("No matching runs found.")

        pooled = torch.cat([s[2] for s in series])

        if quantile == 0:
            x_min = float(pooled.min())
        else:
            x_min = float(torch.quantile(pooled, quantile))

        x_max = float(pooled.max())

        for feature, loss, losses in series:

            x = torch.sort(losses).values
            n = x.numel()

            cdf = torch.arange(1, n + 1, dtype=torch.float32) / n
            survival = 1 - cdf

            eps = 1/(n+1)
            survival = torch.clamp(survival, min=eps)

            mask = x >= x_min

            ax[k].plot(
                x[mask].numpy(),
                survival[mask].numpy(),
                color=COLOR_MAP.get(feature, "black"),
                linestyle=LINESTYLE_MAP.get(loss, "-"),
                linewidth=2,
                label=f"{FEATURE_LABELS.get(feature,feature)} / {loss.upper()}"
            )

        ax[k].set_yscale("log")
        ax[k].set_xlim(left=x_min, right=x_max)

        ax[k].set_xlabel("Loss  $L=-\\mathrm{PnL}$")
        ax[k].set_ylabel("$P(L>x)$")

        tail_pct = int((1-quantile)*100)

        ax[k].set_title(
            f"Loss tail survival (worst {tail_pct}% losses)\n"
            f"Vol regime: {vol_regime}"
        )

        # ax[k].legend()
    handles, labels = ax[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False
    )

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    plt.savefig(out_path, dpi=220)
    plt.close()

def plot_hidden_state_vs_hidden_variance(
    df_base: pd.DataFrame,
    out_path: Path,
    tensor_filename: str,
    *,
    variance_features: list[str] | None = None,
    losses: list[str] | None = None,
    max_points: int = 100_000,
) -> None:
    """
    Scatter plot of learned hidden state vs true latent variance.

    Expected saved tensors:
        hidden_states   : [B, K, 1] or [B, K]
        hidden_variance : [B, K]

    One subplot per (vol_regime, variance_feature, loss) combination found.
    """
    if variance_features is None:
        variance_features = ["learned", "gated"]
    if losses is None:
        losses = ["mse", "cvar"]

    panels: list[tuple[str, str, str, Path]] = []
    for vol_regime in ["normal", "stressed"]:
        for feature in variance_features:
            for loss in losses:
                sub = df_base[
                    (df_base["vol_regime"] == vol_regime)
                    & (df_base["variance_feature"] == feature)
                    & (df_base["loss_name"] == loss)
                ]
                if sub.empty:
                    continue
                sub = sub.sort_values(["seed", "run_name"]).head(1)
                run_dir = Path(sub.iloc[0]["run_dir"])
                panels.append((vol_regime, feature, loss, run_dir))

    if not panels:
        raise ValueError("No matching runs found for hidden-state scatter plot.")

    n_panels = len(panels)
    ncols = min(2, n_panels)
    nrows = (n_panels + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(7 * ncols, 5 * nrows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for ax, (vol_regime, feature, loss, run_dir) in zip(axes_flat, panels):
        payload = load_terminal_payload(run_dir, tensor_filename)

        if "hidden_states" not in payload or "hidden_variance" not in payload:
            ax.set_visible(False)
            continue

        hs = payload["hidden_states"].to(torch.float32)
        hv = payload["hidden_variance"].to(torch.float32)

        if hs.ndim == 3:
            if hs.shape[2] != 1:
                raise ValueError(
                    f"Expected hidden_states shape [B, K, 1], got {tuple(hs.shape)}"
                )
            hs = hs.squeeze(-1)
        elif hs.ndim != 2:
            raise ValueError(f"Expected hidden_states shape [B, K] or [B, K, 1], got {tuple(hs.shape)}")

        if hv.ndim != 2:
            raise ValueError(f"Expected hidden_variance shape [B, K], got {tuple(hv.shape)}")

        x = hs.reshape(-1)
        y = hv.reshape(-1)

        n = x.numel()
        if n > max_points:
            idx = torch.randperm(n)[:max_points]
            x = x[idx]
            y = y[idx]

        ax.scatter(
            x.numpy(),
            y.numpy(),
            s=4,
            alpha=0.15,
            linewidths=0,
        )
        ax.set_xlabel("Learned hidden state")
        ax.set_ylabel("Latent variance")
        ax.set_title(
            f"{FEATURE_LABELS.get(feature, feature)} / {loss.upper()}\n"
            f"Vol regime: {vol_regime}"
        )

        if x.numel() > 1:
            corr = torch.corrcoef(torch.stack([x, y]))[0, 1].item()
            ax.text(
                0.02,
                0.98,
                f"corr = {corr:.3f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

    for ax in axes_flat[n_panels:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

def plot_hidden_state_vs_future_integrated_variance(
    df_base: pd.DataFrame,
    out_path: Path,
    tensor_filename: str,
    *,
    variance_features: list[str] | None = None,
    losses: list[str] | None = None,
    maturity: float = 1.0,
    n_steps: int = 252,
    max_points: int = 100_000,
) -> None:
    r"""
    Scatter plot of learned hidden state h_t against
    integrated future latent variance \int_t^T v_s ds,
    approximated on the saved hedge-time grid.

    This uses hidden_variance, not realised future variance.
    """
    if variance_features is None:
        variance_features = ["learned", "gated"]
    if losses is None:
        losses = ["mse", "cvar"]

    dt = float(maturity) / float(n_steps)

    panels: list[tuple[str, str, str, Path]] = []
    for vol_regime in ["normal", "stressed"]:
        for feature in variance_features:
            for loss in losses:
                sub = df_base[
                    (df_base["vol_regime"] == vol_regime)
                    & (df_base["variance_feature"] == feature)
                    & (df_base["loss_name"] == loss)
                ]
                if sub.empty:
                    continue
                sub = sub.sort_values(["seed", "run_name"]).head(1)
                run_dir = Path(sub.iloc[0]["run_dir"])
                panels.append((vol_regime, feature, loss, run_dir))

    if not panels:
        raise ValueError("No matching runs found for future-integrated-variance scatter plot.")

    n_panels = len(panels)
    ncols = min(2, n_panels)
    nrows = (n_panels + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(7 * ncols, 5 * nrows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for ax, (vol_regime, feature, loss, run_dir) in zip(axes_flat, panels):
        payload = load_terminal_payload(run_dir, tensor_filename)

        if "hidden_states" not in payload or "hidden_variance" not in payload:
            ax.set_visible(False)
            continue

        hs = payload["hidden_states"].to(torch.float32)
        hv = payload["hidden_variance"].to(torch.float32)

        if hs.ndim == 3:
            if hs.shape[2] != 1:
                raise ValueError(
                    f"Expected hidden_states shape [B, K, 1], got {tuple(hs.shape)}"
                )
            hs = hs.squeeze(-1)
        elif hs.ndim != 2:
            raise ValueError(f"Expected hidden_states shape [B, K] or [B, K, 1], got {tuple(hs.shape)}")

        if hv.ndim != 2:
            raise ValueError(f"Expected hidden_variance shape [B, K], got {tuple(hv.shape)}")

        # Approximate \int_t^T v_s ds by reverse cumulative sum.
        future_integrated_var = torch.flip(
            torch.cumsum(torch.flip(hv, dims=[1]), dim=1),
            dims=[1],
        ) * dt

        x = hs.reshape(-1)
        y = future_integrated_var.reshape(-1)

        n = x.numel()
        if n > max_points:
            idx = torch.randperm(n)[:max_points]
            x = x[idx]
            y = y[idx]

        ax.scatter(
            x.numpy(),
            y.numpy(),
            s=4,
            alpha=0.15,
            linewidths=0,
        )
        ax.set_xlabel("Learned hidden state")
        ax.set_ylabel(r"Approx. future integrated latent variance")
        ax.set_title(
            f"{FEATURE_LABELS.get(feature, feature)} / {loss.upper()}\n"
            f"Vol regime: {vol_regime}"
        )

        if x.numel() > 1:
            corr = torch.corrcoef(torch.stack([x, y]))[0, 1].item()
            ax.text(
                0.02,
                0.98,
                f"corr = {corr:.3f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

    for ax in axes_flat[n_panels:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot PnL histograms for selected architecture and volatility regimes.")
    parser.add_argument("--log-dir", type=Path, required=True, help="Root log directory containing run subdirectories.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for histogram figures.")
    parser.add_argument("--arch", type=str, default="64x64", help="Hidden architecture label after normalisation, e.g. 16x16.")
    parser.add_argument("--tc", type=float, default=1e-3, help="Transaction cost rate to filter on.")
    parser.add_argument("--barrier", type=float, default=80.0, help="Barrier level to filter on.")
    parser.add_argument("--seed", type=int, default=0, help="Seed to filter on.")
    parser.add_argument("--tensor-filename", type=str, default="terminal_distributions.pt", help="Filename of saved tensor payload inside each run directory.")
    parser.add_argument("--regimes", nargs="*", default=["normal", "stressed"], help="Volatility regimes to plot.")
    parser.add_argument("--count-hist", action="store_true", help="Plot counts instead of densities.")
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    dfbase = load_runs(args.log_dir)
    dfbase[
        (dfbase["hidden_arch"] == args.arch)
        & (dfbase["transaction_cost_rate"] == float(args.tc))
        & (dfbase["seed"] == int(args.seed))
    ].copy()
    if dfbase.empty:
        raise SystemExit(f"No runs found under {args.log_dir}")
    if "barrier" in dfbase.columns:
        dfbase = dfbase[dfbase["barrier"] == float(args.barrier)]
    
    out_path_surv = out_dir / f"pnl_survival_arch_{args.arch}.png"
    plot_log_survival_comparison(
        df_base=dfbase,
        out_path=out_path_surv,
        tensor_filename=args.tensor_filename,
        quantile=0.50,
    )
    print(f"Wrote {out_path_surv}")

    out_path_tail = out_dir / f"pnl_tail_arch_{args.arch}.png"
    plot_log_survival_comparison(
        df_base=dfbase,
        out_path=out_path_tail,
        tensor_filename=args.tensor_filename,
        quantile=0.95,
    )
    print(f"Wrote {out_path_tail}")

    out_path_hv = out_dir / f"hidden_state_vs_hidden_variance_arch_{args.arch}.png"
    plot_hidden_state_vs_hidden_variance(
        df_base=dfbase,
        out_path=out_path_hv,
        tensor_filename=args.tensor_filename,
        variance_features=["learned", "gated"],
        losses=["cvar"],
        max_points=100_000,
    )
    print(f"Wrote {out_path_hv}")

    out_path_fiv = out_dir / f"hidden_state_vs_future_integrated_variance_arch_{args.arch}.png"
    plot_hidden_state_vs_future_integrated_variance(
        df_base=dfbase,
        out_path=out_path_fiv,
        tensor_filename=args.tensor_filename,
        variance_features=["learned", "gated"],
        losses=["cvar"],
        maturity=1.0,
        n_steps=252,
        max_points=100_000,
    )
    print(f"Wrote {out_path_fiv}")



if __name__ == "__main__":
    main()
