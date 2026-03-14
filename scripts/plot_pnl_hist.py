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


def load_pnl_tensor(run_dir: Path, tensor_filename: str) -> torch.Tensor:
    tensor_path = run_dir / tensor_filename
    if not tensor_path.is_file():
        raise FileNotFoundError(f"Expected tensor file not found: {tensor_path}")

    payload = torch.load(tensor_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload in {tensor_path}, got {type(payload)}")
    if "pnl" not in payload:
        raise KeyError(f"'pnl' key not found in {tensor_path}. Available keys: {list(payload.keys())}")

    pnl = payload["pnl"]
    if not isinstance(pnl, torch.Tensor):
        pnl = torch.as_tensor(pnl)
    return pnl.detach().to(torch.float32).cpu().reshape(-1)


def choose_bins(series_by_feature: dict[str, torch.Tensor], n_bins: int = 120) -> torch.Tensor:
    vals = [v for v in series_by_feature.values() if v.numel() > 0]
    if not vals:
        raise ValueError("No non-empty PnL tensors available for histogram plotting.")
    all_vals = torch.cat(vals)
    lo = torch.quantile(all_vals, 0.001).item()
    hi = torch.quantile(all_vals, 0.999).item()
    if hi <= lo:
        lo = float(all_vals.min().item())
        hi = float(all_vals.max().item())
    if hi <= lo:
        hi = lo + 1e-6
    return torch.linspace(lo, hi, n_bins + 1)


def plot_histogram_for_regime(
    df_regime: pd.DataFrame,
    out_path: Path,
    regime_name: str,
    loss_name: str,
    tensor_filename: str,
    density: bool = True,
    n_bins: int = 120,
) -> None:
    series_by_feature: dict[str, torch.Tensor] = {}

    for feature in FEATURE_ORDER:
        sub = df_regime[df_regime["variance_feature"] == feature]
        if sub.empty:
            continue
        if len(sub) > 1:
            sub = sub.sort_values(["seed", "run_name"]).head(1)
        run_dir = Path(sub.iloc[0]["run_dir"])
        series_by_feature[feature] = load_pnl_tensor(run_dir, tensor_filename)

    if not series_by_feature:
        raise ValueError(f"No matching runs found for regime={regime_name}, loss={loss_name}")

    bins = choose_bins(series_by_feature, n_bins=n_bins)

    plt.figure(figsize=(9, 5))
    for feature in FEATURE_ORDER:
        if feature not in series_by_feature:
            continue
        arr = series_by_feature[feature].numpy()
        plt.hist(
            arr,
            bins=bins.numpy(),
            density=density,
            alpha=0.35,
            label=FEATURE_LABELS.get(feature, feature),
            histtype="step",
        )

    plt.xlabel("Terminal PnL")
    plt.ylabel("Density" if density else "Count")
    plt.title(f"PnL distribution — vol regime={regime_name}, loss={loss_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


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
                pnl = load_pnl_tensor(run_dir, tensor_filename)

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot PnL histograms for selected architecture and volatility regimes.")
    parser.add_argument("--log-dir", type=Path, required=True, help="Root log directory containing run subdirectories.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for histogram figures.")
    parser.add_argument("--arch", type=str, default="16x16x16", help="Hidden architecture label after normalisation, e.g. 16x16x16.")
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
    for regime in args.regimes:
        df = dfbase[
            (dfbase["vol_regime"] == regime)
        ].copy()
        if df.empty:
            raise SystemExit("No runs match the requested filters after applying architecture / tc / barrier / loss / seed filters.")
        for loss_name in ["cvar", "mse"]:
            df_loss = df[df["loss_name"] == loss_name]            

            if df_loss.empty:
                raise SystemExit("No runs match the requested filters after applying architecture / tc / barrier / loss / seed filters.")
            out_path_hist = out_dir / f"pnl_hist_arch_{args.arch}__loss_{loss_name}__regime_{regime}.png"
            plot_histogram_for_regime(
                df_regime=df_loss,
                out_path=out_path_hist,
                regime_name=regime,
                loss_name=loss_name,
                tensor_filename=args.tensor_filename,
                density=not args.count_hist,
            )
            print(f"Wrote {out_path_hist}")



if __name__ == "__main__":
    main()
