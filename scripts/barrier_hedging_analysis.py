from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def flatten_dict(
    d: dict[str, Any],
    parent_key: str = "",
    sep: str = ".",
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, key, sep=sep))
        else:
            out[key] = v
    return out


def safe_read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def available(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def ensure_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def coalesce_columns(df: pd.DataFrame, new_col: str, candidates: list[str]) -> pd.DataFrame:
    vals = None
    for c in candidates:
        if c in df.columns:
            s = df[c]
            vals = s if vals is None else vals.combine_first(s)
    if vals is not None:
        df[new_col] = vals
    return df


def normalise_hidden_sizes(x: Any) -> str:
    if x is None:
        return "unknown"

    if isinstance(x, (list, tuple)):
        return "linear" if len(x) == 0 else "x".join(str(int(v)) for v in x)

    s = str(x).strip()
    if s in {"[]", "()", ""}:
        return "linear"

    nums = re.findall(r"-?\d+", s)
    if nums:
        return "x".join(nums)

    return s


def infer_variance_feature_label(x: Any) -> str:
    """
    Map stored variance feature field to {"markov", "learned", "none"}.
    """
    if x is None:
        return "unknown"

    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"markov", "learned", "none"}:
            return s
        for token in ("markov", "learned", "none"):
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
    }
    return mapping.get(i, f"vf_{i}")


def dataframe_to_markdown(df: pd.DataFrame, index: bool = False, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._\n"
    return df.head(max_rows).to_markdown(index=index) + "\n"


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------

def load_single_run(run_dir: Path) -> dict[str, Any] | None:
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.json"
    training_path = run_dir / "training.json"

    if not config_path.exists() or not metrics_path.exists():
        return None

    config = safe_read_json(config_path)
    metrics = safe_read_json(metrics_path)
    training = safe_read_json(training_path) if training_path.exists() else {}

    if config is None or metrics is None:
        return None

    row: dict[str, Any] = {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
    }
    row.update({f"config.{k}": v for k, v in flatten_dict(config).items()})
    row.update({f"metrics.{k}": v for k, v in flatten_dict(metrics).items()})
    row.update({f"training.{k}": v for k, v in flatten_dict(training).items()})
    return row


def load_all_runs(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for run_dir in root.rglob("*"):
        if run_dir.is_dir():
            row = load_single_run(run_dir)
            if row is not None:
                rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # -------------------------------------------------------------------------
    # Normalised sweep axes
    # -------------------------------------------------------------------------
    df["hidden_arch"] = df.get(
        "config.hidden_sizes", pd.Series([None] * len(df))
    ).map(normalise_hidden_sizes)

    vf_col = first_existing_column(
        df,
        ["config.variance_feature_type", "config.variance_feature", "config.vf"],
    )
    if vf_col is not None:
        df["variance_feature"] = df[vf_col].map(infer_variance_feature_label)
    else:
        df["variance_feature"] = "unknown"

    risk_col = first_existing_column(df, ["config.risk_name", "config.loss_name"])
    if risk_col is not None:
        df["loss_name"] = df[risk_col].astype(str).str.lower()
    else:
        df["loss_name"] = "unknown"

    for src, dst in [
        ("config.seed", "seed"),
        ("config.barrier", "barrier"),
        ("config.transaction_cost_rate", "transaction_cost_rate"),
        ("config.xi", "xi"),
        ("config.cvar_alpha", "cvar_alpha"),
    ]:
        if src in df.columns:
            df[dst] = df[src]

    # -------------------------------------------------------------------------
    # Canonical terminal metrics from _basic_1d_stats and _var_cvar_from_pnl
    # -------------------------------------------------------------------------
    # PnL
    df = coalesce_columns(df, "pnl_mean", ["metrics.terminal_metrics.pnl.mean"])
    df = coalesce_columns(df, "pnl_std", ["metrics.terminal_metrics.pnl.std"])
    df = coalesce_columns(df, "pnl_min", ["metrics.terminal_metrics.pnl.min"])
    df = coalesce_columns(df, "pnl_max", ["metrics.terminal_metrics.pnl.max"])
    df = coalesce_columns(df, "pnl_skewness", ["metrics.terminal_metrics.pnl.skewness"])
    df = coalesce_columns(df, "pnl_kurtosis", ["metrics.terminal_metrics.pnl.kurtosis"])
    df = coalesce_columns(
        df, "pnl_excess_kurtosis", ["metrics.terminal_metrics.pnl.excess_kurtosis"]
    )
    df = coalesce_columns(
        df, "pnl_positive_fraction", ["metrics.terminal_metrics.pnl.positive_fraction"]
    )
    df = coalesce_columns(
        df, "pnl_negative_fraction", ["metrics.terminal_metrics.pnl.negative_fraction"]
    )
    df = coalesce_columns(
        df, "pnl_zero_fraction", ["metrics.terminal_metrics.pnl.zero_fraction"]
    )
    df = coalesce_columns(df, "pnl_q01", ["metrics.terminal_metrics.pnl.quantiles.q01"])
    df = coalesce_columns(df, "pnl_q05", ["metrics.terminal_metrics.pnl.quantiles.q05"])
    df = coalesce_columns(df, "pnl_q10", ["metrics.terminal_metrics.pnl.quantiles.q10"])
    df = coalesce_columns(df, "pnl_q50", ["metrics.terminal_metrics.pnl.quantiles.q50"])
    df = coalesce_columns(df, "pnl_q90", ["metrics.terminal_metrics.pnl.quantiles.q90"])
    df = coalesce_columns(df, "pnl_q95", ["metrics.terminal_metrics.pnl.quantiles.q95"])
    df = coalesce_columns(df, "pnl_q99", ["metrics.terminal_metrics.pnl.quantiles.q99"])
    df = coalesce_columns(
        df,
        "loss_var_alpha",
        ["metrics.terminal_metrics.pnl.loss_var_alpha"],
    )
    df = coalesce_columns(
        df,
        "loss_cvar_alpha",
        ["metrics.terminal_metrics.pnl.loss_cvar_alpha"],
    )

    # Other terminal summaries
    df = coalesce_columns(
        df, "payoff_mean", ["metrics.terminal_metrics.payoff.mean"]
    )
    df = coalesce_columns(
        df, "terminal_wealth_mean", ["metrics.terminal_metrics.terminal_wealth.mean"]
    )
    df = coalesce_columns(
        df, "final_position_mean", ["metrics.terminal_metrics.final_position.mean"]
    )
    df = coalesce_columns(
        df, "final_position_std", ["metrics.terminal_metrics.final_position.std"]
    )
    df = coalesce_columns(
        df, "final_position_abs_q95", ["metrics.terminal_metrics.final_position.quantiles.q95"]
    )
    df = coalesce_columns(
        df,
        "liquidation_cost_mean",
        ["metrics.terminal_metrics.liquidation_cost.mean"],
    )
    df = coalesce_columns(
        df,
        "transaction_cost_mean",
        ["metrics.terminal_metrics.total_transaction_cost.mean"],
    )
    df = coalesce_columns(
        df,
        "transaction_cost_std",
        ["metrics.terminal_metrics.total_transaction_cost.std"],
    )
    df = coalesce_columns(
        df,
        "turnover_mean",
        ["metrics.terminal_metrics.total_turnover.mean"],
    )
    df = coalesce_columns(
        df,
        "turnover_std",
        ["metrics.terminal_metrics.total_turnover.std"],
    )
    df = coalesce_columns(
        df,
        "turnover_q95",
        ["metrics.terminal_metrics.total_turnover.quantiles.q95"],
    )

    # -------------------------------------------------------------------------
    # Derived metrics
    # -------------------------------------------------------------------------
    df = coalesce_columns(
        df,
        "mean_abs_final_position",
        ["metrics.derived_metrics.mean_abs_final_position"],
    )
    df = coalesce_columns(
        df,
        "mean_squared_final_position",
        ["metrics.derived_metrics.mean_squared_final_position"],
    )
    df = coalesce_columns(
        df,
        "mean_cost_per_unit_turnover",
        ["metrics.derived_metrics.mean_cost_per_unit_turnover"],
    )
    df = coalesce_columns(
        df,
        "fraction_nonzero_final_position",
        ["metrics.derived_metrics.fraction_nonzero_final_position"],
    )
    df = coalesce_columns(
        df,
        "fraction_positive_pnl",
        ["metrics.derived_metrics.fraction_positive_pnl"],
    )
    df = coalesce_columns(
        df,
        "fraction_negative_pnl",
        ["metrics.derived_metrics.fraction_negative_pnl"],
    )
    df = coalesce_columns(
        df,
        "corr_payoff_pnl",
        ["metrics.derived_metrics.corr_payoff_pnl"],
    )
    df = coalesce_columns(
        df,
        "corr_turnover_pnl",
        ["metrics.derived_metrics.corr_turnover_pnl"],
    )
    df = coalesce_columns(
        df,
        "corr_cost_pnl",
        ["metrics.derived_metrics.corr_cost_pnl"],
    )
    df = coalesce_columns(
        df,
        "corr_hidden_variance_positions",
        ["metrics.derived_metrics.corr_hidden_variance_positions"],
    )
    df = coalesce_columns(
        df,
        "corr_hidden_variance_trades",
        ["metrics.derived_metrics.corr_hidden_variance_trades"],
    )
    df = coalesce_columns(
        df,
        "hidden_state_variance_max_abs_corr",
        ["metrics.derived_metrics.hidden_state_hidden_variance_correlation.max_abs_corr"],
    )

    # -------------------------------------------------------------------------
    # Path summaries from _time_series_summary
    # -------------------------------------------------------------------------
    for base_name, out_prefix in [
        ("wealth_path", "wealth_path"),
        ("positions", "positions"),
        ("trades", "trades"),
        ("transaction_costs", "transaction_costs"),
        ("hidden_variance", "hidden_variance"),
    ]:
        base = f"metrics.path_summaries.{base_name}"
        df = coalesce_columns(df, f"{out_prefix}_global_mean", [f"{base}.global_mean"])
        df = coalesce_columns(df, f"{out_prefix}_global_std", [f"{base}.global_std"])
        df = coalesce_columns(df, f"{out_prefix}_global_min", [f"{base}.global_min"])
        df = coalesce_columns(df, f"{out_prefix}_global_max", [f"{base}.global_max"])

    # -------------------------------------------------------------------------
    # Optional summaries from _optional_tensor_summary
    # -------------------------------------------------------------------------
    df = coalesce_columns(
        df,
        "hidden_states_global_mean",
        ["metrics.optional_summaries.hidden_states.global_mean"],
    )
    df = coalesce_columns(
        df,
        "hidden_states_global_std",
        ["metrics.optional_summaries.hidden_states.global_std"],
    )
    df = coalesce_columns(
        df,
        "hidden_states_global_min",
        ["metrics.optional_summaries.hidden_states.global_min"],
    )
    df = coalesce_columns(
        df,
        "hidden_states_global_max",
        ["metrics.optional_summaries.hidden_states.global_max"],
    )

    numeric_cols = [
        "seed",
        "barrier",
        "transaction_cost_rate",
        "xi",
        "cvar_alpha",
        "pnl_mean",
        "pnl_std",
        "pnl_min",
        "pnl_max",
        "pnl_skewness",
        "pnl_kurtosis",
        "pnl_excess_kurtosis",
        "pnl_positive_fraction",
        "pnl_negative_fraction",
        "pnl_zero_fraction",
        "pnl_q01",
        "pnl_q05",
        "pnl_q10",
        "pnl_q50",
        "pnl_q90",
        "pnl_q95",
        "pnl_q99",
        "loss_var_alpha",
        "loss_cvar_alpha",
        "payoff_mean",
        "terminal_wealth_mean",
        "final_position_mean",
        "final_position_std",
        "final_position_abs_q95",
        "liquidation_cost_mean",
        "transaction_cost_mean",
        "transaction_cost_std",
        "turnover_mean",
        "turnover_std",
        "turnover_q95",
        "mean_abs_final_position",
        "mean_squared_final_position",
        "mean_cost_per_unit_turnover",
        "fraction_nonzero_final_position",
        "fraction_positive_pnl",
        "fraction_negative_pnl",
        "corr_payoff_pnl",
        "corr_turnover_pnl",
        "corr_cost_pnl",
        "corr_hidden_variance_positions",
        "corr_hidden_variance_trades",
        "hidden_state_variance_max_abs_corr",
        "wealth_path_global_mean",
        "wealth_path_global_std",
        "positions_global_mean",
        "positions_global_std",
        "trades_global_mean",
        "trades_global_std",
        "transaction_costs_global_mean",
        "transaction_costs_global_std",
        "hidden_variance_global_mean",
        "hidden_variance_global_std",
        "hidden_states_global_mean",
        "hidden_states_global_std",
    ]
    df = ensure_numeric(df, numeric_cols)

    return df


# -----------------------------------------------------------------------------
# Aggregation
# -----------------------------------------------------------------------------

SWEEP_AXES = [
    "hidden_arch",
    "barrier",
    "transaction_cost_rate",
    "xi",
    "loss_name",
    "variance_feature",
]


def aggregate_over_seeds(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    group_cols = [c for c in SWEEP_AXES if c in df.columns]

    metric_cols = available(
        df,
        [
            "pnl_mean",
            "pnl_std",
            "pnl_skewness",
            "pnl_excess_kurtosis",
            "pnl_q01",
            "pnl_q05",
            "pnl_q50",
            "pnl_q95",
            "pnl_q99",
            "loss_var_alpha",
            "loss_cvar_alpha",
            "turnover_mean",
            "turnover_q95",
            "transaction_cost_mean",
            "mean_abs_final_position",
            "mean_squared_final_position",
            "mean_cost_per_unit_turnover",
            "fraction_positive_pnl",
            "fraction_negative_pnl",
            "corr_turnover_pnl",
            "corr_cost_pnl",
            "corr_payoff_pnl",
            "corr_hidden_variance_positions",
            "corr_hidden_variance_trades",
            "hidden_state_variance_max_abs_corr",
            "wealth_path_global_mean",
            "wealth_path_global_std",
            "positions_global_std",
            "trades_global_std",
            "hidden_variance_global_mean",
            "hidden_variance_global_std",
        ],
    )

    agg_spec = {m: ["mean", "std", "min", "max", "count"] for m in metric_cols}
    grouped = df.groupby(group_cols, dropna=False).agg(agg_spec)
    grouped.columns = ["__".join(c).strip("_") for c in grouped.columns.to_flat_index()]
    grouped = grouped.reset_index()

    count_cols = [c for c in grouped.columns if c.endswith("__count")]
    if count_cols:
        grouped["n_seeds"] = grouped[count_cols[0]]

    return grouped


def make_pivot(
    df: pd.DataFrame,
    value_col: str,
    index: list[str],
    columns: list[str],
    aggfunc: str = "mean",
) -> pd.DataFrame:
    idx = [c for c in index if c in df.columns]
    cols = [c for c in columns if c in df.columns]
    if value_col not in df.columns or not idx or not cols:
        return pd.DataFrame()

    pt = pd.pivot_table(
        df,
        values=value_col,
        index=idx,
        columns=cols,
        aggfunc=aggfunc,
        dropna=False,
    )
    if isinstance(pt.columns, pd.MultiIndex):
        pt.columns = ["_".join(map(str, tup)) for tup in pt.columns.to_flat_index()]
    return pt.reset_index()


def pairwise_delta_table(
    df: pd.DataFrame,
    group_axes: list[str],
    compare_col: str,
    left_value: str,
    right_value: str,
    metric_cols: list[str],
    suffix: str,
) -> pd.DataFrame:
    needed = [compare_col] + group_axes + metric_cols
    needed = [c for c in needed if c in df.columns]
    if compare_col not in needed:
        return pd.DataFrame()

    sub = df[needed].copy()
    metric_cols = [c for c in metric_cols if c in sub.columns]
    if not metric_cols:
        return pd.DataFrame()

    agg = (
        sub.groupby(group_axes + [compare_col], dropna=False)[metric_cols]
        .mean(numeric_only=True)
        .reset_index()
    )

    left = agg[agg[compare_col] == left_value].copy()
    right = agg[agg[compare_col] == right_value].copy()

    if left.empty or right.empty:
        return pd.DataFrame()

    merged = left.merge(
        right,
        on=group_axes,
        suffixes=(f"__{left_value}", f"__{right_value}"),
        how="inner",
    )

    for m in metric_cols:
        merged[f"{m}__delta_{suffix}"] = (
            merged[f"{m}__{left_value}"] - merged[f"{m}__{right_value}"]
        )

    return merged


def rank_runs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Heuristic ranking:
    - prefer lower tail risk by default: smaller loss_cvar_alpha is better
    - then smaller loss_var_alpha
    - then larger pnl_mean
    - then smaller pnl_std
    """
    out = df.copy()

    if "loss_cvar_alpha" in out.columns:
        out["rank_key_1"] = out["loss_cvar_alpha"]
    else:
        out["rank_key_1"] = np.nan

    if "loss_var_alpha" in out.columns:
        out["rank_key_2"] = out["loss_var_alpha"]
    else:
        out["rank_key_2"] = np.nan

    if "pnl_mean" in out.columns:
        out["rank_key_3"] = -out["pnl_mean"]
    else:
        out["rank_key_3"] = np.nan

    if "pnl_std" in out.columns:
        out["rank_key_4"] = out["pnl_std"]
    else:
        out["rank_key_4"] = np.nan

    sort_cols = ["rank_key_1", "rank_key_2", "rank_key_3", "rank_key_4"]
    out = out.sort_values(sort_cols, ascending=[True, True, True, True], na_position="last")
    out["global_rank"] = np.arange(1, len(out) + 1)

    return out


# -----------------------------------------------------------------------------
# Report
# -----------------------------------------------------------------------------

def write_report(
    out_path: Path,
    runs_df: pd.DataFrame,
    agg_df: pd.DataFrame,
    pivots: dict[str, pd.DataFrame],
    pairwise: dict[str, pd.DataFrame],
) -> None:
    lines: list[str] = []
    lines.append("# Deep Hedging Experiment Analysis\n\n")
    lines.append(f"- Number of runs loaded: **{len(runs_df)}**\n")

    for c in ["hidden_arch", "barrier", "transaction_cost_rate", "xi", "loss_name", "variance_feature"]:
        if c in runs_df.columns:
            vals = sorted(runs_df[c].dropna().astype(str).unique().tolist())
            lines.append(f"- `{c}`: {', '.join(vals)}\n")

    lines.append("\n## Top runs\n\n")
    top_cols = available(
        runs_df,
        [
            "global_rank",
            "run_name",
            "seed",
            "hidden_arch",
            "barrier",
            "transaction_cost_rate",
            "xi",
            "loss_name",
            "variance_feature",
            "pnl_mean",
            "pnl_std",
            "loss_var_alpha",
            "loss_cvar_alpha",
            "turnover_mean",
            "transaction_cost_mean",
            "mean_abs_final_position",
            "pnl_skewness",
            "pnl_excess_kurtosis",
            "hidden_state_variance_max_abs_corr",
        ],
    )
    ranked = rank_runs(runs_df)
    lines.append(dataframe_to_markdown(ranked[top_cols].head(20)))

    lines.append("\n## Seed-aggregated summary\n\n")
    agg_cols = available(
        agg_df,
        [
            "hidden_arch",
            "barrier",
            "transaction_cost_rate",
            "xi",
            "loss_name",
            "variance_feature",
            "pnl_mean__mean",
            "pnl_std__mean",
            "loss_var_alpha__mean",
            "loss_cvar_alpha__mean",
            "turnover_mean__mean",
            "transaction_cost_mean__mean",
            "mean_abs_final_position__mean",
            "hidden_state_variance_max_abs_corr__mean",
            "n_seeds",
        ],
    )
    lines.append(dataframe_to_markdown(agg_df[agg_cols] if agg_cols else agg_df))

    for name, df in pivots.items():
        lines.append(f"\n## {name}\n\n")
        lines.append(dataframe_to_markdown(df))

    for name, df in pairwise.items():
        lines.append(f"\n## {name}\n\n")
        lines.append(dataframe_to_markdown(df))

    out_path.write_text("".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse logged deep hedging experiment runs.")
    parser.add_argument("--root", type=Path, required=True, help="Root log directory.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory.")
    args = parser.parse_args()

    root = args.root
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    runs = load_all_runs(root)
    if runs.empty:
        raise SystemExit(f"No runs found under {root}")

    runs = rank_runs(runs)

    runs.to_csv(out / "runs_flat.csv", index=False)

    agg = aggregate_over_seeds(runs)
    agg.to_csv(out / "seed_aggregated.csv", index=False)

    ranked = rank_runs(runs)
    ranked.to_csv(out / "runs_ranked.csv", index=False)

    pivot_source = agg if not agg.empty else runs

    pivots: dict[str, pd.DataFrame] = {}

    value_col = first_existing_column(
        pivot_source,
        ["loss_cvar_alpha__mean", "loss_cvar_alpha"],
    )
    if value_col is not None:
        pivots["pivot_tail_risk_by_variance_feature"] = make_pivot(
            pivot_source,
            value_col=value_col,
            index=["hidden_arch", "barrier", "transaction_cost_rate", "xi", "loss_name"],
            columns=["variance_feature"],
        )

    value_col = first_existing_column(
        pivot_source,
        ["pnl_mean__mean", "pnl_mean"],
    )
    if value_col is not None:
        pivots["pivot_pnl_mean_by_variance_feature"] = make_pivot(
            pivot_source,
            value_col=value_col,
            index=["hidden_arch", "barrier", "transaction_cost_rate", "xi", "loss_name"],
            columns=["variance_feature"],
        )

    value_col = first_existing_column(
        pivot_source,
        ["turnover_mean__mean", "turnover_mean"],
    )
    if value_col is not None:
        pivots["pivot_turnover_by_variance_feature"] = make_pivot(
            pivot_source,
            value_col=value_col,
            index=["barrier", "transaction_cost_rate", "xi", "loss_name"],
            columns=["variance_feature", "hidden_arch"],
        )

    value_col = first_existing_column(
        pivot_source,
        ["transaction_cost_mean__mean", "transaction_cost_mean"],
    )
    if value_col is not None:
        pivots["pivot_transaction_cost_by_variance_feature"] = make_pivot(
            pivot_source,
            value_col=value_col,
            index=["barrier", "transaction_cost_rate", "xi", "loss_name"],
            columns=["variance_feature", "hidden_arch"],
        )

    value_col = first_existing_column(
        pivot_source,
        ["hidden_state_variance_max_abs_corr__mean", "hidden_state_variance_max_abs_corr"],
    )
    if value_col is not None:
        pivots["pivot_hidden_state_variance_corr"] = make_pivot(
            pivot_source,
            value_col=value_col,
            index=["barrier", "transaction_cost_rate", "xi", "loss_name"],
            columns=["hidden_arch", "variance_feature"],
        )

    value_col = first_existing_column(
        pivot_source,
        ["pnl_excess_kurtosis__mean", "pnl_excess_kurtosis"],
    )
    if value_col is not None:
        pivots["pivot_pnl_excess_kurtosis"] = make_pivot(
            pivot_source,
            value_col=value_col,
            index=["barrier", "transaction_cost_rate", "xi", "loss_name"],
            columns=["variance_feature", "hidden_arch"],
        )

    for name, df in pivots.items():
        df.to_csv(out / f"{name}.csv", index=False)

    pair_metrics = available(
        runs,
        [
            "pnl_mean",
            "pnl_std",
            "loss_var_alpha",
            "loss_cvar_alpha",
            "pnl_q05",
            "pnl_q95",
            "turnover_mean",
            "transaction_cost_mean",
            "mean_abs_final_position",
            "mean_cost_per_unit_turnover",
            "pnl_skewness",
            "pnl_excess_kurtosis",
            "hidden_state_variance_max_abs_corr",
        ],
    )

    pairwise: dict[str, pd.DataFrame] = {}

    df_cmp = pairwise_delta_table(
        runs,
        group_axes=["hidden_arch", "barrier", "transaction_cost_rate", "xi", "loss_name"],
        compare_col="variance_feature",
        left_value="learned",
        right_value="markov",
        metric_cols=pair_metrics,
        suffix="learned_minus_markov",
    )
    if not df_cmp.empty:
        pairwise["pairwise_learned_minus_markov"] = df_cmp
        df_cmp.to_csv(out / "pairwise_learned_minus_markov.csv", index=False)

    df_cmp = pairwise_delta_table(
        runs,
        group_axes=["hidden_arch", "barrier", "transaction_cost_rate", "xi", "loss_name"],
        compare_col="variance_feature",
        left_value="learned",
        right_value="none",
        metric_cols=pair_metrics,
        suffix="learned_minus_none",
    )
    if not df_cmp.empty:
        pairwise["pairwise_learned_minus_none"] = df_cmp
        df_cmp.to_csv(out / "pairwise_learned_minus_none.csv", index=False)

    df_cmp = pairwise_delta_table(
        runs,
        group_axes=["hidden_arch", "barrier", "transaction_cost_rate", "xi", "variance_feature"],
        compare_col="loss_name",
        left_value="cvar",
        right_value="mse",
        metric_cols=pair_metrics,
        suffix="cvar_minus_mse",
    )
    if not df_cmp.empty:
        pairwise["pairwise_cvar_minus_mse"] = df_cmp
        df_cmp.to_csv(out / "pairwise_cvar_minus_mse.csv", index=False)

    df_cmp = pairwise_delta_table(
        runs,
        group_axes=["barrier", "transaction_cost_rate", "xi", "loss_name", "variance_feature"],
        compare_col="hidden_arch",
        left_value="16x16",
        right_value="linear",
        metric_cols=pair_metrics,
        suffix="16x16_minus_linear",
    )
    if not df_cmp.empty:
        pairwise["pairwise_16x16_minus_linear"] = df_cmp
        df_cmp.to_csv(out / "pairwise_16x16_minus_linear.csv", index=False)

    write_report(out / "report.md", runs, agg, pivots, pairwise)

    print(f"Loaded {len(runs)} runs from: {root}")
    print(f"Wrote analysis files to: {out}")


if __name__ == "__main__":
    main()