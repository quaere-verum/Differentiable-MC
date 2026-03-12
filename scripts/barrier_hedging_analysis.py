#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


METRIC_SPECS: dict[str, list[str]] = {
    "pnl": [
        "mean", "std", "min", "max",
        "loss_var_alpha", "loss_cvar_alpha",
        "skewness", "excess_kurtosis",
    ],
    "total_transaction_cost": [
        "mean", "std", "min", "max",
        "skewness", "excess_kurtosis",
    ],
    "total_turnover": [
        "mean", "std", "min", "max",
        "skewness", "excess_kurtosis",
    ],
}

COMPARE_AXES = ["hidden_arch", "variance_feature", "loss_name"]
REGIME_AXES = ["transaction_cost_rate", "vol_regime"]


def flatten_dict(d: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
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


def normalise_hidden_sizes(x: Any) -> str:
    if x is None:
        return "unknown"
    if isinstance(x, (list, tuple)):
        return "linear" if len(x) == 0 else "x".join(str(int(v)) for v in x)
    s = str(x).strip()
    if s in {"[]", "()", ""}:
        return "linear"
    return s


def infer_variance_feature_label(x: Any) -> str:
    if x is None:
        return "unknown"
    try:
        i = int(x)
    except Exception as e:
        raise e
    mapping = {1: "none", 2: "filter", 3: "markov", 4: "gated"}
    return mapping.get(i, f"vf_{i}")


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


def sanitize_float(x: float) -> str:
    s = f"{x:g}"
    return s.replace("-", "m").replace(".", "p")


def make_pairwise_deltas(df: pd.DataFrame, metric_name: str, compare_col: str, left: str, right: str) -> pd.DataFrame:
    cols = [c for c in COMPARE_AXES if c != compare_col and c in df.columns]
    metric_cols = [
        f"{metric_name}.mean",
        f"{metric_name}.std",
        f"{metric_name}.loss_var_alpha",
        f"{metric_name}.loss_cvar_alpha",
    ]
    metric_cols = [c for c in metric_cols if c in df.columns]
    if not metric_cols or compare_col not in df.columns:
        return pd.DataFrame()

    # With one seed, this is mostly a direct reshape; with multiple seeds it averages across runs.
    agg = df.groupby(cols + [compare_col], dropna=False)[metric_cols].mean(numeric_only=True).reset_index()
    left_df = agg[agg[compare_col] == left].copy()
    right_df = agg[agg[compare_col] == right].copy()
    if left_df.empty or right_df.empty:
        return pd.DataFrame()

    merged = left_df.merge(right_df, on=cols, suffixes=(f"__{left}", f"__{right}"), how="inner")
    for m in metric_cols:
        merged[f"delta__{m}__{left}_minus_{right}"] = merged[f"{m}__{left}"] - merged[f"{m}__{right}"]
    return merged.sort_values(cols)


def make_regime_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a matrix whose columns are a MultiIndex:
        (hidden_arch, variance_feature, loss_name)
    and whose rows are a MultiIndex:
        (metric_name, statistic)

    Since the user currently has one seed, each configuration should map to a single row.
    If multiple runs per configuration exist, the first is taken after sorting by seed/run_name.
    """
    value_cols: list[str] = []
    row_tuples: list[tuple[str, str]] = []
    for metric_name, stats in METRIC_SPECS.items():
        for stat in stats:
            col = f"{metric_name}.{stat}"
            if col in df.columns:
                value_cols.append(col)
                row_tuples.append((metric_name, stat))

    if not value_cols:
        return pd.DataFrame()

    config_cols = [c for c in COMPARE_AXES if c in df.columns]
    if len(config_cols) != 3:
        return pd.DataFrame()

    base = (
        df[config_cols + ["seed", "run_name"] + value_cols]
        .sort_values(config_cols + ["seed", "run_name"])
        .drop_duplicates(subset=config_cols, keep="first")
    )

    matrix_data: dict[tuple[str, str, str], list[float]] = {}
    for _, row in base.iterrows():
        col_key = (str(row["hidden_arch"]), str(row["variance_feature"]), str(row["loss_name"]))
        matrix_data[col_key] = [row[c] for c in value_cols]

    if not matrix_data:
        return pd.DataFrame()

    out = pd.DataFrame(matrix_data, index=pd.MultiIndex.from_tuples(row_tuples, names=["metric_name", "statistic"]))
    out.columns = pd.MultiIndex.from_tuples(list(out.columns), names=["architecture", "feature_type", "loss_type"])
    out = out.sort_index(axis=0).sort_index(axis=1)
    return out


def write_regime_report(
    out_path: Path,
    regime_df: pd.DataFrame,
    matrix: pd.DataFrame,
    deltas: dict[str, pd.DataFrame],
) -> None:
    lines: list[str] = []
    tc = regime_df["transaction_cost_rate"].iloc[0]
    vol_regime = regime_df["vol_regime"].iloc[0]
    lines.append(f"# Regime summary: transaction_cost_rate={tc}, vol_regime={vol_regime}\n\n")
    lines.append(f"- runs loaded: **{len(regime_df)}**\n")
    for c in COMPARE_AXES:
        if c in regime_df.columns:
            vals = ", ".join(sorted(regime_df[c].dropna().astype(str).unique().tolist()))
            lines.append(f"- `{c}`: {vals}\n")

    lines.append("\n## metric_matrix\n\n")
    if matrix.empty:
        lines.append("_No data._\n")
    else:
        lines.append(matrix.to_markdown())
        lines.append("\n")

    for name, table in deltas.items():
        lines.append(f"\n## {name}\n\n")
        if table.empty:
            lines.append("_No data._\n")
        else:
            lines.append(table.to_markdown(index=False))
            lines.append("\n")

    out_path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-regime summary for barrier hedging runs.")
    parser.add_argument("--root", type=Path, required=True, help="Root logs directory.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory.")
    args = parser.parse_args()

    df = load_runs(args.root)
    if df.empty:
        raise SystemExit(f"No runs found under {args.root}")

    out_root = args.out
    out_root.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_root / "runs_flat.csv", index=False)

    regime_index: list[dict[str, Any]] = []

    for (tc, vol_regime), regime_df in df.groupby(REGIME_AXES, dropna=False):
        regime_name = f"tc_{sanitize_float(float(tc))}__vol_{vol_regime}"
        regime_dir = out_root / regime_name
        regime_dir.mkdir(parents=True, exist_ok=True)

        regime_df = regime_df.copy().sort_values(COMPARE_AXES)
        regime_df.to_csv(regime_dir / "runs_flat.csv", index=False)

        matrix = make_regime_matrix(regime_df)
        if not matrix.empty:
            matrix.to_csv(regime_dir / "metric_matrix.csv")

        deltas: dict[str, pd.DataFrame] = {}
        for metric_name in METRIC_SPECS:
            for left, right in [("learned", "none"), ("markov", "learned"), ("markov", "none")]:
                name = f"delta_{metric_name}_{left}_minus_{right}"
                table = make_pairwise_deltas(regime_df, metric_name, "variance_feature", left, right)
                deltas[name] = table
                if not table.empty:
                    table.to_csv(regime_dir / f"{name}.csv", index=False)

            name = f"delta_{metric_name}_cvar_minus_mse"
            table = make_pairwise_deltas(regime_df, metric_name, "loss_name", "cvar", "mse")
            deltas[name] = table
            if not table.empty:
                table.to_csv(regime_dir / f"{name}.csv", index=False)

        write_regime_report(regime_dir / "report.md", regime_df, matrix, deltas)

        regime_index.append({
            "transaction_cost_rate": float(tc),
            "vol_regime": vol_regime,
            "regime_dir": regime_name,
            "n_runs": int(len(regime_df)),
        })

    pd.DataFrame(regime_index).sort_values(REGIME_AXES).to_csv(out_root / "regime_index.csv", index=False)


if __name__ == "__main__":
    main()
