#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


FEATURE_LABELS = {
    "none": "None",
    "filter": "Learned Filter",
    "gated": "Gated Filter",
    "markov": "Oracle Variance",
}

LOSS_LABELS = {
    "mse": "MSE",
    "cvar": "CVaR",
}

FEATURE_ORDER = ["none", "filter", "gated", "markov"]
LOSS_ORDER = ["mse", "cvar"]
ARCH_ORDER = ["8x6", "16x16", "32x32", "64x64"]
MAIN_ARCH = "64x64"


def load_matrix(regime_dir: Path) -> pd.DataFrame:
    return pd.read_csv(
        regime_dir / "metric_matrix.csv",
        index_col=[0, 1],
        header=[0, 1, 2],
    )


def write_latex_table(
    df: pd.DataFrame,
    out_path: Path,
    caption: str,
    label: str,
    column_format: str | None = None,
) -> None:
    if column_format is None:
        column_format = "l" * min(2, len(df.columns)) + "r" * max(0, len(df.columns) - 2)

    tex = df.to_latex(
        index=False,
        escape=False,
        float_format="%.3f".__mod__,
        caption=caption,
        label=label,
        column_format=column_format,
        bold_rows=False,
    )
    out_path.write_text(tex, encoding="utf-8")

def write_main_comparison_table(
    df: pd.DataFrame,
    out_path: Path,
    caption: str,
    label: str,
) -> None:
    vol_order = ["Normal", "Stressed"]
    obj_order = ["MSE", "CVaR"]
    info_order = ["None", "Learned Filter", "Gated Filter", "Oracle Variance"]

    df = df.copy()
    df["Volatility Regime"] = pd.Categorical(df["Volatility Regime"], categories=vol_order, ordered=True)
    df["Objective"] = pd.Categorical(df["Objective"], categories=obj_order, ordered=True)
    df["Information Regime"] = pd.Categorical(df["Information Regime"], categories=info_order, ordered=True)
    df = df.sort_values(["Volatility Regime", "Objective", "Information Regime"]).reset_index(drop=True)

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\caption{" + caption + "}")
    lines.append(r"\label{" + label + "}")
    lines.append(r"\begin{tabular}{lllrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Volatility Regime & Information Regime & Objective & Std(PnL) & CVaR & Mean TC & Turnover \\")
    lines.append(r"\midrule")

    prev_block = None
    for _, row in df.iterrows():
        current_block = (row["Volatility Regime"], row["Objective"])

        if prev_block is not None and current_block != prev_block:
            lines.append(r"\midrule")

        vol_text = str(row["Volatility Regime"])
        obj_text = str(row["Objective"])

        # Blank repeated entries within each (vol, objective) block
        if prev_block == current_block:
            vol_text = ""
            obj_text = ""

        lines.append(
            f"{vol_text} & "
            f"{row['Information Regime']} & "
            f"{obj_text} & "
            f"{row['Std(PnL)']:.3f} & "
            f"{row['CVaR']:.3f} & "
            f"{row['Mean TC']:.3f} & "
            f"{row['Turnover']:.3f} \\\\"
        )

        prev_block = current_block

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_trade_behaviour_table(
    df: pd.DataFrame,
    out_path: Path,
    caption: str,
    label: str,
) -> None:
    vol_order = ["Normal", "Stressed"]
    obj_order = ["MSE", "CVaR"]
    info_order = ["None", "Learned Filter", "Gated Filter", "Oracle Variance"]

    df = df.copy()
    df["Volatility Regime"] = pd.Categorical(df["Volatility Regime"], categories=vol_order, ordered=True)
    df["Objective"] = pd.Categorical(df["Objective"], categories=obj_order, ordered=True)
    df["Information Regime"] = pd.Categorical(df["Information Regime"], categories=info_order, ordered=True)
    df = df.sort_values(["Volatility Regime", "Objective", "Information Regime"]).reset_index(drop=True)

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\caption{" + caption + "}")
    lines.append(r"\label{" + label + "}")
    lines.append(r"\begin{tabular}{lllrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Volatility Regime & Information Regime & Objective & Mean TC & Turnover & $\Delta$ TC vs None (\%) & $\Delta$ Turnover vs None (\%) \\")
    lines.append(r"\midrule")

    prev_block = None
    for _, row in df.iterrows():
        current_block = (row["Volatility Regime"], row["Objective"])

        if prev_block is not None and current_block != prev_block:
            lines.append(r"\midrule")

        vol_text = str(row["Volatility Regime"])
        obj_text = str(row["Objective"])

        # Blank repeated entries within each (vol, objective) block
        if prev_block == current_block:
            vol_text = ""
            obj_text = ""

        lines.append(
            f"{vol_text} & "
            f"{row['Information Regime']} & "
            f"{obj_text} & "
            f"{row['Mean TC']:.3f} & "
            f"{row['Turnover']:.3f} & "
            f"{row[r"$\Delta$ TC vs None (\%)"]} & "
            f"{row[ r"$\Delta$ Turnover vs None (\%)"]} \\\\"
        )

        prev_block = current_block

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def regime_metadata(regime_index: pd.DataFrame) -> list[dict]:
    rows = []
    for _, r in regime_index.iterrows():
        rows.append({
            "regime_dir": str(r["regime_dir"]),
            "transaction_cost_rate": float(r["transaction_cost_rate"]) if "transaction_cost_rate" in regime_index.columns else None,
            "vol_regime": r["vol_regime"] if "vol_regime" in regime_index.columns else None,
        })
    return rows


def main_comparison_table(
    regimes: list[dict],
    matrices: dict[str, pd.DataFrame],
    arch: str = MAIN_ARCH,
) -> pd.DataFrame:
    rows = []
    for r in regimes:
        matrix = matrices[r["regime_dir"]]
        for feat in FEATURE_ORDER:
            for loss in LOSS_ORDER:
                col = (arch, feat, loss)
                if col not in matrix.columns:
                    continue
                rows.append({
                    "Volatility Regime": r["vol_regime"].capitalize(),
                    "Information Regime": FEATURE_LABELS[feat],
                    "Objective": LOSS_LABELS[loss],
                    "Std(PnL)": matrix.loc[("pnl", "std"), col],
                    "CVaR": matrix.loc[("pnl", "loss_cvar_alpha"), col],
                    "Mean TC": matrix.loc[("total_transaction_cost", "mean"), col],
                    "Turnover": matrix.loc[("total_turnover", "mean"), col],
                })
    return pd.DataFrame(rows)


def trading_behaviour_table(
    regimes: list[dict],
    matrices: dict[str, pd.DataFrame],
    arch: str = MAIN_ARCH,
) -> pd.DataFrame:
    rows = []
    for r in regimes:
        matrix = matrices[r["regime_dir"]]

        baseline = {}
        for loss in LOSS_ORDER:
            none_col = (arch, "none", loss)
            if none_col in matrix.columns:
                baseline[loss] = {
                    "tc": matrix.loc[("total_transaction_cost", "mean"), none_col],
                    "turnover": matrix.loc[("total_turnover", "mean"), none_col],
                }

        for feat in FEATURE_ORDER:
            for loss in LOSS_ORDER:
                col = (arch, feat, loss)
                if col not in matrix.columns or loss not in baseline:
                    continue

                mean_tc = matrix.loc[("total_transaction_cost", "mean"), col]
                turnover = matrix.loc[("total_turnover", "mean"), col]
                base_tc = baseline[loss]["tc"]
                base_turnover = baseline[loss]["turnover"]

                delta_tc_pct = 100.0 * (mean_tc - base_tc) / base_tc if base_tc != 0 else float("nan")
                delta_turnover_pct = 100.0 * (turnover - base_turnover) / base_turnover if base_turnover != 0 else float("nan")

                rows.append({
                    "Volatility Regime": r["vol_regime"].capitalize(),
                    "Information Regime": FEATURE_LABELS[feat],
                    "Objective": LOSS_LABELS[loss],
                    "Mean TC": mean_tc,
                    "Turnover": turnover,
                    r"$\Delta$ TC vs None (\%)": delta_tc_pct,
                    r"$\Delta$ Turnover vs None (\%)": delta_turnover_pct,
                })

    df = pd.DataFrame(rows)

    # Optional: make the percentage columns visually cleaner
    for col in [r"$\Delta$ TC vs None (\%)", r"$\Delta$ Turnover vs None (\%)"]:
        df[col] = df[col].map(lambda x: f"{x:+.1f}" if pd.notna(x) else "")

    return df


def architecture_summary_table(matrix: pd.DataFrame, objective_function: str) -> pd.DataFrame:
    target_pairs = [("filter", objective_function), ("gated", objective_function), ("markov", objective_function)]
    rows = []
    for feat, loss in target_pairs:
        for arch in ARCH_ORDER:
            n_hidden = len(arch.split("x"))
            col = (arch, feat, loss)
            if col not in matrix.columns:
                continue
            rows.append({
                "Hidden Layers": str(n_hidden),
                "Information Regime": FEATURE_LABELS[feat],
                "Objective": LOSS_LABELS[loss],
                "Std(PnL)": matrix.loc[("pnl", "std"), col],
                "CVaR": matrix.loc[("pnl", "loss_cvar_alpha"), col],
                "Mean TC": matrix.loc[("total_transaction_cost", "mean"), col],
                "Turnover": matrix.loc[("total_turnover", "mean"), col],
            })
    return pd.DataFrame(rows)


def plot_metric_by_information(
    regimes: list[dict],
    matrices: dict[str, pd.DataFrame],
    metric_row: tuple[str, str],
    out_path: Path,
    ylabel: str,
    arch: str = MAIN_ARCH,
) -> None:
    tc_values = sorted({r["transaction_cost_rate"] for r in regimes})
    vol_regime_values = sorted({r["vol_regime"] for r in regimes})
    for tc in tc_values:
        fig, axes = plt.subplots(1, len(vol_regime_values), figsize=(6 * len(vol_regime_values), 4), squeeze=False)
        axes = axes[0]
        for ax, vol_regime in zip(axes, vol_regime_values):
            regime = next((r for r in regimes if r["transaction_cost_rate"] == tc and r["vol_regime"] == vol_regime), None)
            if regime is None:
                ax.set_visible(False)
                continue
            matrix = matrices[regime["regime_dir"]]
            x = list(range(len(FEATURE_ORDER)))
            width = 0.35
            for j, loss in enumerate(LOSS_ORDER):
                vals = []
                for feat in FEATURE_ORDER:
                    col = (arch, feat, loss)
                    vals.append(matrix.loc[metric_row, col] if col in matrix.columns else float("nan"))
                pos = [i + (j - 0.5) * width for i in x]
                ax.bar(pos, vals, width=width, label=LOSS_LABELS[loss] + " Loss")
            ax.set_xticks(x)
            ax.set_xticklabels([FEATURE_LABELS[f] for f in FEATURE_ORDER], rotation=20, ha="right")
            ax.set_title(rf"Volatility Regime={vol_regime}, TC={tc}")
            ax.set_ylabel(ylabel)
            ax.legend()
        fig.tight_layout()
        fig.savefig(out_path.parent / f"{out_path.stem}_tc_{str(tc).replace('.', 'p')}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def plot_architecture_robustness(regimes: list[dict], matrices: dict[str, pd.DataFrame], figs_dir: Path) -> None:
    if not regimes:
        return
    for regime in regimes:
        # matrix = matrices[regimes[0]["regime_dir"]]
        matrix = matrices[regime["regime_dir"]]
        target_pairs_cvar = [("none", "cvar"), ("filter", "cvar"), ("gated", "cvar"), ("markov", "cvar")]
        target_pairs_mse = [("none", "mse"), ("filter", "mse"), ("gated", "mse"), ("markov", "mse")]
        hidden_size = int(ARCH_ORDER[0].split("x")[0])
        for arch in ARCH_ORDER:
            assert all(int(x) == hidden_size for x in arch.split("x")), "hidden size must match across architectures."
        n_hidden = {arch: len(arch.split("x")) for arch in ARCH_ORDER}
        cmap = plt.get_cmap("tab10")
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, metric_row, ylabel in zip(
            axes,
            [("pnl", "loss_cvar_alpha"), ("pnl", "std"), ("total_turnover", "mean")],
            ["CVaR", "Standard Deviation", "Mean Turnover"],
        ):
            
            for k, (feat, loss) in enumerate(target_pairs_cvar):
                c = cmap(k)
                xs, ys = [], []
                for arch in ARCH_ORDER:
                    col = (arch, feat, loss)
                    if col in matrix.columns:
                        xs.append(n_hidden[arch])
                        ys.append(matrix.loc[metric_row, col])
                if xs:
                    ax.plot(xs, ys, marker="o", label=FEATURE_LABELS[feat], c=c)
            for k, (feat, loss) in enumerate(target_pairs_mse):
                c = cmap(k)
                xs, ys = [], []
                for arch in ARCH_ORDER:
                    col = (arch, feat, loss)
                    if col in matrix.columns:
                        xs.append(n_hidden[arch])
                        ys.append(matrix.loc[metric_row, col])
                if xs:
                    ax.plot(xs, ys, marker="o", c=c, linestyle="--")
            ax.set_ylabel(ylabel)
            ax.set_xlabel(f"Number of hidden layers (hidden size={hidden_size})")
            ax.set_xticks(list(n_hidden.values()))
            ax.legend()
        fig.tight_layout()
        fig.savefig(figs_dir / f"{Path(regime["regime_dir"]).name}_architecture_robustness.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate condensed LaTeX tables and plots from per-regime metric matrices.")
    parser.add_argument("--analysis-root", type=Path, required=True, help="Root analysis directory containing regime_index.csv and regime subdirectories.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for tables and plots.")
    args = parser.parse_args()

    analysis_root = args.analysis_root
    out_root = args.out
    out_root.mkdir(parents=True, exist_ok=True)
    tables_dir = out_root / "results_tables"
    figs_dir = out_root / "results_figures"
    tables_dir.mkdir(exist_ok=True)
    figs_dir.mkdir(exist_ok=True)

    regime_index = pd.read_csv(analysis_root / "regime_index.csv")
    regimes = regime_metadata(regime_index)

    matrices = {}
    for r in regimes:
        matrices[r["regime_dir"]] = load_matrix(analysis_root / r["regime_dir"])

    # Main summary table
    main_df = main_comparison_table(regimes, matrices, arch=MAIN_ARCH)
    write_main_comparison_table(
        main_df,
        tables_dir / "main_comparison.tex",
        caption=(
            "Main comparison across information regimes, objectives, and volatility regimes. "
            f"Policy uses {len(MAIN_ARCH.split("x"))} hidden layers of size {MAIN_ARCH.split("x")[0]}."
        ),
        label="tab:main-comparison",
    )

    # Trading behaviour table
    trading_df = trading_behaviour_table(regimes, matrices, arch=MAIN_ARCH)
    write_trade_behaviour_table(
        trading_df,
        tables_dir / "trading_behaviour.tex",
        caption=(
            "Trading behaviour across information regimes, objectives, and volatility regimes "
            f"Policy uses {len(MAIN_ARCH.split("x"))} hidden layers of size {MAIN_ARCH.split("x")[0]}. "
            "Percentage changes are reported relative to the "
            "corresponding no-volatility baseline within each volatility regime and objective."
        ),
        label="tab:trading-behaviour",
        # column_format="lllrrll",
    )

    # Optional: keep one small architecture table if desired
    for r in regimes:
        regime_name = Path(r["regime_dir"]).name
        matrix = matrices[r["regime_dir"]]
        arch_df = architecture_summary_table(matrix, "cvar")
        if not arch_df.empty:
            write_latex_table(
                arch_df,
                tables_dir / f"{regime_name}_architecture_summary_cvar.tex",
                caption=(
                    f"Architecture robustness under the CVaR objective for the {r['vol_regime']} "
                    f"volatility regime and transaction cost rate $\\lambda={r['transaction_cost_rate']}$."
                ),
                label=f"tab:{regime_name}:architecture-summary-cvar",
                column_format="lllrrrr",
            )

    include_text = (
        "\\input{results_tables/main_comparison}\n"
        "\\input{results_tables/trading_behaviour}\n"
    )
    (tables_dir / "include_all_tables.tex").write_text(include_text, encoding="utf-8")

    plot_metric_by_information(
        regimes, matrices, ("pnl", "std"),
        figs_dir / "pnl_std_by_information.png",
        "Std(PnL)", arch=MAIN_ARCH
    )
    plot_metric_by_information(
        regimes, matrices, ("pnl", "loss_cvar_alpha"),
        figs_dir / "pnl_cvar_by_information.png",
        "CVaR(-PnL)", arch=MAIN_ARCH
    )
    plot_metric_by_information(
        regimes, matrices, ("total_turnover", "mean"),
        figs_dir / "turnover_by_information.png",
        "Mean Turnover", arch=MAIN_ARCH
    )
    plot_metric_by_information(
        regimes, matrices, ("total_transaction_cost", "mean"),
        figs_dir / "transaction_cost_by_information.png",
        "Mean Transaction Cost", arch=MAIN_ARCH
    )
    plot_architecture_robustness(regimes, matrices, figs_dir)


if __name__ == "__main__":
    main()