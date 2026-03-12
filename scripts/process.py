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
ARCH_ORDER = ["16", "16x16", "16x16x16", "16x16x16x16", "32x32"]


def load_matrix(regime_dir: Path) -> pd.DataFrame:
    return pd.read_csv(
        regime_dir / "metric_matrix.csv",
        index_col=[0, 1],
        header=[0, 1, 2],
    )


def write_latex_table(df: pd.DataFrame, out_path: Path, caption: str, label: str, column_format: str | None = None) -> None:
    if column_format is None:
        column_format = "l" + "r" * (len(df.columns) - 1)
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


def main_performance_table(matrix: pd.DataFrame, arch: str = "16x16") -> pd.DataFrame:
    rows = []
    for feat in FEATURE_ORDER:
        for loss in LOSS_ORDER:
            col = (arch, feat, loss)
            if col not in matrix.columns:
                continue
            rows.append({
                "Information Regime": FEATURE_LABELS[feat],
                "Objective": LOSS_LABELS[loss],
                "Std(PnL)": matrix.loc[("pnl", "std"), col],
                "CVaR": matrix.loc[("pnl", "loss_cvar_alpha"), col],
                "Mean TC": matrix.loc[("total_transaction_cost", "mean"), col],
                "Turnover": matrix.loc[("total_turnover", "mean"), col],
            })
    return pd.DataFrame(rows)


def architecture_summary_table(matrix: pd.DataFrame) -> pd.DataFrame:
    target_pairs = [("filter", "cvar"), ("gated", "cvar"), ("markov", "cvar")]
    rows = []
    for feat, loss in target_pairs:
        for arch in ARCH_ORDER:
            col = (arch, feat, loss)
            if col not in matrix.columns:
                continue
            rows.append({
                "Architecture": arch,
                "Information Regime": FEATURE_LABELS[feat],
                "Objective": LOSS_LABELS[loss],
                "Std(PnL)": matrix.loc[("pnl", "std"), col],
                "CVaR": matrix.loc[("pnl", "loss_cvar_alpha"), col],
                "Mean TC": matrix.loc[("total_transaction_cost", "mean"), col],
                "Turnover": matrix.loc[("total_turnover", "mean"), col],
            })
    return pd.DataFrame(rows)


def regime_metadata(regime_index: pd.DataFrame) -> list[dict]:
    rows = []
    for _, r in regime_index.iterrows():
        rows.append({
            "regime_dir": str(r["regime_dir"]),
            "transaction_cost_rate": float(r["transaction_cost_rate"]) if "transaction_cost_rate" in regime_index.columns else None,
            "vol_regime": r["vol_regime"] if "vol_regime" in regime_index.columns else None,
        })
    return rows


def plot_metric_by_information(regimes: list[dict], matrices: dict[str, pd.DataFrame], metric_row: tuple[str, str], out_path: Path, ylabel: str, arch: str = "16x16") -> None:
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
                ax.bar(pos, vals, width=width, label=LOSS_LABELS[loss])
            ax.set_xticks(x)
            ax.set_xticklabels([FEATURE_LABELS[f] for f in FEATURE_ORDER], rotation=20, ha="right")
            ax.set_title(rf"$\Vol Regmie={vol_regime}$, TC={tc}")
            ax.set_ylabel(ylabel)
            ax.legend()
        fig.tight_layout()
        fig.savefig(out_path.parent / f"{out_path.stem}_tc_{str(tc).replace('.', 'p')}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def plot_architecture_robustness(regimes: list[dict], matrices: dict[str, pd.DataFrame], out_path: Path) -> None:
    if not regimes:
        return
    matrix = matrices[regimes[0]["regime_dir"]]
    target_pairs = [("filter", "cvar"), ("gated", "cvar"), ("markov", "cvar")]
    x_positions = {arch: i for i, arch in enumerate(ARCH_ORDER)}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric_row, ylabel in zip(
        axes,
        [("pnl", "loss_cvar_alpha"), ("total_turnover", "mean")],
        ["CVaR", "Mean Turnover"],
    ):
        for feat, loss in target_pairs:
            xs, ys = [], []
            for arch in ARCH_ORDER:
                col = (arch, feat, loss)
                if col in matrix.columns:
                    xs.append(x_positions[arch])
                    ys.append(matrix.loc[metric_row, col])
            if xs:
                ax.plot(xs, ys, marker="o", label=FEATURE_LABELS[feat])
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Architecture")
        ax.set_xticks(range(len(ARCH_ORDER)))
        ax.set_xticklabels(ARCH_ORDER)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate condensed LaTeX tables and bar plots from per-regime metric matrices.")
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

    include_lines = []
    for r in regimes:
        regime_name = Path(r["regime_dir"]).name
        matrix = matrices[r["regime_dir"]]
        regime_tables = tables_dir / regime_name
        regime_tables.mkdir(parents=True, exist_ok=True)

        perf = main_performance_table(matrix, arch="16x16")
        write_latex_table(
            perf,
            regime_tables / "main_performance.tex",
            caption=f"Main performance summary for {r["vol_regime"]} volatility regime and transaction cost rate $\\lambda={r['transaction_cost_rate']}$.",
            label=f"tab:{regime_name}:main-performance",
            column_format="llrrrr",
        )

        arch_df = architecture_summary_table(matrix)
        if not arch_df.empty:
            write_latex_table(
                arch_df,
                regime_tables / "architecture_summary.tex",
                caption=f"Architecture robustness summary for {r["vol_regime"]} volatility regime and transaction cost rate $\\lambda={r['transaction_cost_rate']}$.",
                label=f"tab:{regime_name}:architecture-summary",
                column_format="lllrrrr",
            )

        include_lines.append(f"% {regime_name}\n")
        include_lines.append(f"\\input{{results_tables/{regime_name}/main_performance}}\n")
        if not arch_df.empty:
            include_lines.append(f"\\input{{results_tables/{regime_name}/architecture_summary}}\n")
        include_lines.append("\n")

    (tables_dir / "include_all_tables.tex").write_text("".join(include_lines), encoding="utf-8")

    plot_metric_by_information(
        regimes, matrices, ("pnl", "std"),
        figs_dir / "pnl_std_by_information.png",
        "Std(PnL)", arch="16x16"
    )
    plot_metric_by_information(
        regimes, matrices, ("total_turnover", "mean"),
        figs_dir / "turnover_by_information.png",
        "Mean Turnover", arch="16x16"
    )
    plot_metric_by_information(
        regimes, matrices, ("total_transaction_cost", "mean"),
        figs_dir / "transaction_cost_by_information.png",
        "Mean Transaction Cost", arch="16x16"
    )
    plot_architecture_robustness(regimes, matrices, figs_dir / "architecture_robustness.png")

    fig_include = r"""
\begin{figure}[ht]
    \centering
    \includegraphics[width=0.49\textwidth]{results_figures/pnl_std_by_information_tc_0p001.png}
    \includegraphics[width=0.49\textwidth]{results_figures/pnl_std_by_information_tc_0p005.png}
    \caption{Standard deviation of terminal PnL across information regimes and objectives, shown separately by transaction cost regime and volatility regime.}
    \label{fig:pnl-std-by-information}
\end{figure}

\begin{figure}[ht]
    \centering
    \includegraphics[width=0.49\textwidth]{results_figures/turnover_by_information_tc_0p001.png}
    \includegraphics[width=0.49\textwidth]{results_figures/turnover_by_information_tc_0p005.png}
    \caption{Mean turnover across information regimes and objectives, shown separately by transaction cost regime and volatility regime.}
    \label{fig:turnover-by-information}
\end{figure}

\begin{figure}[ht]
    \centering
    \includegraphics[width=0.49\textwidth]{results_figures/transaction_cost_by_information_tc_0p001.png}
    \includegraphics[width=0.49\textwidth]{results_figures/transaction_cost_by_information_tc_0p005.png}
    \caption{Mean transaction cost across information regimes and objectives, shown separately by transaction cost regime and volatility regime.}
    \label{fig:tc-by-information}
\end{figure}

\begin{figure}[ht]
    \centering
    \includegraphics[width=0.7\textwidth]{results_figures/architecture_robustness.png}
    \caption{Architecture robustness for the available ablation runs.}
    \label{fig:architecture-robustness}
\end{figure}
"""
    (figs_dir / "include_all_figures.tex").write_text(fig_include.strip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
