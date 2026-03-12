from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib
import pandas as pd
from scipy.stats import kendalltau, spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def add_metric_vote_scores(
    results_df: pd.DataFrame,
    vote_metrics: Sequence[str],
    lower_is_better: Sequence[str] = ("jacob_cov", "grasp"),
) -> pd.DataFrame:
    df = results_df.copy()
    n_models = len(df)
    vote_score = pd.Series(0.0, index=df.index)
    lower_set = set(lower_is_better)

    for metric in vote_metrics:
        ascending = metric in lower_set
        rank_col = f"rank_{metric}"
        points_col = f"points_{metric}"
        df[rank_col] = df[metric].rank(method="average", ascending=ascending)
        df[points_col] = n_models - df[rank_col] + 1
        vote_score += df[points_col]

    df["vote_score"] = vote_score
    df["vote_rank"] = df["vote_score"].rank(method="min", ascending=False).astype(int)
    return df


def build_metric_mapping_table(
    results_df: pd.DataFrame,
    metrics: Sequence[str],
    targets: Sequence[str],
    group_column: str | None = None,
) -> pd.DataFrame:
    rows: List[Dict[str, float | str]] = []
    grouped = [("overall", results_df)]
    if group_column is not None and group_column in results_df.columns:
        grouped = list(results_df.groupby(group_column, sort=True))

    for group_name, group_df in grouped:
        for target in targets:
            for metric in metrics:
                rho, rho_p = spearmanr(group_df[metric], group_df[target])
                tau, tau_p = kendalltau(group_df[metric], group_df[target])
                rows.append(
                    {
                        "group": group_name,
                        "metric": metric,
                        "target": target,
                        "spearman_rho": rho,
                        "spearman_p": rho_p,
                        "kendall_tau": tau,
                        "kendall_p": tau_p,
                    }
                )
    return pd.DataFrame(rows)


def plot_metric_correlations(
    results_df: pd.DataFrame,
    out_dir: Path,
    metrics: Sequence[str],
    target: str,
    prefix: str,
) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    output_paths: Dict[str, Path] = {}
    has_family = "family" in results_df.columns
    family_colors = {"plain_cnn": "#1f77b4", "residual_cnn": "#d62728"}

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(6, 4))
        if has_family:
            for fam_name, fam_df in results_df.groupby("family"):
                ax.scatter(
                    fam_df[metric],
                    fam_df[target],
                    alpha=0.8,
                    label=fam_name,
                    color=family_colors.get(str(fam_name), None),
                )
            ax.legend(frameon=False)
        else:
            ax.scatter(results_df[metric], results_df[target], alpha=0.8)
        ax.set_xlabel(metric)
        ax.set_ylabel(target)
        ax.set_title(f"{metric} vs {target}")
        fig.tight_layout()

        path = out_dir / f"{prefix}_{metric}_vs_{target}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        output_paths[metric] = path

    return output_paths


def plot_metric_pair_correlation(
    results_df: pd.DataFrame,
    out_dir: Path,
    x_metric: str,
    y_metric: str,
    prefix: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    has_family = "family" in results_df.columns
    family_colors = {"plain_cnn": "#1f77b4", "residual_cnn": "#d62728"}

    pair_df = results_df[[x_metric, y_metric]].dropna()
    if len(pair_df) >= 2:
        spearman_rho, spearman_p = spearmanr(pair_df[x_metric], pair_df[y_metric])
        pearson_r = pair_df[x_metric].corr(pair_df[y_metric], method="pearson")
    else:
        spearman_rho, spearman_p, pearson_r = float("nan"), float("nan"), float("nan")

    fig, ax = plt.subplots(figsize=(6, 4))
    if has_family:
        for fam_name, fam_df in results_df.groupby("family"):
            ax.scatter(
                fam_df[x_metric],
                fam_df[y_metric],
                alpha=0.8,
                label=fam_name,
                color=family_colors.get(str(fam_name), None),
            )
        ax.legend(frameon=False)
    else:
        ax.scatter(results_df[x_metric], results_df[y_metric], alpha=0.8)

    ax.set_xlabel(x_metric)
    ax.set_ylabel(y_metric)
    if pd.notna(spearman_rho) and pd.notna(pearson_r):
        ax.set_title(
            f"{x_metric} vs {y_metric}\n"
            f"Spearman rho={spearman_rho:.3f} (p={spearman_p:.2g}), Pearson r={pearson_r:.3f}"
        )
    else:
        ax.set_title(f"{x_metric} vs {y_metric}")
    fig.tight_layout()

    path = out_dir / f"{prefix}_{x_metric}_vs_{y_metric}_correlation.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _group_frames(results_df: pd.DataFrame, group_column: str | None) -> List[tuple[str, pd.DataFrame]]:
    grouped: List[tuple[str, pd.DataFrame]] = [("overall", results_df)]
    if group_column is not None and group_column in results_df.columns:
        grouped = [(str(name), group.copy()) for name, group in results_df.groupby(group_column, sort=True)]
    return grouped


def _safe_suffix(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in name)


def plot_metric_correlation_heatmaps(
    results_df: pd.DataFrame,
    out_dir: Path,
    metrics: Sequence[str],
    targets: Sequence[str],
    prefix: str,
    group_column: str | None = None,
) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    output_paths: Dict[str, Path] = {}

    metrics_list = list(metrics)
    targets_list = list(targets)
    for group_name, group_df in _group_frames(results_df, group_column):
        corr_matrix = group_df[metrics_list + targets_list].corr(method="spearman")
        corr_view = corr_matrix.loc[metrics_list, targets_list]

        fig_width = max(6.0, 1.2 * len(targets_list) + 2.0)
        fig_height = max(4.5, 0.45 * len(metrics_list) + 2.0)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        image = ax.imshow(corr_view.to_numpy(dtype=float), cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(targets_list)))
        ax.set_xticklabels(targets_list, rotation=35, ha="right")
        ax.set_yticks(range(len(metrics_list)))
        ax.set_yticklabels(metrics_list)
        ax.set_title(f"Spearman correlation heatmap ({group_name})")

        for row_idx, metric in enumerate(metrics_list):
            for col_idx, target in enumerate(targets_list):
                value = corr_view.loc[metric, target]
                if pd.isna(value):
                    text = "nan"
                    color = "black"
                else:
                    text = f"{value:.2f}"
                    color = "white" if abs(value) >= 0.5 else "black"
                ax.text(col_idx, row_idx, text, ha="center", va="center", fontsize=8, color=color)

        fig.colorbar(image, ax=ax, label="Spearman rho")
        fig.tight_layout()

        path = out_dir / f"{prefix}_corr_heatmap_{_safe_suffix(group_name)}.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        output_paths[group_name] = path

    return output_paths


def plot_rank_agreement_bump_chart(
    results_df: pd.DataFrame,
    out_dir: Path,
    metrics: Sequence[str],
    target: str,
    prefix: str,
    lower_is_better: Sequence[str] = ("jacob_cov", "grasp"),
    group_column: str | None = None,
) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    output_paths: Dict[str, Path] = {}
    lower_set = set(lower_is_better)
    metric_list = list(metrics)
    rank_columns = [target] + metric_list
    family_colors = {"plain_cnn": "#1f77b4", "residual_cnn": "#d62728"}

    for group_name, group_df in _group_frames(results_df, group_column):
        rank_df = pd.DataFrame(index=group_df.index)
        rank_df[target] = group_df[target].rank(method="min", ascending=False)
        for metric in metric_list:
            rank_df[metric] = group_df[metric].rank(method="min", ascending=(metric in lower_set))

        x_positions = list(range(len(rank_columns)))
        fig_width = max(7.5, 1.25 * len(rank_columns))
        fig_height = max(5.0, 0.28 * len(group_df) + 3.5)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        for row_idx in group_df.index:
            family = str(group_df.at[row_idx, "family"]) if "family" in group_df.columns else "overall"
            color = family_colors.get(family, "#7f7f7f")
            ax.plot(
                x_positions,
                rank_df.loc[row_idx, rank_columns].to_numpy(dtype=float),
                color=color,
                alpha=0.45,
                linewidth=1.1,
            )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(rank_columns, rotation=30, ha="right")
        ax.set_ylabel("Rank (1 = best)")
        ax.set_xlabel("Ranking source")
        ax.set_title(f"Rank agreement bump chart vs {target} ({group_name})")
        ax.set_ylim(len(group_df) + 0.5, 0.5)
        ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)

        if "family" in group_df.columns and group_df["family"].nunique() > 1:
            legend_handles = [
                Line2D([0], [0], color=color, lw=2, label=name)
                for name, color in family_colors.items()
                if name in set(group_df["family"].astype(str))
            ]
            if legend_handles:
                ax.legend(handles=legend_handles, frameon=False, loc="upper right")

        fig.tight_layout()
        path = out_dir / f"{prefix}_rank_agreement_{_safe_suffix(group_name)}_{target}.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        output_paths[group_name] = path

    return output_paths
