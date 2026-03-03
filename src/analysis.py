from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib
import pandas as pd
from scipy.stats import kendalltau, spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def add_metric_vote_scores(
    results_df: pd.DataFrame,
    vote_metrics: Sequence[str],
    lower_is_better: Sequence[str] = ("jacob_cov",),
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
