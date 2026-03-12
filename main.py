from __future__ import annotations

import argparse
import gzip
import hashlib
from pathlib import Path
import shutil
import urllib.request

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.analysis import (
    add_metric_vote_scores,
    build_metric_mapping_table,
    plot_metric_pair_correlation,
    plot_metric_correlation_heatmaps,
    plot_metric_correlations,
    plot_rank_agreement_bump_chart,
)
from src.metrics import compute_all_zero_cost_metrics
from src.models import get_model_suite
from src.train import train_short


ZERO_COST_METRICS = ["snip", "grasp", "synflow", "fisher", "jacob_cov", "grad_norm", "naswot"]
SMALL_VOTE_METRICS = ("synflow", "naswot")
SMALL_VOTE_SCORE_COL = "vote_score_synflow_naswot"
SMALL_VOTE_RANK_COL = "vote_rank_synflow_naswot"
CORR_TARGETS = ["val_acc", "val_f1_macro", "val_mse", "train_time_sec", "infer_ms_per_sample"]
DEFAULT_LOWER_IS_BETTER = ("jacob_cov", "grasp")
PROJECT_DIR = Path(__file__).resolve().parent


MNIST_MIRRORS = (
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
)
MNIST_RESOURCES = (
    ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
    ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
    ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
    ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c"),
)


def _resolve_default_data_dir() -> Path:
    candidates = [
        PROJECT_DIR / "data",
        PROJECT_DIR.parent.parent / "DistUNNce" / "data",
        PROJECT_DIR.parent / "DistUNNce" / "data",
        PROJECT_DIR.parent / "data",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DEFAULT_DATA_DIR = _resolve_default_data_dir()
DEFAULT_OUT_DIR = PROJECT_DIR / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Untrained CNN benchmark on MNIST (plain vs residual).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--metric-batches", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-models", type=int, default=24)
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--include-family-split", action="store_true")
    parser.add_argument(
        "--replot-only",
        action="store_true",
        help="Skip metric/training run and regenerate CSV mappings/plots from an existing results CSV.",
    )
    parser.add_argument(
        "--results-csv",
        type=str,
        default=None,
        help="Input results CSV for --replot-only. Defaults to <out-dir>/results_mnist.csv.",
    )
    parser.add_argument(
        "--lower-is-better",
        type=str,
        default="jacob_cov,grasp",
        help="Comma-separated metrics that should be ranked ascending for vote/rank plots.",
    )
    parser.add_argument(
        "--log-batches",
        type=int,
        default=50,
        help="Log every N training batches (0 disables batch-level logging).",
    )
    return parser.parse_args()


def get_device(require_cuda: bool = False) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if require_cuda:
        raise RuntimeError("CUDA wurde angefordert (--require-cuda), aber keine CUDA-GPU gefunden.")
    return torch.device("cpu")


def _file_md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _download_mnist_resource(raw_dir: Path, filename: str, expected_md5: str) -> None:
    target_path = raw_dir / filename
    if target_path.exists() and _file_md5(target_path) == expected_md5:
        return
    if target_path.exists():
        target_path.unlink()

    errors: list[str] = []
    for mirror in MNIST_MIRRORS:
        url = f"{mirror}{filename}"
        try:
            urllib.request.urlretrieve(url, target_path)
            if _file_md5(target_path) == expected_md5:
                return
            errors.append(f"{url} -> md5 mismatch")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{url} -> {exc}")
        if target_path.exists():
            target_path.unlink()

    raise RuntimeError(f"MNIST resource download failed for {filename}: {'; '.join(errors)}")


def _extract_gzip_if_needed(raw_dir: Path, filename: str) -> None:
    gz_path = raw_dir / filename
    out_path = raw_dir / Path(filename).stem
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    with gzip.open(gz_path, "rb") as src, out_path.open("wb") as dst:
        shutil.copyfileobj(src, dst)


def _prepare_mnist_raw_files(data_dir: str) -> None:
    raw_dir = Path(data_dir) / "MNIST" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for filename, md5 in MNIST_RESOURCES:
        _download_mnist_resource(raw_dir=raw_dir, filename=filename, expected_md5=md5)
        _extract_gzip_if_needed(raw_dir=raw_dir, filename=filename)


def get_mnist_loaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:
    from torchvision import datasets, transforms

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    try:
        train_ds = datasets.MNIST(root=data_dir, train=True, transform=transform, download=True)
        val_ds = datasets.MNIST(root=data_dir, train=False, transform=transform, download=True)
    except RuntimeError:
        print("Standard-MNIST-Download fehlgeschlagen, starte Mirror-Fallback (S3/Google).", flush=True)
        try:
            _prepare_mnist_raw_files(data_dir=data_dir)
            train_ds = datasets.MNIST(root=data_dir, train=True, transform=transform, download=False)
            val_ds = datasets.MNIST(root=data_dir, train=False, transform=transform, download=False)
        except Exception as fallback_exc:  # noqa: BLE001
            raise RuntimeError(
                "MNIST Download fehlgeschlagen. Loesche ggf. defekte Dateien in "
                f"'{Path(data_dir) / 'MNIST' / 'raw'}' und starte erneut."
            ) from fallback_exc

    use_cuda = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    return train_loader, val_loader


def average_zero_cost_scores(
    model: torch.nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    metric_batches: int,
) -> dict[str, float]:
    sums = {k: 0.0 for k in ZERO_COST_METRICS}
    used = 0

    total_batches = len(train_loader)
    for images, labels in train_loader:
        scores = compute_all_zero_cost_metrics(model=model, images=images, labels=labels, device=device)
        for metric in ZERO_COST_METRICS:
            sums[metric] += float(scores[metric])
        used += 1
        print(f"   zero-cost batch {used}/{metric_batches} (loader has {total_batches} batches)", flush=True)
        if used >= metric_batches:
            break

    return {k: v / max(used, 1) for k, v in sums.items()}


def _parse_lower_is_better(raw: str) -> tuple[str, ...]:
    metrics = tuple(metric.strip() for metric in raw.split(",") if metric.strip())
    if not metrics:
        return DEFAULT_LOWER_IS_BETTER
    unknown = sorted(set(metrics) - set(ZERO_COST_METRICS))
    if unknown:
        raise ValueError(f"Unknown metrics in --lower-is-better: {', '.join(unknown)}")
    return metrics


def _postprocess_and_plot(
    results_df: pd.DataFrame,
    out_dir: Path,
    include_family_split: bool,
    lower_is_better: tuple[str, ...],
) -> tuple[pd.DataFrame, Path, Path, Path | None]:
    required_cols = set(ZERO_COST_METRICS + CORR_TARGETS)
    missing = sorted(required_cols - set(results_df.columns))
    if missing:
        raise RuntimeError(f"results CSV is missing required columns: {', '.join(missing)}")

    results_df = add_metric_vote_scores(
        results_df=results_df,
        vote_metrics=ZERO_COST_METRICS,
        lower_is_better=lower_is_better,
    )
    small_vote_df = add_metric_vote_scores(
        results_df=results_df,
        vote_metrics=SMALL_VOTE_METRICS,
        lower_is_better=lower_is_better,
    )
    results_df[SMALL_VOTE_SCORE_COL] = small_vote_df["vote_score"]
    results_df[SMALL_VOTE_RANK_COL] = small_vote_df["vote_rank"]

    metrics_for_corr = ZERO_COST_METRICS + ["vote_score", SMALL_VOTE_SCORE_COL]

    results_path = out_dir / "results_mnist.csv"
    results_df.to_csv(results_path, index=False)

    mapping_overall_df = build_metric_mapping_table(
        results_df=results_df,
        metrics=metrics_for_corr,
        targets=CORR_TARGETS,
        group_column=None,
    )
    mapping_overall_path = out_dir / "metric_mapping_overall.csv"
    mapping_overall_df.to_csv(mapping_overall_path, index=False)

    family_mapping_path = None
    if include_family_split and "family" in results_df.columns:
        mapping_family_df = build_metric_mapping_table(
            results_df=results_df,
            metrics=metrics_for_corr,
            targets=CORR_TARGETS,
            group_column="family",
        )
        family_mapping_path = out_dir / "metric_mapping_by_family.csv"
        mapping_family_df.to_csv(family_mapping_path, index=False)

    for target in ("val_acc", "val_f1_macro"):
        _ = plot_metric_correlations(
            results_df=results_df,
            out_dir=out_dir,
            metrics=metrics_for_corr,
            target=target,
            prefix="overall",
        )
    _ = plot_metric_pair_correlation(
        results_df=results_df,
        out_dir=out_dir,
        x_metric="synflow",
        y_metric="naswot",
        prefix="overall",
    )
    _ = plot_metric_correlation_heatmaps(
        results_df=results_df,
        out_dir=out_dir,
        metrics=metrics_for_corr,
        targets=CORR_TARGETS,
        prefix="overall",
        group_column=None,
    )
    _ = plot_rank_agreement_bump_chart(
        results_df=results_df,
        out_dir=out_dir,
        metrics=metrics_for_corr,
        target="val_acc",
        prefix="overall",
        lower_is_better=lower_is_better,
    )
    if include_family_split and "family" in results_df.columns:
        _ = plot_metric_correlation_heatmaps(
            results_df=results_df,
            out_dir=out_dir,
            metrics=metrics_for_corr,
            targets=CORR_TARGETS,
            prefix="by_family",
            group_column="family",
        )
        _ = plot_rank_agreement_bump_chart(
            results_df=results_df,
            out_dir=out_dir,
            metrics=metrics_for_corr,
            target="val_acc",
            prefix="by_family",
            lower_is_better=lower_is_better,
            group_column="family",
        )

    return results_df, results_path, mapping_overall_path, family_mapping_path


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    lower_is_better = _parse_lower_is_better(args.lower_is_better)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir.resolve()}", flush=True)

    if args.replot_only:
        source_results_path = Path(args.results_csv) if args.results_csv is not None else out_dir / "results_mnist.csv"
        if not source_results_path.exists():
            raise FileNotFoundError(
                f"--replot-only requested, but no results CSV found at: {source_results_path.resolve()}"
            )
        print(f"Replot-only mode: loading existing results from {source_results_path.resolve()}", flush=True)
        results_df = pd.read_csv(source_results_path)
        (
            results_df,
            results_path,
            mapping_overall_path,
            family_mapping_path,
        ) = _postprocess_and_plot(
            results_df=results_df,
            out_dir=out_dir,
            include_family_split=args.include_family_split,
            lower_is_better=lower_is_better,
        )
        total_models = len(results_df)
        top_acc = results_df.sort_values("val_acc", ascending=False).iloc[0]
        top_vote = results_df.sort_values("vote_score", ascending=False).iloc[0]
        top_small_vote = results_df.sort_values(SMALL_VOTE_SCORE_COL, ascending=False).iloc[0]
        print("\n=== Replot finished (no training run) ===", flush=True)
        print(f"Models in CSV: {total_models}", flush=True)
        print(f"Results: {results_path.resolve()}", flush=True)
        print(f"Overall mapping: {mapping_overall_path.resolve()}", flush=True)
        if family_mapping_path is not None:
            print(f"Family mapping: {family_mapping_path.resolve()}", flush=True)
        print(f"Top by val_acc: {top_acc['model']} ({top_acc['val_acc']:.4f})", flush=True)
        print(f"Top by vote_score: {top_vote['model']} ({top_vote['vote_score']:.2f})", flush=True)
        print(
            f"Top by {SMALL_VOTE_SCORE_COL}: {top_small_vote['model']} ({top_small_vote[SMALL_VOTE_SCORE_COL]:.2f})",
            flush=True,
        )
        print("Plots created: scatter correlations, correlation heatmaps, rank-agreement bump charts", flush=True)
        return

    device = get_device(require_cuda=args.require_cuda)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"Device: {device} | GPU: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("Device: cpu (Hinweis: fuer deinen Use-Case ist CUDA empfohlen)", flush=True)
    print(f"Data dir: {Path(args.data_dir).resolve()}", flush=True)

    train_loader, val_loader = get_mnist_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    suite = get_model_suite(num_classes=10, max_models=args.max_models)
    total_models = len(suite)
    if total_models == 0:
        raise RuntimeError("Keine Modelle im Suite-Setup erzeugt.")

    results = []
    for idx, (spec, model) in enumerate(suite, start=1):
        print(f"\n[{idx}/{total_models}] {spec.name}: zero-cost metrics...", flush=True)
        zero_cost_scores = average_zero_cost_scores(
            model=model,
            train_loader=train_loader,
            device=device,
            metric_batches=args.metric_batches,
        )

        print(f"[{idx}/{total_models}] {spec.name}: training...", flush=True)
        (
            val_loss,
            val_acc,
            val_f1,
            val_mse,
            epochs_trained,
            train_time_sec,
            infer_ms_per_sample,
        ) = train_short(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
            use_amp=(device.type == "cuda" and not args.disable_amp),
            num_classes=10,
            status_prefix=f"[{idx}/{total_models}] {spec.name} | ",
            log_every_epoch=True,
            log_every_batches=args.log_batches,
        )

        print(
            f"[{idx}/{total_models}] {spec.name}: "
            f"val_acc={val_acc:.4f} | f1={val_f1:.4f} | train_sec={train_time_sec:.1f}",
            flush=True,
        )

        results.append(
            {
                "model": spec.name,
                "family": spec.family,
                "stage_untrained_metrics": "untrained",
                "stage_trained_eval": "trained",
                "depth": spec.depth,
                "width": spec.width,
                "use_batchnorm": spec.use_batchnorm,
                **zero_cost_scores,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1_macro": val_f1,
                "val_mse": val_mse,
                "epochs_trained": epochs_trained,
                "train_time_sec": train_time_sec,
                "infer_ms_per_sample": infer_ms_per_sample,
            }
        )

    results_df = pd.DataFrame(results)
    (
        results_df,
        results_path,
        mapping_overall_path,
        family_mapping_path,
    ) = _postprocess_and_plot(
        results_df=results_df,
        out_dir=out_dir,
        include_family_split=args.include_family_split,
        lower_is_better=lower_is_better,
    )

    top_acc = results_df.sort_values("val_acc", ascending=False).iloc[0]
    top_vote = results_df.sort_values("vote_score", ascending=False).iloc[0]
    top_small_vote = results_df.sort_values(SMALL_VOTE_SCORE_COL, ascending=False).iloc[0]

    print("\n=== Untrained CNN MNIST Artifact finished ===", flush=True)
    print(f"Models: {total_models} (plain_cnn + residual_cnn)", flush=True)
    print(f"Results: {results_path.resolve()}", flush=True)
    print(f"Overall mapping: {mapping_overall_path.resolve()}", flush=True)
    if family_mapping_path is not None:
        print(f"Family mapping: {family_mapping_path.resolve()}", flush=True)
    print(f"Top by val_acc: {top_acc['model']} ({top_acc['val_acc']:.4f})", flush=True)
    print(f"Top by vote_score: {top_vote['model']} ({top_vote['vote_score']:.2f})", flush=True)
    print(
        f"Top by {SMALL_VOTE_SCORE_COL}: {top_small_vote['model']} ({top_small_vote[SMALL_VOTE_SCORE_COL]:.2f})",
        flush=True,
    )
    print("Plots created: scatter correlations, correlation heatmaps, rank-agreement bump charts", flush=True)


if __name__ == "__main__":
    main()
