from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


@dataclass
class EvalResult:
    loss: float
    acc: float
    f1_macro: float
    mse: float
    infer_ms_per_sample: float


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _macro_f1_from_confusion(conf_mat: torch.Tensor, eps: float = 1e-12) -> float:
    tp = torch.diag(conf_mat)
    fp = conf_mat.sum(dim=0) - tp
    fn = conf_mat.sum(dim=1) - tp
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return float(f1.mean().item())


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
    num_classes: int = 10,
) -> EvalResult:
    loss_fn = nn.CrossEntropyLoss()
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total = 0
    total_mse = 0.0
    conf_mat = torch.zeros((num_classes, num_classes), device=device, dtype=torch.float64)

    _sync_device(device)
    start = time.perf_counter()

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda")):
                logits = model(images)
                loss = loss_fn(logits, labels)

            probs = torch.softmax(logits, dim=1)
            one_hot = F.one_hot(labels, num_classes=num_classes).to(dtype=probs.dtype)
            mse_batch = torch.mean((probs - one_hot) ** 2, dim=1).sum().item()

            preds = torch.argmax(logits, dim=1)
            idx = labels * num_classes + preds
            conf_mat += torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)

            total_loss += loss.item() * images.size(0)
            total_mse += mse_batch
            total_correct += (preds == labels).sum().item()
            total += images.size(0)

    _sync_device(device)
    infer_total_sec = time.perf_counter() - start
    infer_ms_per_sample = (infer_total_sec * 1000.0) / max(total, 1)

    return EvalResult(
        loss=total_loss / max(total, 1),
        acc=total_correct / max(total, 1),
        f1_macro=_macro_f1_from_confusion(conf_mat),
        mse=total_mse / max(total, 1),
        infer_ms_per_sample=infer_ms_per_sample,
    )


def train_short(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 30,
    lr: float = 1e-3,
    early_stopping_patience: int = 6,
    early_stopping_min_delta: float = 1e-4,
    use_amp: bool = False,
    num_classes: int = 10,
    status_prefix: str = "",
    log_every_epoch: bool = True,
    log_every_batches: int = 0,
) -> Tuple[float, float, float, float, int, float, float]:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))

    best_val_loss = float("inf")
    patience_counter = 0
    epochs_trained = 0

    _sync_device(device)
    train_start = time.perf_counter()

    final_eval = EvalResult(loss=float("inf"), acc=0.0, f1_macro=0.0, mse=0.0, infer_ms_per_sample=0.0)

    num_train_batches = len(train_loader)

    for _ in range(epochs):
        model.train()
        running_loss = 0.0
        total_samples = 0
        for batch_idx, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda")):
                logits = model(images)
                loss = loss_fn(logits, labels)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

            if log_every_batches > 0 and (batch_idx % log_every_batches == 0 or batch_idx == num_train_batches):
                batch_loss = running_loss / max(total_samples, 1)
                print(
                    f"{status_prefix}Epoch {epochs_trained + 1:02d}/{epochs} | "
                    f"batch {batch_idx}/{num_train_batches} | train_loss={batch_loss:.4f}",
                    flush=True,
                )

        final_eval = evaluate(
            model=model,
            data_loader=val_loader,
            device=device,
            use_amp=use_amp,
            num_classes=num_classes,
        )
        epochs_trained += 1
        train_loss = running_loss / max(total_samples, 1)

        if log_every_epoch:
            print(
                f"{status_prefix}Epoch {epochs_trained:02d}/{epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={final_eval.loss:.4f} | "
                f"val_acc={final_eval.acc:.4f} | val_f1={final_eval.f1_macro:.4f} | "
                f"patience={patience_counter}/{early_stopping_patience}",
                flush=True,
            )

        if final_eval.loss < (best_val_loss - early_stopping_min_delta):
            best_val_loss = final_eval.loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                if log_every_epoch:
                    print(f"{status_prefix}Early stopping triggered.", flush=True)
                break

    _sync_device(device)
    train_time_sec = time.perf_counter() - train_start

    return (
        final_eval.loss,
        final_eval.acc,
        final_eval.f1_macro,
        final_eval.mse,
        epochs_trained,
        train_time_sec,
        final_eval.infer_ms_per_sample,
    )
