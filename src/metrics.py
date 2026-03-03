from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn


def grad_norm_score(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    loss_fn: nn.Module,
) -> float:
    model.zero_grad(set_to_none=True)
    logits = model(images)
    loss = loss_fn(logits, labels)
    loss.backward()
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return total**0.5


def jacob_cov_score(model: nn.Module, images: torch.Tensor, eps: float = 1e-6) -> float:
    was_training = model.training
    model.eval()
    try:
        x = images.detach().clone().requires_grad_(True)
        logits = model(x)
        per_sample = logits.sum(dim=1)
        grads = torch.autograd.grad(
            outputs=per_sample,
            inputs=x,
            grad_outputs=torch.ones_like(per_sample),
            create_graph=False,
            retain_graph=False,
        )[0]
    finally:
        model.train(was_training)

    jac = grads.flatten(start_dim=1).to(dtype=torch.float64)
    jac = jac - jac.mean(dim=1, keepdim=True)
    jac_std = jac.std(dim=1, unbiased=False, keepdim=True).clamp_min(eps)
    jac_norm = jac / jac_std
    corr = (jac_norm @ jac_norm.t()) / max(jac_norm.shape[1], 1)
    corr = corr - torch.diag_embed(torch.diagonal(corr))
    denom = max(corr.numel() - corr.shape[0], 1)
    return float(corr.abs().sum().item() / denom)


def synflow_score(model: nn.Module, input_shape: List[int], device: torch.device) -> float:
    signs: Dict[str, torch.Tensor] = {}
    was_training = model.training
    model.eval()

    for name, p in model.named_parameters():
        signs[name] = torch.sign(p.data)
        p.data = p.data.abs()

    try:
        model.zero_grad(set_to_none=True)
        x = torch.ones(input_shape, device=device)
        y = model(x)
        torch.sum(y).backward()
        score = 0.0
        for p in model.parameters():
            if p.grad is not None:
                score += torch.sum(torch.abs(p.grad * p)).item()
        return score
    finally:
        for name, p in model.named_parameters():
            p.data.mul_(signs[name])
        model.train(was_training)


def snip_score(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    loss_fn: nn.Module,
) -> float:
    model.zero_grad(set_to_none=True)
    logits = model(images)
    loss = loss_fn(logits, labels)
    loss.backward()
    score = 0.0
    for p in model.parameters():
        if p.grad is not None:
            score += torch.sum(torch.abs(p.grad * p)).item()
    return score


def fisher_score(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    loss_fn: nn.Module,
) -> float:
    activations: List[torch.Tensor] = []
    hooks = []

    def _capture_output(_: nn.Module, __: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        if isinstance(output, torch.Tensor):
            output.retain_grad()
            activations.append(output)

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(_capture_output))

    try:
        model.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        score = 0.0
        for activation in activations:
            if activation.grad is not None:
                score += torch.sum(activation.grad.detach().pow(2)).item()
        return score
    finally:
        for hook in hooks:
            hook.remove()


def naswot_score(model: nn.Module, images: torch.Tensor, eps: float = 1e-6) -> float:
    activations: List[torch.Tensor] = []
    hooks = []

    def _relu_hook(_: nn.Module, __: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        if isinstance(output, torch.Tensor):
            activations.append(output.detach())

    for module in model.modules():
        if isinstance(module, nn.ReLU):
            hooks.append(module.register_forward_hook(_relu_hook))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            _ = model(images)
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)

    if not activations:
        return 0.0

    batch_size = images.shape[0]
    kernel = torch.zeros((batch_size, batch_size), device=images.device, dtype=torch.float64)
    for act in activations:
        binary = (act > 0).flatten(start_dim=1).to(dtype=torch.float64)
        ones = torch.ones_like(binary)
        kernel = kernel + (binary @ binary.t()) + ((ones - binary) @ (ones - binary).t())
    kernel = kernel + eps * torch.eye(batch_size, device=images.device, dtype=torch.float64)
    sign, logabsdet = torch.linalg.slogdet(kernel)
    if sign <= 0:
        return float("nan")
    return float(logabsdet.item())


def _swap_maxpool_to_avgpool(model: nn.Module) -> List[tuple[nn.Module, str, nn.Module]]:
    swaps: List[tuple[nn.Module, str, nn.Module]] = []

    def _visit(module: nn.Module) -> None:
        for name, child in module.named_children():
            if isinstance(child, nn.MaxPool2d):
                avg = nn.AvgPool2d(
                    kernel_size=child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    ceil_mode=child.ceil_mode,
                    count_include_pad=False,
                )
                setattr(module, name, avg)
                swaps.append((module, name, child))
            else:
                _visit(child)

    _visit(model)
    return swaps


def _restore_pool_layers(swaps: List[tuple[nn.Module, str, nn.Module]]) -> None:
    for parent, name, original_layer in swaps:
        setattr(parent, name, original_layer)


def grasp_score(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    loss_fn: nn.Module,
) -> float:
    batch_size = images.shape[0]
    if batch_size < 2:
        return 0.0

    split = batch_size // 2
    x1, y1 = images[:split], labels[:split]
    x2, y2 = images[split:], labels[split:]
    params = [p for p in model.parameters() if p.requires_grad]
    swaps = _swap_maxpool_to_avgpool(model)

    try:
        model.zero_grad(set_to_none=True)
        logits1 = model(x1)
        loss1 = loss_fn(logits1, y1)
        grads1 = torch.autograd.grad(loss1, params, create_graph=False, retain_graph=False)

        model.zero_grad(set_to_none=True)
        logits2 = model(x2)
        loss2 = loss_fn(logits2, y2)
        grads2 = torch.autograd.grad(loss2, params, create_graph=True, retain_graph=True)

        grad_dot = torch.zeros((), device=images.device)
        for g1, g2 in zip(grads1, grads2):
            grad_dot = grad_dot + torch.sum(g1.detach() * g2)

        model.zero_grad(set_to_none=True)
        grad_dot.backward()

        score = 0.0
        for p in params:
            if p.grad is not None:
                score += torch.sum(-(p * p.grad)).item()
        return score
    finally:
        _restore_pool_layers(swaps)


def compute_all_zero_cost_metrics(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    model = model.to(device)
    model.train()
    loss_fn = nn.CrossEntropyLoss()

    x = images.to(device, non_blocking=True)
    y = labels.to(device, non_blocking=True)

    grad_score = grad_norm_score(model, x, y, loss_fn)
    model.zero_grad(set_to_none=True)
    jacob_cov = jacob_cov_score(model, x)
    model.zero_grad(set_to_none=True)
    synflow = synflow_score(model, input_shape=list(x.shape), device=device)
    model.zero_grad(set_to_none=True)
    snip = snip_score(model, x, y, loss_fn)
    model.zero_grad(set_to_none=True)
    fisher = fisher_score(model, x, y, loss_fn)
    model.zero_grad(set_to_none=True)
    grasp = grasp_score(model, x, y, loss_fn)
    model.zero_grad(set_to_none=True)
    naswot = naswot_score(model, x)

    return {
        "grad_norm": grad_score,
        "jacob_cov": jacob_cov,
        "synflow": synflow,
        "snip": snip,
        "fisher": fisher,
        "grasp": grasp,
        "naswot": naswot,
    }

