from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    depth: int
    width: int
    use_batchnorm: bool


class PlainCNN(nn.Module):
    def __init__(self, depth: int, width: int, num_classes: int = 10, use_batchnorm: bool = False):
        super().__init__()
        layers: List[nn.Module] = []
        in_ch = 1
        for idx in range(depth):
            layers.append(nn.Conv2d(in_ch, width, kernel_size=3, padding=1))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(width))
            layers.append(nn.ReLU(inplace=False))
            if idx % 2 == 1:
                layers.append(nn.MaxPool2d(kernel_size=2))
            in_ch = width
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(width, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.head(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, use_batchnorm: bool):
        super().__init__()
        block: List[nn.Module] = [nn.Conv2d(channels, channels, kernel_size=3, padding=1)]
        if use_batchnorm:
            block.append(nn.BatchNorm2d(channels))
        block.extend([nn.ReLU(inplace=False), nn.Conv2d(channels, channels, kernel_size=3, padding=1)])
        if use_batchnorm:
            block.append(nn.BatchNorm2d(channels))
        self.block = nn.Sequential(*block)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.block(x) + x)


class ResidualCNN(nn.Module):
    def __init__(self, depth: int, width: int, num_classes: int = 10, use_batchnorm: bool = False):
        super().__init__()
        stem: List[nn.Module] = [nn.Conv2d(1, width, kernel_size=3, padding=1)]
        if use_batchnorm:
            stem.append(nn.BatchNorm2d(width))
        stem.append(nn.ReLU(inplace=False))
        self.stem = nn.Sequential(*stem)

        blocks: List[nn.Module] = []
        for idx in range(depth):
            blocks.append(ResidualBlock(channels=width, use_batchnorm=use_batchnorm))
            if idx % 2 == 1:
                blocks.append(nn.MaxPool2d(kernel_size=2))
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(width, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.head(x)


def _build_specs(
    family: str,
    depths: Sequence[int],
    widths: Sequence[int],
    batchnorm_options: Sequence[bool],
) -> List[ModelSpec]:
    specs: List[ModelSpec] = []
    model_idx = 1
    for depth in depths:
        for width in widths:
            for use_batchnorm in batchnorm_options:
                bn_tag = "bn1" if use_batchnorm else "bn0"
                if family == "plain_cnn":
                    prefix = "PLAINCNN"
                else:
                    prefix = "RESCNN"
                name = f"{prefix}_{model_idx:02d}_d{depth}_w{width}_{bn_tag}"
                specs.append(
                    ModelSpec(
                        name=name,
                        family=family,
                        depth=depth,
                        width=width,
                        use_batchnorm=use_batchnorm,
                    )
                )
                model_idx += 1
    return specs


def get_model_suite(
    num_classes: int = 10,
    depths: Sequence[int] = (2, 3),
    widths: Sequence[int] = (8, 12, 16),
    batchnorm_options: Sequence[bool] = (False, True),
    max_models: int | None = 24,
) -> List[Tuple[ModelSpec, nn.Module]]:
    plain_specs = _build_specs("plain_cnn", depths, widths, batchnorm_options)
    residual_specs = _build_specs("residual_cnn", depths, widths, batchnorm_options)
    all_specs = plain_specs + residual_specs
    if max_models is not None:
        all_specs = all_specs[:max_models]

    suite: List[Tuple[ModelSpec, nn.Module]] = []
    for spec in all_specs:
        if spec.family == "plain_cnn":
            model = PlainCNN(
                depth=spec.depth,
                width=spec.width,
                num_classes=num_classes,
                use_batchnorm=spec.use_batchnorm,
            )
        else:
            model = ResidualCNN(
                depth=spec.depth,
                width=spec.width,
                num_classes=num_classes,
                use_batchnorm=spec.use_batchnorm,
            )
        suite.append((spec, model))
    return suite
