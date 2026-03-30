"""CNN model factory — DenseNet121, ResNet50, EfficientNet-B4, ConvNeXt-Small."""
from __future__ import annotations

import torch.nn as nn
import torchvision.models as tvm
from torchvision.models import (
    ConvNeXt_Small_Weights,
    DenseNet121_Weights,
    EfficientNet_B4_Weights,
    ResNet50_Weights,
)

SUPPORTED_ARCHS = ("densenet121", "resnet50", "efficientnet_b4", "convnext_small")


def build_model(
    architecture: str,
    num_classes: int = 14,
    pretrained: bool = True,
) -> nn.Module:
    """
    Returns a torchvision model with the classification head replaced by
    nn.Linear(in_features, num_classes) — raw logits, no sigmoid/softmax.

    Use BCEWithLogitsLoss for training.
    """
    arch = architecture.lower()

    if arch == "densenet121":
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = tvm.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)

    elif arch == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = tvm.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif arch == "efficientnet_b4":
        weights = EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        model = tvm.efficientnet_b4(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif arch == "convnext_small":
        weights = ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = tvm.convnext_small(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(
            f"Unknown architecture '{architecture}'. Supported: {SUPPORTED_ARCHS}"
        )

    return model


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Returns (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
