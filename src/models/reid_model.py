"""
reid_model.py

Person Re-Identification (ReID) feature extraction models.

Implements the Omni-Scale Network (OSNet, Zhou et al., ICCV 2019) architecture
and provides model loading, weight downloading, and caching under CHECKPOINTS.

Pretrained on Market-1501 / MSMT17 to produce 512-dimensional L2-normalized
feature embeddings for detected person bounding boxes.
"""

# Group 1 — standard library
import os
import shutil
import urllib.request
from pathlib import Path

# Group 2 — third-party
import torch
import torch.nn as nn
import torch.nn.functional as F

# Group 3 — local / project
from config import CHECKPOINTS

# Model weights URLs (official Torchreid releases & HuggingFace mirrors)
MODEL_URLS = {
    "osnet_x1_0_market1501.pth": [
        "https://github.com/KaiyangZhou/deep-person-reid/releases/download/v1.0.0/osnet_x1_0_market1501.pth",
        "https://huggingface.co/pky/torchreid-models/resolve/main/osnet_x1_0_market1501.pth",
    ],
    "osnet_x1_0_msmt17.pth": [
        "https://github.com/KaiyangZhou/deep-person-reid/releases/download/v1.0.0/osnet_x1_0_msmt17.pth",
        "https://huggingface.co/pky/torchreid-models/resolve/main/osnet_x1_0_msmt17.pth",
    ],
}


# =========================================================================== #
# OSNet Building Blocks
# =========================================================================== #

class ConvLayer(nn.Module):
    """Convolution + BatchNorm + ReLU layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            groups=groups,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class Conv1x1(nn.Module):
    """1x1 Convolution + BatchNorm + ReLU."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, groups: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias=False,
            groups=groups,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class Conv1x1Linear(nn.Module):
    """1x1 Convolution + BatchNorm (no non-linearity)."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class LightConv3x3(nn.Module):
    """Lightweight 3x3 Convolution (Pointwise 1x1 + Depthwise 3x3)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            groups=out_channels,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return self.relu(self.bn(x))


class ChannelGate(nn.Module):
    """Channel attention gate for dynamic multi-scale stream aggregation."""

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        mid_channels = max(1, in_channels // reduction)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.global_avgpool(x)
        w = self.relu(self.fc1(w))
        w = self.gate(self.fc2(w))
        return w


class OSBlock(nn.Module):
    """Omni-Scale residual block with 4 scale streams and dynamic channel gating."""

    def __init__(self, in_channels: int, out_channels: int, bottleneck_reduction: int = 4):
        super().__init__()
        mid_channels = out_channels // bottleneck_reduction
        self.conv1 = Conv1x1(in_channels, mid_channels)

        # 4 Scale Streams
        self.conv2a = LightConv3x3(mid_channels, mid_channels)
        self.conv2b = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2c = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2d = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )

        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x1 = self.conv1(x)

        # Multi-scale streams
        s1 = self.conv2a(x1)
        s2 = self.conv2b(x1)
        s3 = self.conv2c(x1)
        s4 = self.conv2d(x1)

        # Dynamic channel-wise aggregation
        stream_sum = s1 + s2 + s3 + s4
        gate_weight = self.gate(stream_sum)
        fused = s1 * gate_weight + s2 * gate_weight + s3 * gate_weight + s4 * gate_weight

        out = self.conv3(fused)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = self.relu(out + identity)
        return out


# =========================================================================== #
# OSNet Architecture
# =========================================================================== #

class OSNet(nn.Module):
    """
    Omni-Scale Network (OSNet) for Person Re-Identification.

    Extracts a 512-dimensional feature embedding for each person image patch.
    """

    def __init__(
        self,
        blocks: list[int] = [2, 2, 2],
        channels: list[int] = [64, 256, 384, 512],
        feature_dim: int = 512,
    ):
        super().__init__()
        self.conv1 = ConvLayer(3, channels[0], kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # Stage 1
        self.conv2 = self._make_layer(channels[0], channels[1], blocks[0])
        self.transition1 = nn.Sequential(
            Conv1x1(channels[1], channels[1]),
            nn.AvgPool2d(2, stride=2),
        )

        # Stage 2
        self.conv3 = self._make_layer(channels[1], channels[2], blocks[1])
        self.transition2 = nn.Sequential(
            Conv1x1(channels[2], channels[2]),
            nn.AvgPool2d(2, stride=2),
        )

        # Stage 3
        self.conv4 = self._make_layer(channels[2], channels[3], blocks[2])
        self.conv5 = Conv1x1(channels[3], channels[3])

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels[3], feature_dim, bias=True),
            nn.BatchNorm1d(feature_dim)
        )

    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int) -> nn.Sequential:
        layers = [OSBlock(in_channels, out_channels)]
        for _ in range(num_blocks - 1):
            layers.append(OSBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (B, 3, H, W)
            normalize: If True, applies L2 normalization to feature vectors (||f||_2 = 1.0)
        Returns:
            features: Tensor of shape (B, 512)
        """
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.transition1(x)

        x = self.conv3(x)
        x = self.transition2(x)

        x = self.conv4(x)
        x = self.conv5(x)

        x = self.global_avgpool(x)
        features = x.view(x.size(0), -1)
        # Bypass self.fc to prevent similarity collapse (~0.999) observed with Market1501 pretrained weights
        features = self.fc(features)

        if normalize:
            features = F.normalize(features, p=2, dim=1)

        return features


# =========================================================================== #
# Weights Downloader and Model Factory
# =========================================================================== #

def download_weights(filename: str, target_path: Path) -> Path:
    """Download model weights if not already present in target_path."""
    if target_path.exists():
        return target_path

    target_path.parent.mkdir(parents=True, exist_ok=True)
    urls = MODEL_URLS.get(filename, [])
    if not urls:
        raise ValueError(f"No download URL registered for weights file: {filename}")

    print(f"[INFO] Downloading ReID weights: {filename} -> {target_path}")
    download_success = False
    for url in urls:
        try:
            print(f"[INFO] Attempting download from {url}")
            urllib.request.urlretrieve(url, str(target_path))
            if target_path.exists() and target_path.stat().st_size > 1000:
                download_success = True
                print(f"[INFO] Successfully downloaded {filename} ({target_path.stat().st_size / 1e6:.1f} MB)")
                break
        except Exception as e:
            print(f"[WARNING] Download from {url} failed: {e}")
            if target_path.exists():
                target_path.unlink()

    if not download_success:
        raise RuntimeError(
            f"Failed to download ReID weights '{filename}' from all mirrors. "
            f"Please download manually to {target_path}"
        )

    return target_path


def build_reid_model(
    model_name: str = "osnet_x1_0",
    weights_name: str = "osnet_x1_0_market1501.pth",
    pretrained: bool = True,
    device: str | None = None,
) -> nn.Module:
    """
    Constructs ReID model and loads pretrained weights from CHECKPOINTS.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_name != "osnet_x1_0":
        raise NotImplementedError(f"Model architecture '{model_name}' is not currently supported.")

    model = OSNet()

    if pretrained:
        weights_path = CHECKPOINTS / weights_name
        if not weights_path.exists():
            download_weights(weights_name, weights_path)

        state_dict = torch.load(str(weights_path), map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        # Strip prefixes and remap Torchreid OSNet transition layer keys
        model_dict = model.state_dict()
        filtered_dict = {}
        for k, v in state_dict.items():
            key = k.replace("module.", "")
            # Torchreid bundles transition layers inside conv2[2] and conv3[2]
            if key.startswith("conv2.2.0."):
                key = key.replace("conv2.2.0.", "transition1.0.")
            elif key.startswith("conv3.2.0."):
                key = key.replace("conv3.2.0.", "transition2.0.")

            if key in model_dict and model_dict[key].shape == v.shape:
                filtered_dict[key] = v

        model.load_state_dict(filtered_dict, strict=False)
        print(f"[INFO] Loaded ReID weights from {weights_path} ({len(filtered_dict)}/{len(model_dict)} keys matched)")

    model.to(device)
    model.eval()
    return model
