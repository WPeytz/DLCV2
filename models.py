"""
Model architectures for video classification.
Includes per-frame models, early/late fusion, 3D CNNs, and dual-stream networks.

Now with optional per-model seeding:
- Every class accepts `seed: Optional[int] = None`.
- When set, all parameter initializations in __init__ run inside a forked RNG
  context seeded with that value, so weights are reproducible without affecting
  the global RNG used by your training loop or DataLoader workers.
"""

from typing import Optional
from contextlib import contextmanager

import torch
import torch.nn as nn
from torchvision import models


# ---------------------- Seeding utilities ---------------------- #
@contextmanager
def _seed_context(seed: Optional[int]):
    """
    Fork the RNG state and (optionally) seed torch for deterministic
    initialization without affecting the caller's global RNG state.
    """
    if seed is None:
        yield
        return
    with torch.random.fork_rng(enabled=True):
        # CPU (and CUDA if available; harmless otherwise)
        torch.manual_seed(int(seed))
        try:
            torch.cuda.manual_seed_all(int(seed))
        except Exception:
            pass
        yield


def _init_linear_like_resnet_fc(layer: nn.Linear):
    """
    Initialize a Linear layer similar to torchvision's ResNet FC:
      weight ~ N(0, 0.01), bias = 0
    """
    nn.init.normal_(layer.weight, mean=0.0, std=0.01)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
# --------------------------------------------------------------- #


class PerFrameCNN(nn.Module):
    """
    Per-frame CNN model with aggregation.
    Processes each frame independently and aggregates predictions.
    """

    def __init__(self, num_classes: int = 10, backbone: str = 'resnet18',
                 pretrained: bool = True, aggregation: str = 'mean',
                 seed: Optional[int] = None):
        super().__init__()
        self.aggregation = aggregation

        with _seed_context(seed):
            # Load pretrained backbone
            if backbone == 'resnet18':
                self.backbone = models.resnet18(pretrained=pretrained)
                in_features = self.backbone.fc.in_features
                self.backbone.fc = nn.Linear(in_features, num_classes)
                # Init classifier deterministically
                _init_linear_like_resnet_fc(self.backbone.fc)
            elif backbone == 'resnet50':
                self.backbone = models.resnet50(pretrained=pretrained)
                in_features = self.backbone.fc.in_features
                self.backbone.fc = nn.Linear(in_features, num_classes)
                _init_linear_like_resnet_fc(self.backbone.fc)
            else:
                raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, x):
        """
        Args:
            x: Single frame [B, C, H, W] or stacked frames [B, C, T, H, W]
        """
        if len(x.shape) == 5:  # [B, C, T, H, W]
            B, C, T, H, W = x.shape
            # Reshape to [B*T, C, H, W] to process all frames
            x = x.permute(0, 2, 1, 3, 4).contiguous()  # [B, T, C, H, W]
            x = x.view(B * T, C, H, W)

            # Process through backbone
            frame_outputs = self.backbone(x)  # [B*T, num_classes]

            # Reshape back to [B, T, num_classes]
            frame_outputs = frame_outputs.view(B, T, -1)

            # Aggregate predictions
            if self.aggregation == 'mean':
                output = torch.mean(frame_outputs, dim=1)
            elif self.aggregation == 'max':
                output, _ = torch.max(frame_outputs, dim=1)
            else:
                raise ValueError(f"Unsupported aggregation: {self.aggregation}")

            return output
        else:
            # Single frame processing
            return self.backbone(x)


class LateFusionCNN(nn.Module):
    """
    Late fusion model: Extract features from each frame independently,
    then combine features before classification.
    """

    def __init__(self, num_classes: int = 10, backbone: str = 'resnet18',
                 pretrained: bool = True, fusion: str = 'concat',
                 seed: Optional[int] = None):
        super().__init__()
        self.fusion = fusion

        with _seed_context(seed):
            # Load pretrained backbone without classification layer
            if backbone == 'resnet18':
                base_model = models.resnet18(pretrained=pretrained)
                self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
                feature_dim = 512
            elif backbone == 'resnet50':
                base_model = models.resnet50(pretrained=pretrained)
                self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
                feature_dim = 2048
            else:
                raise ValueError(f"Unsupported backbone: {backbone}")

            # Classifier after fusion
            if fusion == 'concat':
                # Concatenate all frame features
                self.classifier = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(feature_dim * 10, 512),  # Assuming 10 frames
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(512, num_classes)
                )
                # Optional: initialize the last classifier like ResNet FC for stability
                _init_linear_like_resnet_fc(self.classifier[-1])
            elif fusion == 'mean':
                self.classifier = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(feature_dim, num_classes)
                )
                _init_linear_like_resnet_fc(self.classifier[-1])
            else:
                raise ValueError(f"Unsupported fusion: {fusion}")

    def forward(self, x):
        """
        Args:
            x: Stacked frames [B, C, T, H, W]
        """
        B, C, T, H, W = x.shape

        # Reshape to [B*T, C, H, W]
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # [B, T, C, H, W]
        x = x.view(B * T, C, H, W)

        # Extract features
        features = self.feature_extractor(x)  # [B*T, feature_dim, 1, 1]
        features = features.view(B, T, -1)  # [B, T, feature_dim]

        # Fusion
        if self.fusion == 'concat':
            features = features.view(B, -1)  # [B, T * feature_dim]
        elif self.fusion == 'mean':
            features = torch.mean(features, dim=1)  # [B, feature_dim]

        # Classification
        output = self.classifier(features)
        return output


class EarlyFusionCNN(nn.Module):
    """
    Early fusion model: Concatenate all frames in the channel dimension,
    then process with 2D CNN.
    """

    def __init__(self, num_classes: int = 10, backbone: str = 'resnet18',
                 pretrained: bool = True, num_frames: int = 10,
                 seed: Optional[int] = None):
        super().__init__()
        self.num_frames = num_frames

        with _seed_context(seed):
            # Modify first conv layer to accept more channels
            if backbone == 'resnet18':
                base_model = models.resnet18(pretrained=pretrained)
            elif backbone == 'resnet50':
                base_model = models.resnet50(pretrained=pretrained)
            else:
                raise ValueError(f"Unsupported backbone: {backbone}")

            # Replace first conv layer to accept 3 * num_frames channels
            original_conv = base_model.conv1
            self.conv1 = nn.Conv2d(
                3 * num_frames,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False
            )

            # Copy pretrained weights for the first 3 channels, average/replicate for others
            if pretrained:
                with torch.no_grad():
                    # Start with zeros, then distribute pretrained weights evenly
                    nn.init.zeros_(self.conv1.weight)
                    for i in range(num_frames):
                        self.conv1.weight[:, i*3:(i+1)*3, :, :] += original_conv.weight / num_frames

            # Use rest of the model
            self.features = nn.Sequential(*list(base_model.children())[1:-1])
            in_features = base_model.fc.in_features
            self.fc = nn.Linear(in_features, num_classes)
            _init_linear_like_resnet_fc(self.fc)

    def forward(self, x):
        """
        Args:
            x: Stacked frames [B, C, T, H, W]
        """
        B, C, T, H, W = x.shape

        # Reshape to [B, C*T, H, W]
        x = x.view(B, C * T, H, W)

        # Forward pass
        x = self.conv1(x)
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class CNN3D(nn.Module):
    """
    3D CNN model using 3D convolutions to capture spatiotemporal features.
    """

    def __init__(self, num_classes: int = 10, seed: Optional[int] = None):
        super().__init__()

        with _seed_context(seed):
            self.features = nn.Sequential(
                # First 3D conv block
                nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),

                # Second 3D conv block
                nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                nn.BatchNorm3d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),

                # Third 3D conv block
                nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                nn.BatchNorm3d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),

                # Fourth 3D conv block
                nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                nn.BatchNorm3d(512),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
            )

            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool3d((1, 1, 1)),
                nn.Flatten(),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )

            # Optional: make the final classifier deterministic like ResNet FC
            _init_linear_like_resnet_fc(self.classifier[-1])

    def forward(self, x):
        """
        Args:
            x: Stacked frames [B, C, T, H, W]
        """
        x = self.features(x)
        x = self.classifier(x)
        return x


class DualStreamNetwork(nn.Module):
    """
    Dual-stream (two-stream) network that processes RGB and optical flow separately,
    then fuses predictions.

    Fusion modes:
      - 'late'   : average logits (predictions) from two classifiers
      - 'concat' : concatenate features then classify
      - 'mean'   : average features from both streams then classify
    """

    def __init__(self, num_classes: int = 10, backbone: str = 'resnet18',
                 pretrained: bool = True, fusion: str = 'late',
                 seed: Optional[int] = None):
        super().__init__()
        self.fusion = fusion

        with _seed_context(seed):
            # Build backbones and record feature_dim
            if backbone == 'resnet18':
                spatial = models.resnet18(pretrained=pretrained)
                temporal = models.resnet18(pretrained=pretrained)
                feature_dim = 512
            elif backbone == 'resnet50':
                spatial = models.resnet50(pretrained=pretrained)
                temporal = models.resnet50(pretrained=pretrained)
                feature_dim = 2048
            else:
                raise ValueError(f"Unsupported backbone: {backbone}")

            if fusion == 'late':
                # Late fusion: classify each stream, then average logits
                in_features = spatial.fc.in_features
                spatial.fc = nn.Linear(in_features, num_classes)
                temporal.fc = nn.Linear(in_features, num_classes)
                _init_linear_like_resnet_fc(spatial.fc)
                _init_linear_like_resnet_fc(temporal.fc)
                self.spatial_stream = spatial
                self.temporal_stream = temporal

            elif fusion == 'concat':
                # Concatenate pooled features, then classify
                self.spatial_stream = nn.Sequential(*list(spatial.children())[:-1])   # [B, C, 1, 1]
                self.temporal_stream = nn.Sequential(*list(temporal.children())[:-1]) # [B, C, 1, 1]
                self.fusion_classifier = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(feature_dim * 2, 512),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(512, num_classes)
                )
                _init_linear_like_resnet_fc(self.fusion_classifier[-1])

            elif fusion == 'mean':
                # Average pooled features from both streams, then classify
                self.spatial_stream = nn.Sequential(*list(spatial.children())[:-1])   # [B, C, 1, 1]
                self.temporal_stream = nn.Sequential(*list(temporal.children())[:-1]) # [B, C, 1, 1]
                self.mean_classifier = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(feature_dim, num_classes)
                )
                _init_linear_like_resnet_fc(self.mean_classifier[-1])
            else:
                raise ValueError(f"Unsupported fusion: {fusion}")

    def forward(self, rgb, flow):
        """
        Args:
            rgb:  [B, C, T, H, W] or [B, C, H, W]
            flow: [B, C, T, H, W] or [B, C, H, W]
        """
        # If 5D input, pool over time for simplicity
        if rgb.dim() == 5:
            rgb = torch.mean(rgb, dim=2)   # -> [B, C, H, W]
            flow = torch.mean(flow, dim=2)

        if self.fusion == 'late':
            rgb_logits = self.spatial_stream(rgb)
            flow_logits = self.temporal_stream(flow)
            return (rgb_logits + flow_logits) / 2

        elif self.fusion == 'concat':
            rgb_feat = self.spatial_stream(rgb)         # [B, C, 1, 1]
            flow_feat = self.temporal_stream(flow)      # [B, C, 1, 1]
            rgb_feat = torch.flatten(rgb_feat, 1)       # [B, C]
            flow_feat = torch.flatten(flow_feat, 1)     # [B, C]
            fused = torch.cat([rgb_feat, flow_feat], dim=1)  # [B, 2C]
            return self.fusion_classifier(fused)

        elif self.fusion == 'mean':
            rgb_feat = self.spatial_stream(rgb)         # [B, C, 1, 1]
            flow_feat = self.temporal_stream(flow)      # [B, C, 1, 1]
            rgb_feat = torch.flatten(rgb_feat, 1)       # [B, C]
            flow_feat = torch.flatten(flow_feat, 1)     # [B, C]
            fused = 0.5 * (rgb_feat + flow_feat)        # [B, C]
            return self.mean_classifier(fused)


if __name__ == "__main__":
    # Test models
    batch_size = 2
    num_frames = 10
    x = torch.randn(batch_size, 3, num_frames, 224, 224)

    print("Testing PerFrameCNN:")
    model = PerFrameCNN(num_classes=10, seed=42)
    out = model(x)
    print(f"Output shape: {out.shape}")

    print("\nTesting LateFusionCNN:")
    model = LateFusionCNN(num_classes=10, seed=42)
    out = model(x)
    print(f"Output shape: {out.shape}")

    print("\nTesting EarlyFusionCNN:")
    model = EarlyFusionCNN(num_classes=10, seed=42)
    out = model(x)
    print(f"Output shape: {out.shape}")

    print("\nTesting CNN3D:")
    model = CNN3D(num_classes=10, seed=42)
    out = model(x)
    print(f"Output shape: {out.shape}")

    print("\nTesting DualStreamNetwork:")
    flow = torch.randn(batch_size, 3, num_frames, 224, 224)
    model = DualStreamNetwork(num_classes=10, fusion='mean', seed=42)
    out = model(x, flow)
    print(f"Output shape: {out.shape}")
