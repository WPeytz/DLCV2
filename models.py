"""
Model architectures for video classification.
Includes per-frame models, early/late fusion, 3D CNNs, and dual-stream networks.
"""

import torch
import torch.nn as nn
from torchvision import models


class PerFrameCNN(nn.Module):
    """
    Per-frame CNN model with aggregation.
    Processes each frame independently and aggregates predictions.
    """

    def __init__(self, num_classes=10, backbone='resnet18', pretrained=True, aggregation='mean'):
        super().__init__()
        self.aggregation = aggregation

        # Load pretrained backbone
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
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

    def __init__(self, num_classes=10, backbone='resnet18', pretrained=True, fusion='concat'):
        super().__init__()
        self.fusion = fusion

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
        elif fusion == 'mean':
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(feature_dim, num_classes)
            )
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

    def __init__(self, num_classes=10, backbone='resnet18', pretrained=True, num_frames=10):
        super().__init__()
        self.num_frames = num_frames

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

        # Copy pretrained weights for the first 3 channels, repeat for others
        if pretrained:
            with torch.no_grad():
                for i in range(num_frames):
                    self.conv1.weight[:, i*3:(i+1)*3, :, :] = original_conv.weight / num_frames

        # Use rest of the model
        self.features = nn.Sequential(*list(base_model.children())[1:-1])
        in_features = base_model.fc.in_features
        self.fc = nn.Linear(in_features, num_classes)

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

    def __init__(self, num_classes=10):
        super().__init__()

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

    def forward(self, x):
        """
        Args:
            x: Stacked frames [B, C, T, H, W]
        """
        x = self.features(x)
        x = self.classifier(x)
        return x


class PerFrame3D(nn.Module):
    """
    Per-frame 3D CNN model: Process small temporal clips with 3D convolutions,
    then aggregate predictions across clips.
    """

    def __init__(self, num_classes=10, clip_size=3, aggregation='mean'):
        super().__init__()
        self.clip_size = clip_size
        self.aggregation = aggregation

        # 3D CNN for processing small temporal clips
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
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),

            # Third 3D conv block
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
        )

        # Classifier for each clip
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: Stacked frames [B, C, T, H, W]
        """
        B, C, T, H, W = x.shape

        # Create overlapping clips
        clips = []
        for i in range(0, T - self.clip_size + 1):
            clip = x[:, :, i:i+self.clip_size, :, :]  # [B, C, clip_size, H, W]
            clips.append(clip)

        if len(clips) == 0:
            # If video is shorter than clip_size, pad it
            clip = x
            if T < self.clip_size:
                padding = self.clip_size - T
                clip = torch.nn.functional.pad(clip, (0, 0, 0, 0, 0, padding))
            clips.append(clip)

        # Process each clip
        clip_outputs = []
        for clip in clips:
            features = self.features(clip)
            output = self.classifier(features)
            clip_outputs.append(output)

        # Stack all clip predictions [B, num_clips, num_classes]
        clip_outputs = torch.stack(clip_outputs, dim=1)

        # Aggregate predictions
        if self.aggregation == 'mean':
            output = torch.mean(clip_outputs, dim=1)
        elif self.aggregation == 'max':
            output, _ = torch.max(clip_outputs, dim=1)
        else:
            raise ValueError(f"Unsupported aggregation: {self.aggregation}")

        return output


class LateFusion3D(nn.Module):
    """
    Late fusion 3D CNN model: Extract 3D spatiotemporal features from temporal segments,
    then combine features before classification.
    """

    def __init__(self, num_classes=10, num_segments=3, fusion='concat'):
        super().__init__()
        self.num_segments = num_segments
        self.fusion = fusion

        # 3D CNN feature extractor - uses adaptive pooling to handle variable segment sizes
        self.conv_blocks = nn.Sequential(
            # First 3D conv block
            nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),

            # Second 3D conv block
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),

            # Third 3D conv block
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),

            # Fourth 3D conv block
            nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        )

        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flatten = nn.Flatten()

        feature_dim = 512

        # Classifier after fusion
        if fusion == 'concat':
            # Concatenate all segment features
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(feature_dim * num_segments, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
        elif fusion == 'mean':
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(feature_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
        else:
            raise ValueError(f"Unsupported fusion: {fusion}")

    def forward(self, x):
        """
        Args:
            x: Stacked frames [B, C, T, H, W]
        """
        B, C, T, H, W = x.shape

        # Divide video into segments
        segment_length = T // self.num_segments
        if segment_length == 0:
            segment_length = 1
            num_segments = T
        else:
            num_segments = self.num_segments

        segments = []
        for i in range(num_segments):
            start = i * segment_length
            end = start + segment_length if i < num_segments - 1 else T
            segment = x[:, :, start:end, :, :]
            segments.append(segment)

        # Extract features from each segment
        segment_features = []
        for segment in segments:
            features = self.conv_blocks(segment)  # [B, 512, T', H', W']
            features = self.global_pool(features)  # [B, 512, 1, 1, 1]
            features = self.flatten(features)  # [B, 512]
            segment_features.append(features)

        # Stack features [B, num_segments, feature_dim]
        segment_features = torch.stack(segment_features, dim=1)

        # Fusion
        if self.fusion == 'concat':
            features = segment_features.view(B, -1)  # [B, num_segments * feature_dim]
        elif self.fusion == 'mean':
            features = torch.mean(segment_features, dim=1)  # [B, feature_dim]

        # Classification
        output = self.classifier(features)
        return output


class DualStreamNetwork(nn.Module):
    """
    Dual-stream (two-stream) network that processes RGB and optical flow separately,
    then fuses predictions.
    """

    def __init__(self, num_classes=10, backbone='resnet18', pretrained=True, fusion='late'):
        super().__init__()
        self.fusion = fusion

        # Spatial stream (RGB frames)
        if backbone == 'resnet18':
            self.spatial_stream = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        elif backbone == 'resnet50':
            self.spatial_stream = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Temporal stream (optical flow)
        if backbone == 'resnet18':
            self.temporal_stream = models.resnet18(pretrained=pretrained)
        elif backbone == 'resnet50':
            self.temporal_stream = models.resnet50(pretrained=pretrained)

        if fusion == 'late':
            # Late fusion: average predictions
            in_features = self.spatial_stream.fc.in_features
            self.spatial_stream.fc = nn.Linear(in_features, num_classes)
            self.temporal_stream.fc = nn.Linear(in_features, num_classes)
        elif fusion == 'concat':
            # Feature concatenation before classification
            self.spatial_stream = nn.Sequential(*list(self.spatial_stream.children())[:-1])
            self.temporal_stream = nn.Sequential(*list(self.temporal_stream.children())[:-1])
            self.fusion_classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(feature_dim * 2, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
        else:
            raise ValueError(f"Unsupported fusion: {fusion}")

    def forward(self, rgb, flow):
        """
        Args:
            rgb: RGB frames [B, C, T, H, W] or [B, C, H, W]
            flow: Optical flow frames [B, C, T, H, W] or [B, C, H, W]
        """
        # Handle both 4D and 5D inputs
        if len(rgb.shape) == 5:
            # Average pool over time dimension for simplicity
            rgb = torch.mean(rgb, dim=2)  # [B, C, H, W]
            flow = torch.mean(flow, dim=2)

        # Process both streams
        if self.fusion == 'late':
            rgb_out = self.spatial_stream(rgb)
            flow_out = self.temporal_stream(flow)
            # Average predictions
            output = (rgb_out + flow_out) / 2
        elif self.fusion == 'concat':
            rgb_features = self.spatial_stream(rgb).squeeze()
            flow_features = self.temporal_stream(flow).squeeze()
            # Concatenate features
            combined = torch.cat([rgb_features, flow_features], dim=1)
            output = self.fusion_classifier(combined)

        return output


if __name__ == "__main__":
    # Test models
    batch_size = 2
    num_frames = 10
    x = torch.randn(batch_size, 3, num_frames, 224, 224)

    print("Testing PerFrameCNN:")
    model = PerFrameCNN(num_classes=10)
    out = model(x)
    print(f"Output shape: {out.shape}")

    print("\nTesting LateFusionCNN:")
    model = LateFusionCNN(num_classes=10)
    out = model(x)
    print(f"Output shape: {out.shape}")

    print("\nTesting EarlyFusionCNN:")
    model = EarlyFusionCNN(num_classes=10)
    out = model(x)
    print(f"Output shape: {out.shape}")

    print("\nTesting CNN3D:")
    model = CNN3D(num_classes=10)
    out = model(x)
    print(f"Output shape: {out.shape}")

    print("\nTesting PerFrame3D:")
    model = PerFrame3D(num_classes=10, clip_size=3)
    out = model(x)
    print(f"Output shape: {out.shape}")

    print("\nTesting LateFusion3D:")
    model = LateFusion3D(num_classes=10, num_segments=3)
    out = model(x)
    print(f"Output shape: {out.shape}")

    print("\nTesting DualStreamNetwork:")
    flow = torch.randn(batch_size, 3, num_frames, 224, 224)
    model = DualStreamNetwork(num_classes=10)
    out = model(x, flow)
    print(f"Output shape: {out.shape}")
