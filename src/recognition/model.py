import cv2
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn

"""
IResNet (Improved ResNet) Architecture for Face Recognition
- Input: Aligned face image (3 channels, 112x112 pixels)
- Output: 512-dimensional feature embedding vector

Architecture Layout:
- Input Stem: 3x3 Conv (stride 1) -> BatchNorm -> PReLU
- Stage 1: 3 Residual Blocks   (64 channels, spatial: 56x56)
- Stage 2: 13 Residual Blocks  (128 channels, spatial: 28x28)
- Stage 3: 30 Residual Blocks  (256 channels, spatial: 14x14)
- Stage 4: 3 Residual Blocks   (512 channels, spatial: 7x7)
- Output Head: BatchNorm -> Dropout -> Linear -> BatchNorm (512-D)
Total layers for IResNet100 = 100 convolutional/linear layers
"""


class ResidualBlock(nn.Module):
    """
    Standard building block for IResNet:
    input -> BatchNorm -> Conv3x3 -> BatchNorm -> PReLU -> Conv3x3 -> BatchNorm -> (+ shortcut) -> output
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # Main path layers
        self.batch_norm_1 = nn.BatchNorm2d(in_channels, eps=1e-5)
        self.conv_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.batch_norm_2 = nn.BatchNorm2d(out_channels, eps=1e-5)
        self.activation = nn.PReLU(out_channels)
        self.conv_2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.batch_norm_3 = nn.BatchNorm2d(out_channels, eps=1e-5)

        # Shortcut connection (downsample if dimensions or stride change)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels, eps=1e-5)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.batch_norm_1(x)
        out = self.conv_1(out)
        out = self.batch_norm_2(out)
        out = self.activation(out)
        out = self.conv_2(out)
        out = self.batch_norm_3(out)

        out = out + identity
        return out


class IResNet(nn.Module):
    def __init__(self, block_counts, embedding_size=512, dropout_rate=0.0):
        """
        Args:
            block_counts: list of 4 integers representing number of blocks in each stage.
                          For IResNet100: [3, 13, 30, 3]
                          For IResNet50:  [3, 4, 14, 3]
            embedding_size: Dimension of output face representation (default: 512)
            dropout_rate: Dropout probability before the embedding layer
        """
        super(IResNet, self).__init__()

        self.current_channels = 64

        # 1. Input Stem (preserves 112x112 resolution without immediate aggressive pooling)
        self.input_conv = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.input_batch_norm = nn.BatchNorm2d(64, eps=1e-5)
        self.input_activation = nn.PReLU(64)

        # 2. Four Residual Stages
        self.stage_1 = self._build_stage(out_channels=64,  num_blocks=block_counts[0], stride=2)
        self.stage_2 = self._build_stage(out_channels=128, num_blocks=block_counts[1], stride=2)
        self.stage_3 = self._build_stage(out_channels=256, num_blocks=block_counts[2], stride=2)
        self.stage_4 = self._build_stage(out_channels=512, num_blocks=block_counts[3], stride=2)

        # 3. Output Embedding Head
        # At this stage, feature map spatial size is 7x7 with 512 channels
        flattened_features_dim = 512 * 7 * 7

        self.output_batch_norm_2d = nn.BatchNorm2d(512, eps=1e-5)
        self.dropout = nn.Dropout(p=dropout_rate, inplace=True)
        self.fully_connected = nn.Linear(flattened_features_dim, embedding_size, bias=False)
        self.output_batch_norm_1d = nn.BatchNorm1d(embedding_size, eps=1e-5)

        # Initialize network weights
        self._initialize_weights()

    def _build_stage(self, out_channels, num_blocks, stride):
        layers = []
        # First block handles downsampling with stride=2
        layers.append(ResidualBlock(self.current_channels, out_channels, stride=stride))
        self.current_channels = out_channels

        # Remaining blocks keep spatial dimension the same (stride=1)
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(self.current_channels, out_channels, stride=1))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, face_images):
        """
        Args:
            face_images: Tensor of shape (batch_size, 3, 112, 112)
        Returns:
            feature_embeddings: Tensor of shape (batch_size, 512)
        """
        # Step 1: Input Stem
        x = self.input_conv(face_images)
        x = self.input_batch_norm(x)
        x = self.input_activation(x)

        # Step 2: Residual Stages
        x = self.stage_1(x)  # Shape: (batch, 64, 56, 56)
        x = self.stage_2(x)  # Shape: (batch, 128, 28, 28)
        x = self.stage_3(x)  # Shape: (batch, 256, 14, 14)
        x = self.stage_4(x)  # Shape: (batch, 512, 7, 7)

        # Step 3: Embedding Head
        x = self.output_batch_norm_2d(x)
        x = torch.flatten(x, start_dim=1)  # Flatten to (batch, 512 * 7 * 7)
        x = self.dropout(x)
        x = self.fully_connected(x)        # Project to 512-D
        feature_embeddings = self.output_batch_norm_1d(x)

        return feature_embeddings


# Helper functions to build standard model variants
def build_iresnet100(embedding_size=512, dropout_rate=0.0):
    """IResNet-100 with [3, 13, 30, 3] blocks"""
    return IResNet(block_counts=[3, 13, 30, 3], embedding_size=embedding_size, dropout_rate=dropout_rate)


def build_iresnet50(embedding_size=512, dropout_rate=0.0):
    """IResNet-50 with [3, 4, 14, 3] blocks"""
    return IResNet(block_counts=[3, 4, 14, 3], embedding_size=embedding_size, dropout_rate=dropout_rate)


FaceRecognitionBackbone = IResNet


class OnnxRecognitionModel:
    """OpenCV DNN wrapper for ONNX face recognition models (e.g., iresnet100.onnx)."""

    def __init__(self, onnx_path: str):
        self.net = cv2.dnn.readNet(str(onnx_path))

    def __call__(self, x_tensor: torch.Tensor) -> torch.Tensor:
        if hasattr(x_tensor, "detach"):
            arr = x_tensor.detach().cpu().numpy()
        else:
            arr = np.asarray(x_tensor)
        self.net.setInput(arr)
        out = self.net.forward()
        return torch.from_numpy(out)


def load_recognition_model(weights_path: str = "models/face_recognition/iresnet100.onnx", device: str = "cpu"):
    """Load face recognition model from ONNX or PyTorch weights."""
    path = Path(weights_path)
    if path.suffix.lower() == ".onnx" and path.exists():
        return OnnxRecognitionModel(str(path))

    default_onnx = Path("models/face_recognition/iresnet100.onnx")
    if not path.exists() and default_onnx.exists():
        return OnnxRecognitionModel(str(default_onnx))

    model = build_iresnet100()
    if path.exists() and path.suffix.lower() in [".pth", ".pt"]:
        checkpoint = torch.load(str(path), map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()
    return model

















# Step                         Tensor Shape                Feature Details
# ────────────────────────────────────────────────────────────────────────────────
# Raw Input                    [Batch, 3, 112, 112]        Raw RGB Pixels
#    │
#    ▼ Input Stem
# Input Conv (3x3)             [Batch, 64, 112, 112]       Initial 64 low-level filters
#    │
#    ▼ Stage 1 (3 blocks)
# Downsample (stride=2)        [Batch, 64, 56, 56]         Edges, simple textures
#    │
#    ▼ Stage 2 (13 blocks)
# Downsample (stride=2)        [Batch, 128, 28, 28]        Facial parts (eyes, nose)
#    │
#    ▼ Stage 3 (30 blocks)
# Downsample (stride=2)        [Batch, 256, 14, 14]        Facial structure & geometry
#    │
#    ▼ Stage 4 (3 blocks)
# Downsample (stride=2)        [Batch, 512, 7, 7]          High-level identity traits
#    │
#    ▼ Output Head
# Flatten                      [Batch, 25088]              Flattened feature array
# Linear Projection            [Batch, 512]                512-D Raw Face Embedding
# BatchNorm1d                  [Batch, 512]                Final Face Representation




#                Input: x
#                   │
#         ┌─────────┴─────────┐
#         │                   │
#   [Main Path: F(x)]     [Shortcut Path]
#         │                   │
#   BatchNorm2d           (Identity ya
#         │                Conv 1x1 if stride=2)
#      Conv 3x3               │
#         │                   │
#   BatchNorm2d               │
#         │                   │
#       PReLU                 │
#         │                   │
#      Conv 3x3               │
#         │                   │
#   BatchNorm2d               │
#         │                   │
#         └─────────┬─────────┘
#                   ▼
#             y = F(x) + x   <── (Element-wise Addition)
#                   │
#              Output: y

# Step                         Tensor Shape                Feature Details
# ────────────────────────────────────────────────────────────────────────────────
# Raw Input                    [Batch, 3, 112, 112]        Raw RGB Pixels
#    │
#    ▼ Input Stem
# Input Conv (3x3)             [Batch, 64, 112, 112]       Initial 64 low-level filters
#    │
#    ▼ Stage 1 (3 blocks)
# Downsample (stride=2)        [Batch, 64, 56, 56]         Edges, simple textures
#    │
#    ▼ Stage 2 (13 blocks)
# Downsample (stride=2)        [Batch, 128, 28, 28]        Facial parts (eyes, nose)
#    │
#    ▼ Stage 3 (30 blocks)
# Downsample (stride=2)        [Batch, 256, 14, 14]        Facial structure & geometry
#    │
#    ▼ Stage 4 (3 blocks)
# Downsample (stride=2)        [Batch, 512, 7, 7]          High-level identity traits
#    │
#    ▼ Output Head
# Flatten                      [Batch, 25088]              Flattened feature array
# Linear Projection            [Batch, 512]                512-D Raw Face Embedding
# BatchNorm1d                  [Batch, 512]                Final Face Representation




#                Input: x
#                   │
#         ┌─────────┴─────────┐
#         │                   │
#   [Main Path: F(x)]     [Shortcut Path]
#         │                   │
#   BatchNorm2d           (Identity ya
#         │                Conv 1x1 if stride=2)
#      Conv 3x3               │
#         │                   │
#   BatchNorm2d               │
#         │                   │
#       PReLU                 │
#         │                   │
#      Conv 3x3               │
#         │                   │
#   BatchNorm2d               │
#         │                   │
#         └─────────┬─────────┘
#                   ▼
#             y = F(x) + x   <── (Element-wise Addition)
#                   │
#              Output: y
