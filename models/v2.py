from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import ResNet50_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.resnet import resnet50


class StructuralFeatureAdapter(nn.Module):
    """
    Adds explicit normalized spatial structure to FPN features.

    The input feature map is:
        [B, C, H, W]

    We construct normalized coordinate maps:
        x ∈ [-1, 1]
        y ∈ [-1, 1]

    These are projected and fused with the visual features.
    """

    def __init__(
        self,
        channels: int = 256,
    ):
        super().__init__()

        self.visual_projection = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
        )

        self.structure_projection = nn.Sequential(
            nn.Conv2d(
                2,
                channels,
                kernel_size=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
            ),
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(
                channels * 2,
                channels,
                kernel_size=1,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:

        batch_size, _, height, width = features.shape

        device = features.device
        dtype = features.dtype

        # Normalized horizontal coordinates.
        x = torch.linspace(
            -1.0,
            1.0,
            width,
            device=device,
            dtype=dtype,
        )

        # Normalized vertical coordinates.
        y = torch.linspace(
            -1.0,
            1.0,
            height,
            device=device,
            dtype=dtype,
        )

        yy, xx = torch.meshgrid(
            y,
            x,
            indexing="ij",
        )

        coordinates = torch.stack(
            [xx, yy],
            dim=0,
        )

        coordinates = coordinates.unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
            -1,
        )

        visual_features = self.visual_projection(
            features
        )

        structural_features = self.structure_projection(
            coordinates
        )

        fused = torch.cat(
            [
                visual_features,
                structural_features,
            ],
            dim=1,
        )

        return self.fusion(fused)


class StructuralBackbone(nn.Module):
    """
    ResNet-50 + FPN backbone with a structural feature adapter
    applied independently to every FPN level.
    """

    def __init__(
        self,
        pretrained: bool = True,
        out_channels: int = 256,
    ):
        super().__init__()
        self.out_channels=out_channels

        weights = (
            ResNet50_Weights.DEFAULT
            if pretrained
            else None
        )

        backbone = resnet50(
            weights=weights,
        )

        return_layers = {
            "layer1": "0",
            "layer2": "1",
            "layer3": "2",
            "layer4": "3",
        }

        self.backbone = BackboneWithFPN(
            backbone,
            return_layers=return_layers,
            in_channels_list=[
                256,
                512,
                1024,
                2048,
            ],
            out_channels=out_channels,
        )

        self.adapters = nn.ModuleDict(
            {
                "0": StructuralFeatureAdapter(
                    channels=out_channels,
                ),
                "1": StructuralFeatureAdapter(
                    channels=out_channels,
                ),
                "2": StructuralFeatureAdapter(
                    channels=out_channels,
                ),
                "3": StructuralFeatureAdapter(
                    channels=out_channels,
                ),
            }
        )

    def forward(self, images, targets=None):

        features = self.backbone(
            images,
        )

        structural_features = {}

        for name, feature in features.items():

            if name in self.adapters:
                structural_features[name] = (
                    self.adapters[name](feature)
                )
            else:
                structural_features[name] = feature

        return structural_features


def build_v2_model(
    num_foreground_classes: int,
    pretrained_backbone: bool = True,
    min_size: int = 320,
    max_size: int = 512,
    rpn_pre_nms_top_n_train: int = 1000,
    rpn_post_nms_top_n_train: int = 1000,
    rpn_pre_nms_top_n_test: int = 500,
    rpn_post_nms_top_n_test: int = 500,
    box_detections_per_img: int = 300,
):
    """
    Build V2 Structural-Invariant DLA model.
    """

    backbone = StructuralBackbone(
        pretrained=pretrained_backbone,
        out_channels=256,
    )

    anchor_generator = AnchorGenerator(
        sizes=(
            (32,),
            (64,),
            (128,),
            (256,),
            (512,),
        ),
        aspect_ratios=(
            (0.5, 1.0, 2.0),
        ) * 5,
    )

    num_classes = (
        num_foreground_classes + 1
    )

    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        min_size=min_size,
        max_size=max_size,
        rpn_pre_nms_top_n_train=(
            rpn_pre_nms_top_n_train
        ),
        rpn_post_nms_top_n_train=(
            rpn_post_nms_top_n_train
        ),
        rpn_pre_nms_top_n_test=(
            rpn_pre_nms_top_n_test
        ),
        rpn_post_nms_top_n_test=(
            rpn_post_nms_top_n_test
        ),
        box_detections_per_img=(
            box_detections_per_img
        ),
    )

    return model