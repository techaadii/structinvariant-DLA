from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool


class RelationalFeatureBlock(nn.Module):
    """
    Learns local and larger-context spatial relationships from an FPN feature map.

    Input:
        [B, C, H, W]

    Output:
        [B, C, H, W]
    """

    def __init__(self, channels: int = 256):
        super().__init__()

        # Local spatial structure.
        self.local = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # Larger receptive field for broader layout relationships.
        self.context = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=2,
                dilation=2,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # Fuse the relational features.
        self.fusion = nn.Sequential(
            nn.Conv2d(
                channels * 2,
                channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.local(x)
        context = self.context(x)

        relational = self.fusion(
            torch.cat([local, context], dim=1)
        )

        # Residual connection preserves useful detection features.
        return x + relational


class StructuralFPN(nn.Module):
    """
    Applies the relational structural block independently to every FPN level.
    """

    def __init__(self, channels: int = 256):
        super().__init__()

        self.blocks = nn.ModuleDict()

        for level in ["0", "1", "2", "3", "pool"]:
            self.blocks[level] = RelationalFeatureBlock(channels)

        self.out_channels = channels

    def forward(self, features):
        return {
            name: self.blocks[name](feature)
            for name, feature in features.items()
        }


class ResNet50FPNBackbone(nn.Module):
    """
    ResNet-50 + FPN backbone followed by relational structural processing.
    """

    def __init__(
        self,
        pretrained: bool = True,
        out_channels: int = 256,
    ):
        super().__init__()

        weights = (
            ResNet50_Weights.DEFAULT
            if pretrained
            else None
        )

        backbone = resnet50(weights=weights)

        self.body = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

        # Extract the four ResNet stages.
        self.return_layers = {
            "4": "0",
            "5": "1",
            "6": "2",
            "7": "3",
        }

        in_channels_list = [256, 512, 1024, 2048]

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool(),
        )

        self.structural = StructuralFPN(out_channels)

        self.out_channels = out_channels

    def forward(self, x):
        outputs = {}

        x = self.body[0](x)
        x = self.body[1](x)
        x = self.body[2](x)
        x = self.body[3](x)

        x = self.body[4](x)
        outputs["0"] = x

        x = self.body[5](x)
        outputs["1"] = x

        x = self.body[6](x)
        outputs["2"] = x

        x = self.body[7](x)
        outputs["3"] = x

        features = self.fpn(outputs)

        return self.structural(features)


def build_model(
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
    backbone = ResNet50FPNBackbone(
        pretrained=pretrained_backbone,
        out_channels=256,
    )

    anchor_sizes = (
        (32,),
        (64,),
        (128,),
        (256,),
        (512,),
    )

    aspect_ratios = (
        (0.5, 1.0, 2.0),
    ) * 5

    anchor_generator = AnchorGenerator(
        sizes=anchor_sizes,
        aspect_ratios=aspect_ratios,
    )

    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_foreground_classes + 1,
        rpn_anchor_generator=anchor_generator,
        min_size=min_size,
        max_size=max_size,
        rpn_pre_nms_top_n_train=rpn_pre_nms_top_n_train,
        rpn_post_nms_top_n_train=rpn_post_nms_top_n_train,
        rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
        rpn_post_nms_top_n_test=rpn_post_nms_top_n_test,
        box_detections_per_img=box_detections_per_img,
    )

    return model
