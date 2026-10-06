from __future__ import annotations

from torchvision.models import ResNet50_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.resnet import resnet50


def build_v1_model(
    num_foreground_classes: int,
    pretrained_backbone: bool = True,
    min_size: int = 640,
    max_size: int = 1024,
    rpn_pre_nms_top_n_train: int = 1000,
    rpn_post_nms_top_n_train: int = 1000,
    rpn_pre_nms_top_n_test: int = 500,
    rpn_post_nms_top_n_test: int = 500,
    box_detections_per_img: int = 300,
):
    """
    V1 baseline:

        Image
          ↓
        ResNet-50
          ↓
        FPN
          ↓
        Faster R-CNN
          ↓
        42 IndicDLP foreground classes

    Background is class 0.
    """

    weights = (
        ResNet50_Weights.DEFAULT
        if pretrained_backbone
        else None
    )

    backbone = resnet50(weights=weights)

    return_layers = {
        "layer1": "0",
        "layer2": "1",
        "layer3": "2",
        "layer4": "3",
    }

    backbone = BackboneWithFPN(
        backbone,
        return_layers=return_layers,
        in_channels_list=[
            256,
            512,
            1024,
            2048,
        ],
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

    num_classes = num_foreground_classes + 1

    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
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