from __future__ import annotations

import argparse

import torch
import yaml
from torch.utils.data import DataLoader
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
)
from torchvision.models import ResNet50_Weights

from datasets.indicdlp import IndicDLP, collate_fn
from utils.seed import set_seed


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_device(requested: str) -> torch.device:
    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")

        print(
            "CUDA requested but unavailable. "
            "Falling back to CPU."
        )

    return torch.device("cpu")


def main() -> None:

    parser = argparse.ArgumentParser(
        description="IndicDLP -> Faster R-CNN smoke test."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/v0.yaml",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    # ------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------

    set_seed(
        config["reproducibility"]["seed"]
    )

    # ------------------------------------------------------------
    # Device
    # ------------------------------------------------------------

    device = get_device(
        config["smoke_test"]["device"]
    )

    print("=" * 70)
    print("IndicDLP Faster R-CNN Smoke Test")
    print("=" * 70)

    print(f"Device: {device}")

    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # ------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------

    dataset_cfg = config["dataset"]

    dataset = IndicDLP(
        split=dataset_cfg["train_split"],
        bbox_format=dataset_cfg["bbox_format"],
        max_samples=dataset_cfg["max_train_samples"],
        cache_dir=dataset_cfg["cache_dir"],
    )

    num_foreground_classes = len(
        dataset.category_id_to_label
    )

    # Faster R-CNN includes background as class 0.
    num_classes = num_foreground_classes + 1

    print(
        f"Foreground classes: "
        f"{num_foreground_classes}"
    )

    print(
        f"Total model classes: "
        f"{num_classes}"
    )

    # ------------------------------------------------------------
    # DataLoader
    # ------------------------------------------------------------

    loader = DataLoader(
        dataset,
        batch_size=config["smoke_test"]["batch_size"],
        shuffle=False,
        num_workers=config["smoke_test"]["num_workers"],
        collate_fn=collate_fn,
    )

    images, targets = next(iter(loader))

    print(
        f"Batch size: {len(images)}"
    )

    # ------------------------------------------------------------
    # Move data to device
    # ------------------------------------------------------------

    images = [
        image.to(device)
        for image in images
    ]

    targets = [
        {
            key: value.to(device)
            if torch.is_tensor(value)
            else value
            for key, value in target.items()
        }
        for target in targets
    ]

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------

    print("\nBuilding Faster R-CNN...")

    model = fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=ResNet50_Weights.DEFAULT,
        num_classes=num_classes,
        min_size=640,
        max_size=1024,
        rpn_pre_nms_top_n_train=1000,
        rpn_post_nms_top_n_train=1000,
        rpn_pre_nms_top_n_test=500,
        rpn_post_nms_top_n_test=500,
        box_detections_per_img=300,
    )

    model.to(device)

    # ------------------------------------------------------------
    # Training mode
    #
    # Faster R-CNN returns losses when model is in train mode
    # and targets are supplied.
    # ------------------------------------------------------------

    model.train()

    print("Running forward pass...")

    with torch.no_grad():

        losses = model(
            images,
            targets,
        )

    # ------------------------------------------------------------
    # Print losses
    # ------------------------------------------------------------

    print("\nLosses:")

    total_loss = torch.tensor(
        0.0,
        device=device,
    )

    for name, value in losses.items():

        print(
            f"  {name:25s}: "
            f"{value.item():.6f}"
        )

        total_loss = total_loss + value

    print(
        f"\n  {'total_loss':25s}: "
        f"{total_loss.item():.6f}"
    )

    # ------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------

    assert losses, (
        "Model returned no losses."
    )

    for name, value in losses.items():

        assert torch.isfinite(value), (
            f"Non-finite loss detected: {name}"
        )

    assert torch.isfinite(total_loss), (
        "Total loss is not finite."
    )

    print("\n" + "=" * 70)
    print("SMOKE TEST PASSED")
    print("=" * 70)
    print(
        "Dataset -> DataLoader -> Faster R-CNN "
        "forward/loss pipeline is functional."
    )


if __name__ == "__main__":
    main()