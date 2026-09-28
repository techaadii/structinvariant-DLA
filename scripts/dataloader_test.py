from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from datasets.indicdlp import IndicDLP, collate_fn
from utils.seed import set_seed


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test IndicDLP DataLoader."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/v0.yaml",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    seed = config["reproducibility"]["seed"]
    set_seed(seed)

    dataset_cfg = config["dataset"]
    loader_cfg = config["dataloader"]

    print("=" * 70)
    print("IndicDLP DataLoader Test")
    print("=" * 70)

    # ------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------

    dataset = IndicDLP(
        split=dataset_cfg["train_split"],
        bbox_format=dataset_cfg["bbox_format"],
        max_samples=dataset_cfg["max_train_samples"],
        cache_dir=dataset_cfg["cache_dir"],
    )

    print(f"Dataset split : {dataset_cfg['train_split']}")
    print(f"Dataset size  : {len(dataset)}")
    print(
        f"Number classes: "
        f"{len(dataset.category_id_to_label)}"
    )

    # ------------------------------------------------------------
    # DataLoader
    # ------------------------------------------------------------

    loader = DataLoader(
        dataset,
        batch_size=loader_cfg["batch_size"],
        shuffle=loader_cfg["shuffle_train"],
        num_workers=loader_cfg["num_workers"],
        pin_memory=loader_cfg["pin_memory"],
        drop_last=loader_cfg["drop_last"],
        collate_fn=collate_fn,
    )

    print(
        f"Batch size    : "
        f"{loader_cfg['batch_size']}"
    )

    print(
        f"Num workers   : "
        f"{loader_cfg['num_workers']}"
    )

    # ------------------------------------------------------------
    # Fetch one batch
    # ------------------------------------------------------------

    print("\nLoading first batch...")

    images, targets = next(iter(loader))

    print("Batch loaded successfully.")

    # ------------------------------------------------------------
    # Validate batch structure
    # ------------------------------------------------------------

    assert isinstance(images, list), (
        f"Expected images to be list, got {type(images)}"
    )

    assert isinstance(targets, list), (
        f"Expected targets to be list, got {type(targets)}"
    )

    assert len(images) == len(targets), (
        "Number of images and targets must match."
    )

    # ------------------------------------------------------------
    # Inspect every sample
    # ------------------------------------------------------------

    print("\nBatch contents:")
    print("-" * 70)

    for i, (image, target) in enumerate(
        zip(images, targets)
    ):
        print(f"\nSample {i}")

        print(
            f"  Image shape : {tuple(image.shape)}"
        )

        print(
            f"  Image dtype : {image.dtype}"
        )

        print(
            f"  Image range : "
            f"[{image.min().item():.4f}, "
            f"{image.max().item():.4f}]"
        )

        boxes = target["boxes"]
        labels = target["labels"]

        print(
            f"  Boxes shape : {tuple(boxes.shape)}"
        )

        print(
            f"  Labels shape: {tuple(labels.shape)}"
        )

        print(
            f"  Num boxes   : {len(boxes)}"
        )

        print(
            f"  Box dtype   : {boxes.dtype}"
        )

        print(
            f"  Label dtype : {labels.dtype}"
        )

        # --------------------------------------------------------
        # Shape checks
        # --------------------------------------------------------

        assert image.ndim == 3
        assert image.shape[0] == 3

        assert boxes.ndim == 2
        assert boxes.shape[1] == 4

        assert labels.ndim == 1

        assert len(boxes) == len(labels)

        # --------------------------------------------------------
        # Value checks
        # --------------------------------------------------------

        assert torch.isfinite(image).all()
        assert torch.isfinite(boxes).all()

        assert image.min() >= 0.0
        assert image.max() <= 1.0

        # --------------------------------------------------------
        # Bounding-box checks
        # --------------------------------------------------------

        if len(boxes) > 0:

            height = image.shape[1]
            width = image.shape[2]

            assert (boxes[:, 0] >= 0).all()
            assert (boxes[:, 1] >= 0).all()

            assert (boxes[:, 2] <= width).all()
            assert (boxes[:, 3] <= height).all()

            assert (
                boxes[:, 2] > boxes[:, 0]
            ).all()

            assert (
                boxes[:, 3] > boxes[:, 1]
            ).all()

            # Faster R-CNN labels must be > 0.
            assert (labels > 0).all()

        # --------------------------------------------------------
        # Target keys
        # --------------------------------------------------------

        required_keys = {
            "boxes",
            "labels",
            "image_id",
            "area",
            "iscrowd",
        }

        assert required_keys.issubset(
            target.keys()
        )

    print("\n" + "=" * 70)
    print("DATA LOADER TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()