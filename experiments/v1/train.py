from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from core.trainer import Trainer
from dataset.indicdlp import (
    IndicDLP,
    collate_fn,
)
from logging_utils.wandb_logger import (
    WandBLogger,
)
from models.baseline import (
    build_v1_model,
)


def set_seed(seed: int):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    set_seed(
        config["experiment"]["seed"]
    )

    device = torch.device(
        "cuda"
        if (
            config["device"]["type"]
            == "cuda"
            and torch.cuda.is_available()
        )
        else "cpu"
    )

    print("=" * 70)
    print("StructInvariant-DLA V1")
    print("=" * 70)

    print(
        f"Device: {device}"
    )

    if device.type == "cuda":

        print(
            "GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            "VRAM: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    train_dataset = IndicDLP(
        split=config[
            "dataset"
        ]["train_split"],

        bbox_format=config[
            "dataset"
        ]["bbox_format"],

        max_samples=config[
            "dataset"
        ]["max_train_samples"],

        cache_dir=config[
            "dataset"
        ]["cache_dir"],
    )

    val_dataset = IndicDLP(
        split=config[
            "dataset"
        ]["val_split"],

        bbox_format=config[
            "dataset"
        ]["bbox_format"],

        max_samples=config[
            "dataset"
        ]["max_val_samples"],

        cache_dir=config[
            "dataset"
        ]["cache_dir"],

        category_id_to_label=(
            train_dataset
            .category_id_to_label
        ),
    )

    print(
        f"Train images: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation images: "
        f"{len(val_dataset)}"
    )

    print(
        f"Foreground classes: "
        f"{len(train_dataset.category_id_to_label)}"
    )

    # ---------------------------------------------------------
    # Save category mapping
    # ---------------------------------------------------------

    train_dataset.save_category_mapping(
        config["outputs"][
            "category_mapping"
        ]
    )

    # ---------------------------------------------------------
    # DataLoaders
    # ---------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=config[
            "training"
        ]["batch_size"],
        shuffle=True,
        num_workers=config[
            "training"
        ]["num_workers"],
        pin_memory=config[
            "training"
        ]["pin_memory"],
        drop_last=config[
            "training"
        ]["drop_last"],
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
    val_dataset,
    batch_size=config["evaluation"]["batch_size"],
    shuffle=False,
    num_workers=config["evaluation"]["num_workers"],
    pin_memory=config["training"]["pin_memory"],
    collate_fn=collate_fn,
)

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = build_v1_model(
        num_foreground_classes=config[
            "model"
        ]["num_foreground_classes"],

        pretrained_backbone=config[
            "model"
        ]["pretrained_backbone"],

        min_size=config[
            "model"
        ]["min_size"],

        max_size=config[
            "model"
        ]["max_size"],

        rpn_pre_nms_top_n_train=config[
            "model"
        ]["rpn_pre_nms_top_n_train"],

        rpn_post_nms_top_n_train=config[
            "model"
        ]["rpn_post_nms_top_n_train"],

        rpn_pre_nms_top_n_test=config[
            "model"
        ]["rpn_pre_nms_top_n_test"],

        rpn_post_nms_top_n_test=config[
            "model"
        ]["rpn_post_nms_top_n_test"],

        box_detections_per_img=config[
            "model"
        ]["box_detections_per_img"],
    )

    model.to(device)

    # ---------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------

    optimizer = SGD(
        model.parameters(),
        lr=config[
            "training"
        ]["learning_rate"],
        momentum=config[
            "training"
        ]["momentum"],
        weight_decay=config[
            "training"
        ]["weight_decay"],
    )

    scheduler = StepLR(
        optimizer,
        step_size=config[
            "training"
        ]["step_size"],
        gamma=config[
            "training"
        ]["gamma"],
    )

    # ---------------------------------------------------------
    # W&B
    # ---------------------------------------------------------

    logger = WandBLogger(
        enabled=config[
            "wandb"
        ]["enabled"],

        project=config[
            "wandb"
        ]["project"],

        entity=config[
            "wandb"
        ]["entity"],


        tags=config[
            "wandb"
        ]["tags"],

        config=config,
    )

    # ---------------------------------------------------------
    # Trainer
    # ---------------------------------------------------------

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config,
        logger=logger,
    )

    try:

        trainer.fit()

    finally:

        logger.finish()


if __name__ == "__main__":
    main()