from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from dataset.indicdlp import IndicDLP, collate_fn
from models.v2 import build_v2_model
from core.evaluator import evaluate_map


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--config",
        default="configs/v2.yaml",
    )

    parser.add_argument(
        "--val-samples",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
    )

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("StructInvariant-DLA V2 Evaluation")
    print("=" * 70)

    print(f"Device: {device}")

    dataset = IndicDLP(
        split=config["dataset"]["val_split"],
        bbox_format=config["dataset"]["bbox_format"],
        max_samples=args.val_samples,
        cache_dir=config["dataset"]["cache_dir"],
        category_id_to_label=None,
    )

    model = build_v2_model(
        num_foreground_classes=config["model"]["num_foreground_classes"],
        pretrained_backbone=False,
        min_size=config["model"]["min_size"],
        max_size=config["model"]["max_size"],
        rpn_pre_nms_top_n_train=config["model"]["rpn_pre_nms_top_n_train"],
        rpn_post_nms_top_n_train=config["model"]["rpn_post_nms_top_n_train"],
        rpn_pre_nms_top_n_test=config["model"]["rpn_pre_nms_top_n_test"],
        rpn_post_nms_top_n_test=config["model"]["rpn_post_nms_top_n_test"],
        box_detections_per_img=config["model"]["box_detections_per_img"],
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
    )

    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint,
    )

    model.load_state_dict(state_dict)
    model.to(device)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config["evaluation"]["num_workers"],
        pin_memory=config["training"]["pin_memory"],
        collate_fn=collate_fn,
    )

    results = evaluate_map(
        model=model,
        dataloader=loader,
        device=device,
    )

    print("\nV2 Evaluation Results")
    print("=" * 70)

    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key:30s}: {value:.4f}")
        else:
            print(f"{key:30s}: {value}")

    output_path = Path(config["outputs"]["metrics"])
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            results,
            indent=2,
        )
    )

    print(
        f"\nSaved metrics to: {output_path}"
    )


if __name__ == "__main__":
    main()
