from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from core.evaluator import evaluate_map
from dataset.indicdlp import IndicDLP, collate_fn
from models.baseline import build_v1_model


def parse_args():

    parser = argparse.ArgumentParser(
        description="Evaluate V1 Faster R-CNN using COCO-style mAP."
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/v1/last.pt",
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

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outputs/v1/map_results.json",
    )

    return parser.parse_args()


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: torch.device,
) -> torch.nn.Module:

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    print(
        f"Checkpoint type: {type(checkpoint)}"
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        elif "model" in checkpoint:
            state_dict = checkpoint["model"]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    return model


def main():

    args = parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("StructInvariant-DLA V1 Evaluation")
    print("=" * 70)

    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    checkpoint_path = Path(
        args.checkpoint
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    dataset = IndicDLP(
        split="validation",
        bbox_format="xywh",
        max_samples=args.val_samples,
    )

    print(
        f"Validation images: {len(dataset)}"
    )

    # --------------------------------------------------
    # DataLoader
    # --------------------------------------------------

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    print(
        f"Evaluation batch size: {args.batch_size}"
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = build_v1_model(
        num_foreground_classes=42,
        pretrained_backbone=False,
        min_size=320,
        max_size=512,
        rpn_pre_nms_top_n_train=1000,
        rpn_post_nms_top_n_train=1000,
        rpn_pre_nms_top_n_test=500,
        rpn_post_nms_top_n_test=500,
        box_detections_per_img=300,
    )

    model = load_checkpoint(
        model,
        str(checkpoint_path),
        device,
    )

    model.to(device)
    model.eval()

    print(
        f"Loaded checkpoint: {checkpoint_path}"
    )

    # --------------------------------------------------
    # GPU memory before evaluation
    # --------------------------------------------------

    if torch.cuda.is_available():

        torch.cuda.reset_peak_memory_stats()

        print(
            "GPU memory allocated before evaluation:",
            round(
                torch.cuda.memory_allocated()
                / 1024**2,
                2,
            ),
            "MB",
        )

    # --------------------------------------------------
    # Evaluate
    # --------------------------------------------------

    start_time = time.perf_counter()

    results = evaluate_map(
        model=model,
        dataloader=dataloader,
        device=device,
        print_freq=10,
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    results["evaluation_time_sec"] = elapsed

    results["images_per_sec"] = (
        len(dataset) / elapsed
    )

    # --------------------------------------------------
    # GPU statistics
    # --------------------------------------------------

    if torch.cuda.is_available():

        results["peak_gpu_memory_allocated_mb"] = (
            torch.cuda.max_memory_allocated()
            / 1024**2
        )

        results["peak_gpu_memory_reserved_mb"] = (
            torch.cuda.max_memory_reserved()
            / 1024**2
        )

    # --------------------------------------------------
    # Main metrics
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("QUANTITATIVE RESULTS")
    print("=" * 70)

    metrics = [
        ("mAP@[0.50:0.95]", "map"),
        ("mAP@0.50", "map_50"),
        ("mAP@0.75", "map_75"),
        ("mAR@1", "mar_1"),
        ("mAR@10", "mar_10"),
        ("mAR@100", "mar_100"),
    ]

    for display_name, key in metrics:

        if key in results:

            print(
                f"{display_name:20s}: "
                f"{results[key]:.4f}"
            )

    print()
    print(
        f"Ground-truth boxes : "
        f"{results['num_ground_truth_boxes']}"
    )

    print(
        f"Predicted boxes    : "
        f"{results['num_predictions']}"
    )

    print(
        f"Evaluation time    : "
        f"{elapsed:.2f} sec"
    )

    print(
        f"Images/sec         : "
        f"{results['images_per_sec']:.2f}"
    )

    if torch.cuda.is_available():

        print(
            f"Peak GPU allocated : "
            f"{results['peak_gpu_memory_allocated_mb']:.2f} MB"
        )

        print(
            f"Peak GPU reserved  : "
            f"{results['peak_gpu_memory_reserved_mb']:.2f} MB"
        )

    # --------------------------------------------------
    # Per-class AP
    # --------------------------------------------------

    if "map_per_class" in results:

        print()
        print("=" * 70)
        print("PER-CLASS AP")
        print("=" * 70)

        ap_values = results["map_per_class"]

        classes = results.get(
            "classes",
            torch.arange(
                len(ap_values)
            ),
        )

        for class_id, ap in zip(
            classes,
            ap_values,
        ):

            print(
                f"Class {int(class_id):3d} "
                f"AP: {float(ap):.4f}"
            )

    # --------------------------------------------------
    # Save JSON
    # --------------------------------------------------

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_results = {}

    for key, value in results.items():

        if isinstance(value, torch.Tensor):

            if value.numel() == 1:
                json_results[key] = float(
                    value.item()
                )
            else:
                json_results[key] = (
                    value.tolist()
                )

        else:
            json_results[key] = value

    output_path.write_text(
        json.dumps(
            json_results,
            indent=2,
        )
    )

    print()
    print(
        f"Results saved to: {output_path}"
    )


if __name__ == "__main__":
    main()