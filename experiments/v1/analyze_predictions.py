from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from dataset.indicdlp import IndicDLP, collate_fn
from models.baseline import build_v1_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="IoU-based qualitative analysis of V1 predictions."
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/v1/last.pt",
    )

    parser.add_argument(
        "--num-images",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outputs/v1/tpfpfn",
    )

    return parser.parse_args()


# ---------------------------------------------------------
# Class colours
# ---------------------------------------------------------

CLASS_COLORS = [
    "#e6194b",
    "#3cb44b",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#bcf60c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#9a6324",
    "#fffac8",
    "#800000",
    "#aaffc3",
    "#808000",
    "#ffd8b1",
    "#000075",
    "#808080",
    "#e6194b",
    "#3cb44b",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#bcf60c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#9a6324",
    "#fffac8",
    "#800000",
    "#aaffc3",
    "#808000",
    "#ffd8b1",
    "#000075",
    "#808080",
    "#e6194b",
    "#3cb44b",
]


def get_class_color(label: int):
    if label <= 0:
        return "#000000"

    return CLASS_COLORS[(label - 1) % len(CLASS_COLORS)]


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

def load_model(
    checkpoint_path: str,
    device: torch.device,
):
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

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
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

    model.to(device)
    model.eval()

    return model


# ---------------------------------------------------------
# IoU
# ---------------------------------------------------------

def box_iou(boxes1, boxes2):
    """
    Compute pairwise IoU.

    boxes1: [N, 4]
    boxes2: [M, 4]

    Returns:
        IoU matrix [N, M]
    """

    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros(
            (len(boxes1), len(boxes2)),
            dtype=torch.float32,
        )

    area1 = (
        (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0)
        *
        (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    )

    area2 = (
        (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0)
        *
        (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    )

    top_left = torch.maximum(
        boxes1[:, None, :2],
        boxes2[None, :, :2],
    )

    bottom_right = torch.minimum(
        boxes1[:, None, 2:],
        boxes2[None, :, 2:],
    )

    wh = (
        bottom_right - top_left
    ).clamp(min=0)

    intersection = wh[:, :, 0] * wh[:, :, 1]

    union = (
        area1[:, None]
        +
        area2[None, :]
        -
        intersection
    )

    return intersection / union.clamp(min=1e-8)


# ---------------------------------------------------------
# TP / FP / FN matching
# ---------------------------------------------------------

def match_predictions(
    gt_boxes,
    gt_labels,
    pred_boxes,
    pred_labels,
    pred_scores,
    iou_threshold,
):
    """
    Greedy one-to-one matching.

    Predictions are processed from highest confidence
    to lowest confidence.

    A prediction is a TP only when:
      1. class matches
      2. IoU >= threshold
      3. GT box has not already been matched
    """

    num_gt = len(gt_boxes)
    num_pred = len(pred_boxes)

    matched_gt = set()

    true_positives = []
    false_positives = []

    # Process predictions by descending confidence.
    order = torch.argsort(
        pred_scores,
        descending=True,
    )

    ious = box_iou(
        pred_boxes,
        gt_boxes,
    )

    for pred_index in order.tolist():

        pred_label = int(
            pred_labels[pred_index]
        )

        best_iou = 0.0
        best_gt_index = None

        for gt_index in range(num_gt):

            if gt_index in matched_gt:
                continue

            gt_label = int(
                gt_labels[gt_index]
            )

            if pred_label != gt_label:
                continue

            current_iou = float(
                ious[pred_index, gt_index]
            )

            if current_iou > best_iou:
                best_iou = current_iou
                best_gt_index = gt_index

        if (
            best_gt_index is not None
            and best_iou >= iou_threshold
        ):
            matched_gt.add(best_gt_index)

            true_positives.append(
                {
                    "pred_index": pred_index,
                    "gt_index": best_gt_index,
                    "iou": best_iou,
                }
            )

        else:
            false_positives.append(
                {
                    "pred_index": pred_index,
                }
            )

    false_negatives = [
        gt_index
        for gt_index in range(num_gt)
        if gt_index not in matched_gt
    ]

    return (
        true_positives,
        false_positives,
        false_negatives,
    )


# ---------------------------------------------------------
# Drawing
# ---------------------------------------------------------

def draw_box(
    ax,
    box,
    label,
    color,
    text,
    linewidth=2.5,
    linestyle="-",
):
    x1, y1, x2, y2 = [
        float(value)
        for value in box
    ]

    rectangle = patches.Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        fill=False,
        edgecolor=color,
        linewidth=linewidth,
        linestyle=linestyle,
    )

    ax.add_patch(rectangle)

    ax.text(
        x1,
        max(y1 - 2, 0),
        text,
        fontsize=7,
        color="white",
        backgroundcolor=color,
        bbox=dict(
            alpha=0.85,
            pad=1.5,
            edgecolor="none",
        ),
    )


def draw_analysis(
    ax,
    image_np,
    gt_boxes,
    gt_labels,
    pred_boxes,
    pred_labels,
    pred_scores,
    matches,
    false_positives,
    false_negatives,
):
    ax.imshow(image_np)

    # ---------------------------------------------
    # True positives
    # ---------------------------------------------

    for match in matches:

        pred_index = match["pred_index"]
        gt_index = match["gt_index"]
        iou = match["iou"]

        label = int(
            pred_labels[pred_index]
        )

        color = get_class_color(label)

        draw_box(
            ax,
            pred_boxes[pred_index],
            label,
            color,
            f"TP C{label} "
            f"{float(pred_scores[pred_index]):.2f} "
            f"IoU {iou:.2f}",
            linewidth=3.0,
        )

    # ---------------------------------------------
    # False positives
    # ---------------------------------------------

    for item in false_positives:

        pred_index = item["pred_index"]

        label = int(
            pred_labels[pred_index]
        )

        draw_box(
            ax,
            pred_boxes[pred_index],
            label,
            "#ff0000",
            f"FP C{label} "
            f"{float(pred_scores[pred_index]):.2f}",
            linewidth=2.5,
            linestyle="--",
        )

    # ---------------------------------------------
    # False negatives
    # ---------------------------------------------

    for gt_index in false_negatives:

        label = int(
            gt_labels[gt_index]
        )

        draw_box(
            ax,
            gt_boxes[gt_index],
            label,
            "#000000",
            f"FN C{label}",
            linewidth=3.0,
            linestyle=":",
        )

    ax.axis("off")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    args = parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    output_dir = Path(
        args.output
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset = IndicDLP(
        split="validation",
        bbox_format="xywh",
        max_samples=args.num_images,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
    )

    model = load_model(
        args.checkpoint,
        device,
    )

    print("=" * 70)
    print("V1 TP / FP / FN ANALYSIS")
    print("=" * 70)
    print(
        f"Images          : {len(dataset)}"
    )
    print(
        f"Score threshold : {args.score_threshold:.2f}"
    )
    print(
        f"IoU threshold   : {args.iou_threshold:.2f}"
    )
    print()

    all_results = []

    total_tp = 0
    total_fp = 0
    total_fn = 0

    with torch.no_grad():

        for index, (
            images,
            targets,
        ) in enumerate(dataloader):

            image = images[0]

            prediction = model(
                [
                    image.to(device)
                ]
            )[0]

            # -----------------------------------------
            # Move everything to CPU
            # -----------------------------------------

            gt_boxes = (
                targets[0]["boxes"]
                .cpu()
            )

            gt_labels = (
                targets[0]["labels"]
                .cpu()
            )

            pred_boxes = (
                prediction["boxes"]
                .detach()
                .cpu()
            )

            pred_labels = (
                prediction["labels"]
                .detach()
                .cpu()
            )

            pred_scores = (
                prediction["scores"]
                .detach()
                .cpu()
            )

            # -----------------------------------------
            # Confidence filtering
            # -----------------------------------------

            keep = (
                pred_scores
                >= args.score_threshold
            )

            pred_boxes = pred_boxes[keep]
            pred_labels = pred_labels[keep]
            pred_scores = pred_scores[keep]

            # -----------------------------------------
            # Match predictions
            # -----------------------------------------

            (
                matches,
                false_positives,
                false_negatives,
            ) = match_predictions(
                gt_boxes=gt_boxes,
                gt_labels=gt_labels,
                pred_boxes=pred_boxes,
                pred_labels=pred_labels,
                pred_scores=pred_scores,
                iou_threshold=args.iou_threshold,
            )

            tp = len(matches)
            fp = len(false_positives)
            fn = len(false_negatives)

            total_tp += tp
            total_fp += fp
            total_fn += fn

            precision = (
                tp / (tp + fp)
                if (tp + fp) > 0
                else 0.0
            )

            recall = (
                tp / (tp + fn)
                if (tp + fn) > 0
                else 0.0
            )

            matched_ious = [
                match["iou"]
                for match in matches
            ]

            mean_iou = (
                sum(matched_ious)
                / len(matched_ious)
                if matched_ious
                else 0.0
            )

            # -----------------------------------------
            # Save visualization
            # -----------------------------------------

            image_np = (
                image
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )

            fig, ax = plt.subplots(
                figsize=(12, 10)
            )

            draw_analysis(
                ax=ax,
                image_np=image_np,
                gt_boxes=gt_boxes,
                gt_labels=gt_labels,
                pred_boxes=pred_boxes,
                pred_labels=pred_labels,
                pred_scores=pred_scores,
                matches=matches,
                false_positives=false_positives,
                false_negatives=false_negatives,
            )

            ax.set_title(
                f"V1 Analysis | "
                f"TP={tp} FP={fp} FN={fn} | "
                f"P={precision:.2f} "
                f"R={recall:.2f} "
                f"mIoU={mean_iou:.2f}",
                fontsize=13,
                fontweight="bold",
            )

            plt.tight_layout()

            output_file = (
                output_dir
                / f"sample_{index:04d}.png"
            )

            plt.savefig(
                output_file,
                dpi=150,
                bbox_inches="tight",
            )

            plt.close(fig)

            # -----------------------------------------
            # Save numerical result
            # -----------------------------------------

            all_results.append(
                {
                    "image_index": index,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "precision": precision,
                    "recall": recall,
                    "mean_matched_iou": mean_iou,
                }
            )

            print(
                f"{index + 1}/{len(dataset)} "
                f"-> TP={tp} "
                f"FP={fp} "
                f"FN={fn} "
                f"P={precision:.3f} "
                f"R={recall:.3f} "
                f"mIoU={mean_iou:.3f}"
            )

    # -----------------------------------------------------
    # Overall metrics
    # -----------------------------------------------------

    overall_precision = (
        total_tp / (total_tp + total_fp)
        if (total_tp + total_fp) > 0
        else 0.0
    )

    overall_recall = (
        total_tp / (total_tp + total_fn)
        if (total_tp + total_fn) > 0
        else 0.0
    )

    overall_f1 = (
        2
        * overall_precision
        * overall_recall
        / (overall_precision + overall_recall)
        if (overall_precision + overall_recall) > 0
        else 0.0
    )

    mean_image_precision = (
        sum(
            result["precision"]
            for result in all_results
        )
        / len(all_results)
        if all_results
        else 0.0
    )

    mean_image_recall = (
        sum(
            result["recall"]
            for result in all_results
        )
        / len(all_results)
        if all_results
        else 0.0
    )

    mean_image_iou = (
        sum(
            result["mean_matched_iou"]
            for result in all_results
        )
        / len(all_results)
        if all_results
        else 0.0
    )

    results = {
        "checkpoint": args.checkpoint,
        "num_images": len(dataset),
        "score_threshold": args.score_threshold,
        "iou_threshold": args.iou_threshold,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "overall_precision": overall_precision,
        "overall_recall": overall_recall,
        "overall_f1": overall_f1,
        "mean_image_precision": mean_image_precision,
        "mean_image_recall": mean_image_recall,
        "mean_image_matched_iou": mean_image_iou,
        "per_image": all_results,
    }

    results_file = (
        output_dir
        / "tpfpfn_results.json"
    )

    results_file.write_text(
        json.dumps(
            results,
            indent=2,
        )
    )

    print()
    print("=" * 70)
    print("OVERALL TP / FP / FN RESULTS")
    print("=" * 70)
    print(
        f"True positives  : {total_tp}"
    )
    print(
        f"False positives : {total_fp}"
    )
    print(
        f"False negatives : {total_fn}"
    )
    print(
        f"Precision       : {overall_precision:.4f}"
    )
    print(
        f"Recall          : {overall_recall:.4f}"
    )
    print(
        f"F1              : {overall_f1:.4f}"
    )
    print(
        f"Mean image IoU  : {mean_image_iou:.4f}"
    )
    print()
    print(
        f"Results saved to: {results_file}"
    )

if __name__ == "__main__":
    main()