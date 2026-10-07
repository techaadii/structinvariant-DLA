from __future__ import annotations

from typing import Any

import torch
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm


@torch.no_grad()
def evaluate_map(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    print_freq: int = 10,
) -> dict[str, Any]:
    """
    Evaluate an object detection model using COCO-style bounding-box mAP.

    Metrics:
        map       : mAP averaged over IoU 0.50:0.95
        map_50    : mAP at IoU 0.50
        map_75    : mAP at IoU 0.75
        mar_1     : mean average recall with max 1 detection/image
        mar_10    : mean average recall with max 10 detections/image
        mar_100   : mean average recall with max 100 detections/image

    Also returns:
        map_per_class
        classes
        number of GT boxes
        number of predictions
    """

    model.eval()

    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        class_metrics=True,
    )

    total_batches = len(dataloader)

    total_gt = 0
    total_predictions = 0

    progress = tqdm(
        dataloader,
        desc="Validation mAP",
        leave=False,
    )

    for batch_idx, (images, targets) in enumerate(
        progress,
        start=1,
    ):
        images = [
            image.to(
                device,
                non_blocking=True,
            )
            for image in images
        ]

        predictions = model(images)

        preds = []
        target_list = []

        for prediction, target in zip(
            predictions,
            targets,
        ):
            pred_boxes = (
                prediction["boxes"]
                .detach()
                .cpu()
            )

            pred_scores = (
                prediction["scores"]
                .detach()
                .cpu()
            )

            pred_labels = (
                prediction["labels"]
                .detach()
                .cpu()
            )

            target_boxes = (
                target["boxes"]
                .detach()
                .cpu()
            )

            target_labels = (
                target["labels"]
                .detach()
                .cpu()
            )

            preds.append(
                {
                    "boxes": pred_boxes,
                    "scores": pred_scores,
                    "labels": pred_labels,
                }
            )

            target_list.append(
                {
                    "boxes": target_boxes,
                    "labels": target_labels,
                }
            )

            total_gt += len(target_boxes)
            total_predictions += len(pred_boxes)

        metric.update(
            preds,
            target_list,
        )

        if (
            batch_idx % print_freq == 0
            or batch_idx == total_batches
        ):
            progress.set_postfix(
                batch=f"{batch_idx}/{total_batches}"
            )

    results = metric.compute()

    output: dict[str, Any] = {}

    for key, value in results.items():

        if isinstance(value, torch.Tensor):

            if value.numel() == 1:
                output[key] = float(
                    value.item()
                )
            else:
                output[key] = (
                    value.detach().cpu()
                )

        else:
            output[key] = value

    output["num_ground_truth_boxes"] = total_gt
    output["num_predictions"] = total_predictions

    return output