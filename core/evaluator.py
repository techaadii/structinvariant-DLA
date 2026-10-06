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

    Returns:
        map       : mAP averaged over IoU 0.50:0.95
        map_50    : mAP at IoU 0.50
        map_75    : mAP at IoU 0.75
        mar_1     : mean average recall with max 1 detection/image
        mar_10    : mean average recall with max 10 detections/image
        mar_100   : mean average recall with max 100 detections/image
        map_per_class : AP for each class
        classes      : class IDs corresponding to map_per_class
    """

    model.eval()

    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        class_metrics=True,
    )

    total_batches = len(dataloader)

    progress = tqdm(
        dataloader,
        desc="Validation mAP",
        leave=False,
    )

    for batch_idx, (images, targets) in enumerate(progress, start=1):

        # Move images to GPU.
        images = [
            image.to(device, non_blocking=True)
            for image in images
        ]

        # Run inference.
        predictions = model(images)

        preds = []
        target_list = []

        for prediction, target in zip(predictions, targets):

            # TorchMetrics expects:
            # boxes  -> [N, 4]
            # scores -> [N]
            # labels -> [N]
            preds.append(
                {
                    "boxes": prediction["boxes"].detach().cpu(),
                    "scores": prediction["scores"].detach().cpu(),
                    "labels": prediction["labels"].detach().cpu(),
                }
            )

            target_list.append(
                {
                    "boxes": target["boxes"].detach().cpu(),
                    "labels": target["labels"].detach().cpu(),
                }
            )

        # Add this batch to the metric.
        metric.update(preds, target_list)

        if batch_idx % print_freq == 0 or batch_idx == total_batches:
            progress.set_postfix(
                batch=f"{batch_idx}/{total_batches}"
            )

    # Calculate final metrics.
    results = metric.compute()

    # Convert scalar tensors to Python floats while keeping
    # per-class tensors available.
    output: dict[str, Any] = {}

    for key, value in results.items():

        if isinstance(value, torch.Tensor):

            if value.numel() == 1:
                output[key] = float(value.item())
            else:
                output[key] = value.detach().cpu()

        else:
            output[key] = value

    return output