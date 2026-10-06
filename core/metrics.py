from __future__ import annotations

from typing import Any

import torch
from torchmetrics.detection import MeanAveragePrecision


class DetectionMetrics:
    def __init__(self, device: torch.device):
        self.metric = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
        ).to(device)

    @torch.no_grad()
    def update(self, predictions, targets):
        preds = []

        for prediction in predictions:
            preds.append(
                {
                    "boxes": prediction["boxes"].detach().cpu(),
                    "scores": prediction["scores"].detach().cpu(),
                    "labels": prediction["labels"].detach().cpu(),
                }
            )

        targs = []

        for target in targets:
            targs.append(
                {
                    "boxes": target["boxes"].detach().cpu(),
                    "labels": target["labels"].detach().cpu(),
                }
            )

        self.metric.update(preds, targs)

    def compute(self) -> dict[str, Any]:
        result = self.metric.compute()

        return {
            key: (
                float(value.item())
                if isinstance(value, torch.Tensor) and value.numel() == 1
                else value
            )
            for key, value in result.items()
        }

    def reset(self):
        self.metric.reset()
