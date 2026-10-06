from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(
    path: str | Path,
    model,
    optimizer,
    scheduler,
    epoch: int,
    best_metric: float,
    history: dict,
):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": (
                scheduler.state_dict()
                if scheduler is not None
                else None
            ),
            "best_metric": best_metric,
            "history": history,
        },
        path,
    )