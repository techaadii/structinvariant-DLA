from __future__ import annotations

from pathlib import Path
from typing import Any

import wandb


class WandBLogger:
    def __init__(
        self,
        enabled: bool,
        project: str,
        config: dict[str, Any],
        entity: str | None = None,
        tags: list[str] | None = None,
    ):
        self.enabled = enabled

        if not enabled:
            self.run = None
            return

        self.run = wandb.init(
            project=project,
            entity=entity,
            config=config,
            tags=tags or [],
        )

    def log(self, metrics: dict[str, Any], step: int | None = None):
        if self.enabled and self.run is not None:
            wandb.log(metrics, step=step)

    def log_summary(self, metrics: dict[str, Any]):
        if self.enabled and self.run is not None:
            for key, value in metrics.items():
                self.run.summary[key] = value

    def save_file(self, path: str | Path):
        if self.enabled and self.run is not None:
            self.run.save(str(path))

    def finish(self):
        if self.enabled and self.run is not None:
            self.run.finish()