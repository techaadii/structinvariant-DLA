from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from logging_utils.system_monitor import GPUMonitor
from utils.checkpoint import save_checkpoint


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        device,
        config,
        logger,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = device
        self.config = config
        self.logger = logger

        self.amp_enabled = (
            config["training"]["amp"]
            and device.type == "cuda"
        )

        self.scaler = GradScaler(
            "cuda",
            enabled=self.amp_enabled,
        )

        self.gpu_monitor = GPUMonitor(device)

        self.global_step = 0

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
        }

        self.best_metric = float("-inf")

    def _move_targets(self, targets):
        return [
            {
                key: value.to(self.device)
                if torch.is_tensor(value)
                else value
                for key, value in target.items()
            }
            for target in targets
        ]

    def train_one_epoch(self, epoch: int):
        self.model.train()

        epoch_loss = 0.0
        start_epoch = time.perf_counter()

        progress = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}",
        )

        for batch_idx, (images, targets) in enumerate(progress):
            batch_start = time.perf_counter()

            images = [
                image.to(self.device, non_blocking=True)
                for image in images
            ]

            targets = self._move_targets(targets)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(
                device_type="cuda",
                enabled=self.amp_enabled,
            ):
                loss_dict = self.model(images, targets)
                total_loss = sum(loss_dict.values())

            if not torch.isfinite(total_loss):
                raise RuntimeError(
                    f"Non-finite loss at epoch={epoch}, "
                    f"batch={batch_idx}: {total_loss.item()}"
                )

            self.scaler.scale(total_loss).backward()

            if self.config["training"]["grad_clip"] is not None:
                self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config["training"]["grad_clip"],
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_time = time.perf_counter() - batch_start

            batch_size = len(images)
            images_per_second = batch_size / batch_time

            epoch_loss += total_loss.item()

            metrics = {
                "train/batch_loss": total_loss.item(),
                "train/loss_classifier": loss_dict[
                    "loss_classifier"
                ].item(),
                "train/loss_box_reg": loss_dict[
                    "loss_box_reg"
                ].item(),
                "train/loss_objectness": loss_dict[
                    "loss_objectness"
                ].item(),
                "train/loss_rpn_box_reg": loss_dict[
                    "loss_rpn_box_reg"
                ].item(),
                "train/learning_rate": self.optimizer.param_groups[
                    0
                ]["lr"],
                "train/batch_time_sec": batch_time,
                "train/images_per_sec": images_per_second,
                "train/epoch": epoch,
                "train/batch": batch_idx,
            }

            metrics.update(self.gpu_monitor.get_metrics())

            self.logger.log(
                metrics,
                step=self.global_step,
            )

            self.global_step += 1

            progress.set_postfix(
                loss=f"{total_loss.item():.4f}",
                gpu=(
                    f"{metrics.get('system/gpu_utilization_percent', 0):.0f}%"
                ),
            )

        epoch_time = time.perf_counter() - start_epoch

        mean_loss = epoch_loss / len(self.train_loader)

        self.history["train_loss"].append(mean_loss)
        self.history["learning_rate"].append(
            self.optimizer.param_groups[0]["lr"]
        )

        self.logger.log(
            {
                "epoch/train_loss": mean_loss,
                "epoch/time_sec": epoch_time,
                "epoch/learning_rate": self.optimizer.param_groups[
                    0
                ]["lr"],
            },
            step=self.global_step,
        )

        return mean_loss

    @torch.no_grad()
    def validate(self, epoch: int):
        self.model.train()

        total_loss = 0.0

        for images, targets in tqdm(
            self.val_loader,
            desc="Validation",
        ):
            images = [
                image.to(self.device, non_blocking=True)
                for image in images
            ]

            targets = self._move_targets(targets)

            loss_dict = self.model(images, targets)

            loss = sum(loss_dict.values())

            total_loss += loss.item()

        mean_loss = total_loss / len(self.val_loader)

        self.history["val_loss"].append(mean_loss)

        self.logger.log(
            {
                "epoch/val_loss": mean_loss,
                "epoch": epoch,
            },
            step=self.global_step,
        )

        return mean_loss

    def save_history(self):
        path = Path(
            self.config["outputs"]["history"]
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_text(
            json.dumps(
                self.history,
                indent=2,
            )
        )

    def fit(self):
        epochs = self.config["training"]["epochs"]

        checkpoint_dir = Path(
            self.config["checkpoint"]["directory"]
        )

        checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch(epoch)

            val_loss = self.validate(epoch)

            if self.scheduler is not None:
                self.scheduler.step()

            if val_loss < self.best_metric:
                self.best_metric = val_loss

            save_checkpoint(
                checkpoint_dir / "last.pt",
                self.model,
                self.optimizer,
                self.scheduler,
                epoch,
                self.best_metric,
                self.history,
            )

            self.save_history()

            print(
                f"\nEpoch {epoch}/{epochs}"
            )
            print(
                f"Train loss: {train_loss:.6f}"
            )
            print(
                f"Val loss:   {val_loss:.6f}"
            )
            print("time for GPU to cool down")
            time.sleep(120)

        self.gpu_monitor.close()