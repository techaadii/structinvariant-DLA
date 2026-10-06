from __future__ import annotations

from typing import Any

import torch

try:
    import pynvml

    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


class GPUMonitor:
    """Collect lightweight GPU metrics during training."""

    def __init__(self, device: torch.device):
        self.device = device
        self.enabled = (
            device.type == "cuda"
            and torch.cuda.is_available()
            and NVML_AVAILABLE
        )

        self.handle = None

        if self.enabled:
            pynvml.nvmlInit()

            device_index = (
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            )

            self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)

    def get_metrics(self) -> dict[str, Any]:
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return {}

        metrics = {
            "system/gpu_memory_allocated_mb": (
                torch.cuda.memory_allocated(self.device) / 1024**2
            ),
            "system/gpu_memory_reserved_mb": (
                torch.cuda.memory_reserved(self.device) / 1024**2
            ),
            "system/gpu_memory_max_allocated_mb": (
                torch.cuda.max_memory_allocated(self.device) / 1024**2
            ),
        }

        if self.enabled and self.handle is not None:
            utilization = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            temperature = pynvml.nvmlDeviceGetTemperature(
                self.handle,
                pynvml.NVML_TEMPERATURE_GPU,
            )

            metrics.update(
                {
                    "system/gpu_utilization_percent": float(
                        utilization.gpu
                    ),
                    "system/gpu_memory_utilization_percent": float(
                        utilization.memory
                    ),
                    "system/gpu_temperature_c": float(temperature),
                }
            )

        return metrics

    def close(self):
        if self.enabled:
            pynvml.nvmlShutdown()
