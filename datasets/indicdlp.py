from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image as PILImage
from datasets import load_dataset


class IndicDLP:
    """
    Hugging Face adapter for ai4bharat/indicdlp.

    Expected Hugging Face columns:
        image
        bboxes
        category_ids

    Bounding boxes can be provided in either:
        xywh = [x, y, width, height]
        xyxy = [x1, y1, x2, y2]

    Internally, all boxes are converted to:
        [x1, y1, x2, y2]

    Labels are converted from the original IndicDLP category IDs
    into contiguous labels starting from 1.

    Example:
        original category IDs:
            [0, 3, 7, 12]

        internal labels:
            [1, 2, 3, 4]

    Label 0 is reserved for background by torchvision detection models.
    """

    def __init__(
        self,
        split: str,
        bbox_format: str = "xywh",
        max_samples: int | None = None,
        cache_dir: str | None = None,
        category_id_to_label: dict[int, int] | None = None,
    ):
        if bbox_format not in {"xywh", "xyxy"}:
            raise ValueError(
                "bbox_format must be 'xywh' or 'xyxy'"
            )

        # ---------------------------------------------------------
        # Load Hugging Face dataset
        # ---------------------------------------------------------
        kwargs: dict[str, Any] = {
            "path": "ai4bharat/indicdlp",
            "split": split,
        }

        if cache_dir is not None:
            kwargs["cache_dir"] = cache_dir

        self.hf = load_dataset(**kwargs)

        self.split = split
        self.bbox_format = bbox_format
        self.max_samples = max_samples

        # ---------------------------------------------------------
        # Category mapping
        # ---------------------------------------------------------
        #
        # For training:
        #     discover all category IDs and create mapping.
        #
        # For validation/test:
        #     receive the mapping created from training.
        #
        if category_id_to_label is None:
            self._category_ids = self._discover_category_ids()

            self.category_id_to_label = {
                category_id: label
                for label, category_id in enumerate(
                    self._category_ids,
                    start=1,
                )
            }

        else:
            self.category_id_to_label = {
                int(category_id): int(label)
                for category_id, label in category_id_to_label.items()
            }

            self._category_ids = sorted(
                self.category_id_to_label.keys()
            )

        # Reverse mapping:
        #
        # internal label -> original dataset category ID
        #
        self.label_to_category_id = {
            label: category_id
            for category_id, label in self.category_id_to_label.items()
        }

    # =============================================================
    # Dataset metadata
    # =============================================================

    def _discover_category_ids(self) -> list[int]:
        """
        Discover all unique category IDs without decoding images.

        This is intentionally done using only the category_ids column
        so that dataset initialization does not unnecessarily load
        image pixels.
        """

        category_ids: set[int] = set()

        metadata = self.hf.select_columns(["category_ids"])

        for row in metadata:
            for category_id in row["category_ids"]:
                category_ids.add(int(category_id))

        discovered_ids = sorted(category_ids)

        if not discovered_ids:
            raise RuntimeError(
                "No category_ids found in the dataset."
            )

        return discovered_ids

    def save_category_mapping(
        self,
        path: str | Path,
    ) -> None:
        """
        Save the dataset category mapping as JSON.
        """

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "category_id_to_label": {
                str(category_id): label
                for category_id, label
                in self.category_id_to_label.items()
            },
            "label_to_category_id": {
                str(label): category_id
                for label, category_id
                in self.label_to_category_id.items()
            },
            "num_classes": len(
                self.category_id_to_label
            ),
        }

        path.write_text(
            json.dumps(
                payload,
                indent=2,
            )
        )

    # =============================================================
    # Basic dataset interface
    # =============================================================

    def __len__(self) -> int:
        """
        Return the effective dataset size.

        If max_samples is specified, only the first max_samples
        examples are exposed.
        """

        dataset_size = len(self.hf)

        if self.max_samples is not None:
            dataset_size = min(
                dataset_size,
                self.max_samples,
            )

        return dataset_size

    # =============================================================
    # Image handling
    # =============================================================

    @staticmethod
    def _to_rgb(
        image: Any,
    ) -> PILImage.Image:
        """
        Convert a Hugging Face image value into an RGB PIL image.

        HF Image features can appear as:
            - PIL.Image.Image
            - dict containing bytes
            - dict containing path
        """

        if isinstance(image, PILImage.Image):
            return image.convert("RGB")

        if isinstance(image, dict):

            # Image stored as raw bytes
            if image.get("bytes") is not None:
                return PILImage.open(
                    io.BytesIO(image["bytes"])
                ).convert("RGB")

            # Image stored as a path
            if image.get("path"):
                return PILImage.open(
                    image["path"]
                ).convert("RGB")

        raise TypeError(
            f"Unsupported image type: {type(image)}"
        )

    # =============================================================
    # Annotation handling
    # =============================================================

    def _convert_annotations(
        self,
        boxes: list,
        category_ids: list,
        width: int,
        height: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert raw IndicDLP annotations into tensors.

        Responsibilities:
            1. Pair each box with its category ID.
            2. Convert xywh -> xyxy when required.
            3. Clip coordinates to image boundaries.
            4. Remove malformed boxes.
            5. Remove zero/negative-area boxes.
            6. Convert original category IDs into contiguous labels.

        Returns:
            boxes:
                FloatTensor of shape [N, 4]
                in [x1, y1, x2, y2] format.

            labels:
                LongTensor of shape [N].
        """

        converted_boxes: list[list[float]] = []
        labels: list[int] = []

        # ---------------------------------------------------------
        # Process boxes and labels together.
        #
        # This is important because invalid boxes may be removed.
        # The corresponding label must be removed at the same time.
        # ---------------------------------------------------------

        for box, category_id in zip(
            boxes,
            category_ids,
        ):

            # -----------------------------------------------------
            # Validate box structure
            # -----------------------------------------------------

            if len(box) != 4:
                continue

            try:
                x1, y1, a, b = [
                    float(value)
                    for value in box
                ]
            except (TypeError, ValueError):
                continue

            # -----------------------------------------------------
            # Convert bounding-box format
            # -----------------------------------------------------

            if self.bbox_format == "xywh":
                x2 = x1 + a
                y2 = y1 + b

            else:
                # Already xyxy
                x2 = a
                y2 = b

            # -----------------------------------------------------
            # Clip coordinates to image boundaries
            # -----------------------------------------------------

            x1 = max(
                0.0,
                min(x1, float(width)),
            )

            y1 = max(
                0.0,
                min(y1, float(height)),
            )

            x2 = max(
                0.0,
                min(x2, float(width)),
            )

            y2 = max(
                0.0,
                min(y2, float(height)),
            )

            # -----------------------------------------------------
            # Remove invalid boxes
            # -----------------------------------------------------

            if x2 <= x1 or y2 <= y1:
                continue

            # -----------------------------------------------------
            # Convert original category ID to internal label
            # -----------------------------------------------------

            category_id = int(category_id)

            if category_id not in self.category_id_to_label:
                raise KeyError(
                    f"Category ID {category_id} was not found "
                    "in category_id_to_label."
                )

            label = self.category_id_to_label[
                category_id
            ]

            converted_boxes.append(
                [x1, y1, x2, y2]
            )

            labels.append(label)

        # =========================================================
        # Convert to tensors
        # =========================================================

        if converted_boxes:
            box_tensor = torch.tensor(
                converted_boxes,
                dtype=torch.float32,
            )
        else:
            box_tensor = torch.zeros(
                (0, 4),
                dtype=torch.float32,
            )

        label_tensor = torch.tensor(
            labels,
            dtype=torch.int64,
        )

        return box_tensor, label_tensor

    # =============================================================
    # Dataset item
    # =============================================================

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:

        if index < 0 or index >= len(self):
            raise IndexError(index)

        # ---------------------------------------------------------
        # Retrieve row from Hugging Face dataset
        # ---------------------------------------------------------

        row = self.hf[index]

        # ---------------------------------------------------------
        # Convert image to RGB
        # ---------------------------------------------------------

        image = self._to_rgb(
            row["image"]
        )

        width, height = image.size

        # ---------------------------------------------------------
        # Read raw annotations
        # ---------------------------------------------------------

        raw_boxes = row["bboxes"]
        raw_category_ids = row["category_ids"]

        # ---------------------------------------------------------
        # Convert annotations
        #
        # All box processing happens in ONE place.
        # ---------------------------------------------------------

        boxes, labels = self._convert_annotations(
            boxes=raw_boxes,
            category_ids=raw_category_ids,
            width=width,
            height=height,
        )

        # ---------------------------------------------------------
        # Convert PIL image -> PyTorch tensor
        #
        # PIL:
        #     H x W x C
        #
        # PyTorch:
        #     C x H x W
        #
        # Values:
        #     [0, 255] -> [0, 1]
        # ---------------------------------------------------------

        image_array = np.asarray(
            image,
            dtype=np.uint8,
        )

        image_tensor = (
            torch.from_numpy(image_array)
            .permute(2, 0, 1)
            .float()
            / 255.0
        )

        # ---------------------------------------------------------
        # Calculate bounding-box areas
        # ---------------------------------------------------------

        if len(boxes) > 0:
            area = (
                (boxes[:, 2] - boxes[:, 0])
                * (boxes[:, 3] - boxes[:, 1])
            )
        else:
            area = torch.zeros(
                (0,),
                dtype=torch.float32,
            )

        # ---------------------------------------------------------
        # Crowd annotations
        #
        # IndicDLP annotations are treated as non-crowd.
        # ---------------------------------------------------------

        iscrowd = torch.zeros(
            (len(boxes),),
            dtype=torch.int64,
        )

        # ---------------------------------------------------------
        # Torchvision detection target
        # ---------------------------------------------------------

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(
                [index],
                dtype=torch.int64,
            ),
            "area": area,
            "iscrowd": iscrowd,
        }

        # ---------------------------------------------------------
        # Return everything useful for the project
        # ---------------------------------------------------------

        return {
            "image": image_tensor,
            "target": target,
            "width": width,
            "height": height,
            "raw": row,
        }


# ================================================================
# DataLoader collate function
# ================================================================

def collate_fn(batch):
    """
    Collate function for torchvision detection models.

    Detection images can have different spatial dimensions and
    each image can contain a different number of bounding boxes.

    Therefore, we keep them as lists instead of stacking them.
    """

    images = [
        item["image"]
        for item in batch
    ]

    targets = [
        item["target"]
        for item in batch
    ]

    return images, targets