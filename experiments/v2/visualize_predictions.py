from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from torch.utils.data import DataLoader

from dataset.indicdlp import IndicDLP, collate_fn
from models.v2 import build_v2_model


def parse_args():

    parser = argparse.ArgumentParser(
        description="Generate qualitative V1 predictions."
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/v1/last.pt",
    )

    parser.add_argument(
        "--num-images",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outputs/v1/qualitative",
    )

    return parser.parse_args()


def load_model(
    checkpoint_path: str,
    device: torch.device,
):

    model = build_v2_model(
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
# Colour palette
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
    """
    Return a consistent colour for a class label.

    Label 0 is reserved for background, while
    foreground classes are numbered 1..42.
    """

    if label <= 0:
        return "#000000"

    return CLASS_COLORS[(label - 1) % len(CLASS_COLORS)]


def draw_boxes(
    ax,
    boxes,
    labels,
    scores=None,
    threshold=0.0,
):

    for index, box in enumerate(boxes):

        if scores is not None:

            score = float(
                scores[index]
            )

            if score < threshold:
                continue

        else:
            score = None

        x1, y1, x2, y2 = [
            float(value)
            for value in box
        ]

        width = x2 - x1
        height = y2 - y1

        label = int(
            labels[index]
        )

        color = get_class_color(label)

        # -------------------------------------------------
        # Bounding box
        # -------------------------------------------------

        rectangle = patches.Rectangle(
            (x1, y1),
            width,
            height,
            fill=False,
            edgecolor=color,
            linewidth=2.0,
        )

        ax.add_patch(rectangle)

        # -------------------------------------------------
        # Label
        # -------------------------------------------------

        if score is None:

            text = f"Class {label}"

        else:

            text = (
                f"Class {label}: "
                f"{score:.2f}"
            )

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

    print(
        f"Generating predictions for "
        f"{len(dataset)} images..."
    )

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

            image_np = (
                image
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )

            fig, axes = plt.subplots(
                1,
                2,
                figsize=(16, 10),
            )

            # ------------------------------------------
            # Ground truth
            # ------------------------------------------

            axes[0].imshow(
                image_np
            )

            axes[0].set_title(
                "Ground Truth",
                fontsize=14,
                fontweight="bold",
            )

            draw_boxes(
                axes[0],
                targets[0]["boxes"],
                targets[0]["labels"],
            )

            axes[0].axis("off")

            # ------------------------------------------
            # Predictions
            # ------------------------------------------

            axes[1].imshow(
                image_np
            )

            axes[1].set_title(
                f"V1 Predictions "
                f"(score ≥ {args.score_threshold:.2f})",
                fontsize=14,
                fontweight="bold",
            )

            draw_boxes(
                axes[1],
                prediction["boxes"]
                .detach()
                .cpu(),
                prediction["labels"]
                .detach()
                .cpu(),
                prediction["scores"]
                .detach()
                .cpu(),
                threshold=args.score_threshold,
            )

            axes[1].axis("off")

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

            print(
                f"Saved: {output_file}"
            )


if __name__ == "__main__":
    main()