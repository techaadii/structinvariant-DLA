import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from dataset.indicdlp import IndicDLP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train")
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--bbox-format", default="xywh", choices=["xywh", "xyxy"])
    parser.add_argument("--output", default="outputs/v0/sample.png")
    args = parser.parse_args()

    ds = IndicDLP(
        split=args.split,
        bbox_format=args.bbox_format,
    )

    ds.save_category_mapping("outputs/category_mapping.json")

    index = args.index
    if index is None:
        index = random.randrange(len(ds))

    sample = ds[index]
    image = sample["image"].permute(1, 2, 0).numpy()

    fig, ax = plt.subplots(figsize=(12, 16))
    ax.imshow(image)

    for box, label in zip(
        sample["target"]["boxes"].numpy(),
        sample["target"]["labels"].numpy(),
    ):
        x1, y1, x2, y2 = box

        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            linewidth=1.5,
        )
        ax.add_patch(rect)

        ax.text(
            x1,
            max(y1 - 2, 2),
            str(label),
            fontsize=7,
            bbox={"facecolor": "white", "alpha": 0.7, "pad": 1},
        )

    ax.axis("off")
    fig.tight_layout(pad=0)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", output)
    print("Index:", index)
    print("Image size:", sample["width"], "x", sample["height"])
    print("Boxes:", len(sample["target"]["boxes"]))


if __name__ == "__main__":
    main()
