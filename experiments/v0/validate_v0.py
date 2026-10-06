import argparse
import math

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--bbox-format", choices=["xywh", "xyxy"], default="xywh")
    args = parser.parse_args()

    ds = load_dataset(
        "ai4bharat/indicdlp",
        split=args.split,
    )

    required = {"image", "bboxes", "category_ids"}
    missing = required - set(ds.column_names)
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    print("Columns:", ds.column_names)
    print("Rows:", len(ds))

    bad = 0
    empty = 0
    total_boxes = 0

    for i in range(min(args.samples, len(ds))):
        row = ds[i]
        image = row["image"]
        width, height = image.size

        boxes = row["bboxes"]
        labels = row["category_ids"]

        if len(boxes) != len(labels):
            print(f"[BAD] {i}: {len(boxes)} boxes vs {len(labels)} labels")
            bad += 1
            continue

        if len(boxes) == 0:
            empty += 1

        for box in boxes:
            if len(box) != 4:
                bad += 1
                continue

            x, y, a, b = [float(v) for v in box]

            if args.bbox_format == "xywh":
                x2, y2 = x + a, y + b
            else:
                x2, y2 = a, b

            values = [x, y, x2, y2]

            if not all(math.isfinite(v) for v in values):
                bad += 1
                continue

            # Allow small numerical tolerance; annotations should be inside
            # the image after clipping.
            if x2 <= x or y2 <= y:
                bad += 1

            total_boxes += 1

    print("\nV0 validation")
    print("Samples checked:", min(args.samples, len(ds)))
    print("Boxes checked:", total_boxes)
    print("Empty annotation rows:", empty)
    print("Potentially bad annotations:", bad)

    if bad:
        print(
            "\nIf many boxes are invalid, rerun with "
            "--bbox-format xyxy and compare."
        )
        raise SystemExit(2)

    print("\nV0 schema/annotation sanity check passed.")


if __name__ == "__main__":
    main()
