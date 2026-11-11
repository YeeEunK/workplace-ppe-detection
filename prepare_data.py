#!/usr/bin/env python3
"""Build a reproducible five-class PPE dataset from the official YOLO archive."""

import argparse
import csv
import json
import random
import shutil
from collections import Counter
from pathlib import Path


CLASS_MAP = {5: (0, "Person"), 0: (1, "helmet"), 6: (2, "no_helmet"),
             2: (3, "vest"), 11: (4, "no_vest")}
NAMES = ["Person", "helmet", "no_helmet", "vest", "no_vest"]
SEED = 42


def convert_line(line, label_path):
    """Remap one YOLO box and clip edges to the image boundary."""
    fields = line.split()
    if len(fields) != 5:
        raise ValueError(f"Malformed label in {label_path}: {line}")
    old_id = int(fields[0])
    if old_id not in range(12):
        raise ValueError(f"Unknown class in {label_path}: {old_id}")
    if old_id not in CLASS_MAP:
        return None, False
    x, y, w, h = map(float, fields[1:])
    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
        raise ValueError(f"Invalid coordinates in {label_path}: {line}")
    x1, x2 = max(0, x - w / 2), min(1, x + w / 2)
    y1, y2 = max(0, y - h / 2), min(1, y + h / 2)
    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"Box outside image in {label_path}: {line}")
    clipped = (x1 != x - w / 2 or x2 != x + w / 2 or
               y1 != y - h / 2 or y2 != y + h / 2)
    new_id = CLASS_MAP[old_id][0]
    return f"{new_id} {(x1+x2)/2:.6f} {(y1+y2)/2:.6f} {x2-x1:.6f} {y2-y1:.6f}", clipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Official construction-ppe folder")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "data" / "construction-ppe-5class")
    args = parser.parse_args()
    src, dst = args.source.resolve(), args.output.resolve()
    if dst.exists() and any(dst.iterdir()):
        raise FileExistsError(f"Output already contains files: {dst}")
    train_files = sorted((src / "images" / "train").glob("*.jpg"))
    original_val = sorted((src / "images" / "val").glob("*.jpg"))
    if len(train_files) != 800 or len(original_val) != 200:
        raise ValueError("Expected the official 800 train / 200 val archive")
    val_names = [p.name for p in original_val]
    random.Random(SEED).shuffle(val_names)
    test_names = set(val_names[:100])
    # Similar camera views were checked manually; keep this original-val frame out of test.
    related_frame = "source-9e10b0409f922546c35b276eb5a5756e.jpg"
    if related_frame in test_names:
        raise ValueError("A known related camera frame landed in test; revise split")

    manifest = []
    counts = Counter()
    objects = Counter()
    clipped_count = 0
    empty_count = 0
    for original_split, files in (("train", train_files), ("val", original_val)):
        for image in files:
            split = "train" if original_split == "train" else ("test" if image.name in test_names else "val")
            original_label = src / "labels" / original_split / (image.stem + ".txt")
            if not original_label.is_file():
                raise FileNotFoundError(original_label)
            target_image = dst / "images" / split / image.name
            target_label = dst / "labels" / split / original_label.name
            target_image.parent.mkdir(parents=True, exist_ok=True)
            target_label.parent.mkdir(parents=True, exist_ok=True)
            converted = [convert_line(line, original_label)
                         for line in original_label.read_text().splitlines() if line.strip()]
            selected = [line for line, _ in converted if line is not None]
            clipped_count += sum(clipped for line, clipped in converted if line is not None)
            if not selected:
                empty_count += 1
            target_label.write_text("\n".join(selected) + ("\n" if selected else ""))
            shutil.copy2(image, target_image)
            counts[split] += 1
            objects[split] += len(selected)
            manifest.append([image.name, original_split, split, len(selected)])

    with (dst / "split_manifest.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image", "original_split", "split", "selected_objects"])
        writer.writerows(sorted(manifest))
    with (dst / "class_mapping.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["original_id", "new_id", "name"])
        for old_id, (new_id, name) in CLASS_MAP.items():
            writer.writerow([old_id, new_id, name])
    (dst / "data.yaml").write_text(
        "train: images/train\nval: images/val\ntest: images/test\nnames:\n"
        + "".join(f"  {i}: {name}\n" for i, name in enumerate(NAMES))
    )
    summary = {"seed": SEED, "split_method": "original train kept; sorted original val shuffled with Python random, first 100 test",
               "images": dict(counts), "selected_objects": dict(objects),
               "empty_labels": empty_count, "boxes_clipped": clipped_count}
    (dst / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
