import argparse
import math
import random
import shutil
from pathlib import Path
from typing import List, Tuple

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_SEED = 42  # set for reproducibility


def collect_subjects(avt_root: Path) -> dict[str, List[Tuple[Path, Path]]]:
    """
    Return {cohort: [(image_path, mask_path), ...]} mapping.
    """
    cohorts = {}
    for cohort_dir in (avt_root).iterdir():
        if not cohort_dir.is_dir():
            continue
        cohort = cohort_dir.name

        ###
        if not cohort == "Dongyang":
            # Skip all cohorts except "Dongyang" for now.
            continue
        ###

        cohorts[cohort] = []
        for subject_dir in cohort_dir.iterdir():
            if not subject_dir.is_dir():
                continue
            subject = subject_dir.name
            img = subject_dir / f"{subject}.nrrd"
            if not img.exists():
                raise FileNotFoundError(f"Image not found: {img}")
            # assume the mask is the only *.seg.nrrd file or named D1.seg.nrrd
            candidates = list(subject_dir.glob("*.seg.nrrd"))
            if not candidates:
                raise FileNotFoundError(f"No *.seg.nrrd in {subject_dir}")
            mask = candidates[0]
            cohorts[cohort].append((img, mask))
    return cohorts


def split_indices(n_items: int) -> dict[str, range]:
    """
    Return dict with index ranges for train/val/test given total items.
    Uses floor for train/val; remainder -> test.
    """
    n_train = math.floor(n_items * SPLIT_RATIOS["train"])
    n_val = math.floor(n_items * SPLIT_RATIOS["val"])
    n_test = n_items - n_train - n_val
    assert n_test >= 0
    return {
        "train": range(0, n_train),
        "val": range(n_train, n_train + n_val),
        "test": range(n_train + n_val, n_items),
    }


def copy_pair(img: Path, mask: Path, out_root: Path, split: str) -> None:
    """
    Copy image and mask into <out_root>/data|mask/<split>/.
    Mask is renamed with seg_ prefix.
    """
    dest_img = out_root / "img" / split / img.name
    dest_mask = out_root / "seg" / "all" / split / img.name
    dest_img.parent.mkdir(parents=True, exist_ok=True)
    dest_mask.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img, dest_img)
    shutil.copy2(mask, dest_mask)


def main(avt_root: Path, out_root: Path) -> None:
    random.seed(RANDOM_SEED)
    cohorts = collect_subjects(avt_root)

    for cohort, pairs in cohorts.items():
        random.shuffle(pairs)
        idx_ranges = split_indices(len(pairs))
        for split, idx_range in idx_ranges.items():
            for i in idx_range:
                img_path, mask_path = pairs[i]
                copy_pair(img_path, mask_path, out_root, split)

    print(f"Finished restructuring into: {out_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-structure AVT NRRD dataset.")
    parser.add_argument(
        "--src",
        required=True,
        type=Path,
        help="Path to the directory that contains data/AVT/…",
    )
    parser.add_argument(
        "--dst",
        required=True,
        type=Path,
        help="Destination root directory for restructured data.",
    )
    args = parser.parse_args()
    main(args.src.resolve(), args.dst.resolve())

