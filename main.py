import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List

import math
import random

import nrrd                       # pip install pynrrd
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

import diffusers
from diffusers.optimization import get_cosine_schedule_with_warmup
import datasets

# ---------- custom project helpers (unchanged) ----------
from training import TrainingConfig, train_loop
from eval import evaluate_generation, evaluate_sample_many

# -------------------------------------------------------

SPLIT_NAMES = {"train", "val", "test"}

# --------------------------------------------------------------------------------------
#  Utilities for building a slice‑level dataset from a directory of 3‑D NRRD volumes
# --------------------------------------------------------------------------------------

def collect_nrrd_slices(split_dir: Path) -> Dict[str, List]:
    """Return a dict suitable for HuggingFace Dataset.from_dict().

    For every *.nrrd* in *split_dir* we enumerate all axial slices and
    create one record per slice.
    """
    records = {
        "volume": [],          # str path to 3‑D image
        "slice_idx": [],       # int       axial index
        "image_filename": []   # str       unique name e.g. volname_z042.nrrd
    }

    ### Subsample volumes
    STEP = 10
    MAX_SLICES = 100

    for vol_path in split_dir.glob("*.nrrd"):
        vol_path = vol_path.resolve()
        vol_data, _ = nrrd.read(vol_path)      # shape: (D,H,W)
        depth = vol_data.shape[0]
        for z in range(depth):
        # for i, z in enumerate(range(0, depth, STEP)):
        #     if MAX_SLICES is not None and i >= MAX_SLICES:
        #         break
            records["volume"].append(str(vol_path))
            records["slice_idx"].append(z)
            records["image_filename"].append(f"{vol_path.stem}_z{z:03d}.nrrd")
    return records


def collect_seg_slices(seg_split_dir: Path, slice_indices: List[int]) -> List[str]:
    """Return list of mask‑volume paths aligned with *slice_indices*.

    ``slice_indices`` is the list produced by *collect_nrrd_slices* for the images,
    so we guarantee alignment by simply repeating the same volume path once per
    slice.
    """

    ### Subsample volumes
    STEP = 10
    MAX_SLICES = 100
    
    paths = []
    for vol_path in seg_split_dir.glob("*.nrrd"):
        vol_path = vol_path.resolve()
        vol_data, _ = nrrd.read(vol_path)
        depth = vol_data.shape[0]
        paths.extend([str(vol_path)] * depth)
    return paths


# --------------------------------------------------------------------------------------
#  Slice loader and preprocessing
# --------------------------------------------------------------------------------------

def load_slice(vol_path: str, z: int, out_size: int) -> torch.Tensor:
    """Read one slice (z) from 3‑D volume → (1, H, W) torch tensor normalised to [0,1]."""
    vol, _ = nrrd.read(vol_path)           # numpy (D,H,W)
    arr = vol[z].astype(np.float32)

    # Min‑max normalise per slice (modify as needed for modality)
    arr -= arr.min()
    rng = arr.max() - arr.min() + 1e-8
    arr /= rng

    t = torch.from_numpy(arr).unsqueeze(0)   # (1,H,W)

    # Resize to Network input
    t = F.interpolate(t.unsqueeze(0), size=(out_size, out_size), mode="bilinear", align_corners=False)
    return t.squeeze(0)                      # (1,H,W)


# --------------------------------------------------------------------------------------
#  Main script
# --------------------------------------------------------------------------------------

def main(
    mode,
    img_size,
    num_img_channels,
    dataset_name,
    img_dir,
    seg_dir,
    model_type,
    segmentation_guided,
    segmentation_channel_mode,
    num_segmentation_classes,
    train_batch_size,
    eval_batch_size,
    num_epochs,
    resume_epoch=None,
    use_ablated_segmentations=False,
    eval_shuffle_dataloader=True,

    # eval‑only args
    eval_mask_removal=False,
    eval_blank_mask=False,
    eval_sample_size=1000,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    # ------------------------------------------------------------------
    #  Config
    # ------------------------------------------------------------------
    output_dir = f"{model_type.lower()}-{dataset_name}-{img_size}"
    if segmentation_guided:
        output_dir += "-segguided"
    if use_ablated_segmentations or eval_mask_removal or eval_blank_mask:
        output_dir += "-ablated"

    if mode == "train":
        evalset_name = "val"
        assert img_dir is not None, "Must provide --img_dir for training"
    elif "eval" in mode:
        evalset_name = "test"
    else:
        raise ValueError("Unsupported mode")

    config = TrainingConfig(
        image_size=img_size,
        dataset=dataset_name,
        segmentation_guided=segmentation_guided,
        segmentation_channel_mode=segmentation_channel_mode,
        num_segmentation_classes=num_segmentation_classes,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_epochs=num_epochs,
        output_dir=output_dir,
        model_type=model_type,
        resume_epoch=resume_epoch,
        use_ablated_segmentations=use_ablated_segmentations,
    )

    # We ALWAYS load as numpy arrays for NRRD
    load_images_as_np_arrays = True

    # ------------------------------------------------------------------
    #  Build HuggingFace Datasets
    # ------------------------------------------------------------------
    if img_dir is not None:
        img_dir = Path(img_dir)
        dset_train_dict = collect_nrrd_slices(img_dir / "train")
        dset_eval_dict = collect_nrrd_slices(img_dir / evalset_name)

    if segmentation_guided:
        assert seg_dir is not None, "Provide --seg_dir for segmentation‑guided training"
        seg_dir = Path(seg_dir)
        seg_types = ["mask"]  # single mask type by default; extend as needed

        # For every seg_type create slice‑level path list aligned with images
        for seg_type in seg_types:
            seg_train_list = collect_seg_slices(seg_dir / "train", dset_train_dict["slice_idx"])
            seg_eval_list = collect_seg_slices(seg_dir / evalset_name, dset_eval_dict["slice_idx"])
            dset_train_dict[f"seg_{seg_type}_volume"] = seg_train_list
            dset_eval_dict[f"seg_{seg_type}_volume"] = seg_eval_list

    dataset_train = datasets.Dataset.from_dict(dset_train_dict)
    dataset_eval = datasets.Dataset.from_dict(dset_eval_dict)

    # ------------------------------------------------------------------
    #  Transforms (slice loader)
    # ------------------------------------------------------------------
    def transform(examples):
        # Load image slice
        images = [load_slice(v, z, config.image_size) for v, z in zip(examples["volume"], examples["slice_idx"])]
        result = {
            "images": images,
            "image_filenames": examples["image_filename"],
        }

        # Load masks if needed
        if segmentation_guided:
            for seg_type in seg_types:
                seg_slices = [
                    load_slice(v, z, config.image_size) for v, z in zip(examples[f"seg_{seg_type}_volume"], examples["slice_idx"])
                ]
                result[f"seg_{seg_type}"] = seg_slices
        return result

    dataset_train.set_transform(transform)
    dataset_eval.set_transform(transform)

    # ------------------------------------------------------------------
    #  DataLoaders
    # ------------------------------------------------------------------
    train_dataloader = DataLoader(dataset_train, batch_size=config.train_batch_size, shuffle=True)
    eval_dataloader = DataLoader(dataset_eval, batch_size=config.eval_batch_size, shuffle=eval_shuffle_dataloader)

    # ------------------------------------------------------------------
    #  Model & diffusion scheduler
    # ------------------------------------------------------------------
    in_channels = num_img_channels
    if segmentation_guided:
        if segmentation_channel_mode == "single":
            in_channels += 1
        elif segmentation_channel_mode == "multi":
            in_channels += len(seg_types)

    model = diffusers.UNet2DModel(
        sample_size=config.image_size,
        in_channels=in_channels,
        out_channels=num_img_channels,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )

    if (mode == "train" and resume_epoch is not None) or "eval" in mode:
        model = model.from_pretrained(Path(config.output_dir) / "unet", use_safetensors=True)

    model = nn.DataParallel(model).to(device)

    if model_type == "DDPM":
        noise_scheduler = diffusers.DDPMScheduler(num_train_timesteps=1000)
    elif model_type == "DDIM":
        noise_scheduler = diffusers.DDIMScheduler(num_train_timesteps=1000)
    else:
        raise ValueError("Unknown model_type")

    # ------------------------------------------------------------------
    #  Train / evaluate
    # ------------------------------------------------------------------
    if mode == "train":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.lr_warmup_steps,
            num_training_steps=len(train_dataloader) * config.num_epochs,
        )

        train_loop(
            config,
            model,
            noise_scheduler,
            optimizer,
            train_dataloader,
            eval_dataloader,
            lr_scheduler,
            device=device,
        )

    elif mode == "eval":
        evaluate_generation(
            config,
            model,
            noise_scheduler,
            eval_dataloader,
            eval_mask_removal=eval_mask_removal,
            eval_blank_mask=eval_blank_mask,
            device=device,
        )

    elif mode == "eval_many":
        evaluate_sample_many(
            eval_sample_size,
            config,
            model,
            noise_scheduler,
            eval_dataloader,
            device=device,
        )
    else:
        raise ValueError(f"Mode {mode} not supported")


# -------------------------------------------------------------------------------------------------
#  CLI
# -------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--num_img_channels", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="avt")
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--seg_dir", type=str, default=None)
    parser.add_argument("--model_type", type=str, default="DDPM", choices=["DDPM", "DDIM"])
    parser.add_argument("--segmentation_guided", action="store_true")
    parser.add_argument("--segmentation_channel_mode", type=str, default="single", choices=["single", "multi"])
    parser.add_argument("--num_segmentation_classes", type=int, default=None)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--resume_epoch", type=int, default=None)
    parser.add_argument("--use_ablated_segmentations", action="store_true")
    parser.add_argument("--eval_noshuffle_dataloader", action="store_true")
    parser.add_argument("--eval_mask_removal", action="store_true")
    parser.add_argument("--eval_blank_mask", action="store_true")
    parser.add_argument("--eval_sample_size", type=int, default=1000)

    args = parser.parse_args()

    main(
        mode=args.mode,
        img_size=args.img_size,
        num_img_channels=args.num_img_channels,
        dataset_name=args.dataset,
        img_dir=args.img_dir,
        seg_dir=args.seg_dir,
        model_type=args.model_type,
        segmentation_guided=args.segmentation_guided,
        segmentation_channel_mode=args.segmentation_channel_mode,
        num_segmentation_classes=args.num_segmentation_classes,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_epochs=args.num_epochs,
        resume_epoch=args.resume_epoch,
        use_ablated_segmentations=args.use_ablated_segmentations,
        eval_shuffle_dataloader=not args.eval_noshuffle_dataloader,
        eval_mask_removal=args.eval_mask_removal,
        eval_blank_mask=args.eval_blank_mask,
        eval_sample_size=args.eval_sample_size,
    )
