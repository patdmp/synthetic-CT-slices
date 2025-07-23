import os
from torch.utils.data import Dataset
import nrrd
import numpy as np
import psutil, shutil

import monai
from monai.config import print_config
from monai.data import DataLoader
from monai.transforms import (
    EnsureChannelFirst,
    Compose,
    Resize,
    ScaleIntensity,
    ToTensor,
)

# ---------- tweak these two values ---------------------------------
MIN_FREE_RAM_GB   = 1.0   # stop if < 1 GB RAM left
MIN_FREE_DISK_GB  = 1.0   # stop if < 1 GB free on the caching partition
CHECK_EVERY_N_SLICES = 50 # how often to poll resources
# -------------------------------------------------------------------

def _enough_resources():
    # --- RAM ---
    avail_ram_gb = psutil.virtual_memory().available / 2**30
    # --- disk (whatever partition holds ~/.cache) ---
    cache_root   = os.path.expanduser("~/.cache")
    avail_disk_gb = shutil.disk_usage(cache_root).free / 2**30
    return (
        avail_ram_gb > MIN_FREE_RAM_GB
        and avail_disk_gb > MIN_FREE_DISK_GB
    ), avail_ram_gb, avail_disk_gb


class NRRDDataset(Dataset):
    def __init__(self, 
                 img_dir = None, 
                 seg_dir = None, 
                 split="train",
                 img_size=256,
                 segmentation_guided=True,
        ):
        super().__init__()
        self.img_dir = img_dir
        self.seg_dir = seg_dir
        self.split = split
        self.segmentation_guided = segmentation_guided
        self.samples = []

        seg_types = os.listdir(self.seg_dir)

        #--- Transforms ---------------------------------------------------
        img_tf = Compose([
            # EnsureChannelFirst(),
            Resize((img_size, img_size)),
            ScaleIntensity(minv=-1.0, maxv=1.0),
            ToTensor(),
        ]) if img_dir is not None else None

        seg_tf = Compose([
            # EnsureChannelFirst(),
            Resize((img_size, img_size), mode="nearest"),
            ToTensor(),
        ]) if segmentation_guided else None

        #--- Get Volume Paths ----------------------------------------
        if img_dir is not None:
            vol_paths = [os.path.join(img_dir, split, f) for f in os.listdir(os.path.join(img_dir, split)) if f.endswith('.nrrd')]
        else:
            vol_paths = [os.path.join(seg_dir, seg_type, split, f) for seg_type in seg_types for f in os.listdir(os.path.join(seg_dir, seg_type, split)) if f.endswith('.nrrd')]

        # --- Pre‑load and Slice Volumes -------------------------------
        slice_counter = 0
        for vol_path in vol_paths:
            # read volume
            vol_img = None
            if img_dir is not None:
                vol_img, _ = nrrd.read(vol_path)             # (H,W,D)

            mask_vols = {}
            if segmentation_guided:
                for seg_type in seg_types:
                    m_path = os.path.join(seg_dir, seg_type, split, os.path.basename(vol_path))
                    mask_vols[seg_type], _ = nrrd.read(m_path)

            depth = vol_img.shape[2] if img_dir else next(iter(mask_vols.values())).shape[2]

            for z in range(depth):
                record = {}

                # image slice
                if img_dir:
                    img_slice = vol_img[:, :, z].astype(np.float32)
                    img_slice = np.expand_dims(img_slice, axis=0)  # (1,H,W)
                    img_slice = img_tf(img_slice)
                    record["images"] = img_slice

                # mask slices
                if segmentation_guided:
                    for st in seg_types:
                        m = mask_vols[st][:, :, z].astype(np.float32)
                        m = np.expand_dims(m, axis=0) 
                        record[f"seg_{st}"] = seg_tf(m)

                # filename
                stem = os.path.splitext(os.path.basename(vol_path))[0]
                record["image_filenames"] = f"{stem}_axial_{z:04d}"

                self.samples.append(record)
                
                # periodic resource check
                slice_counter += 1
                if slice_counter % CHECK_EVERY_N_SLICES == 0:
                    ok, ram, disk = _enough_resources()
                    if not ok:
                        print(
                            f"[NRRDDataset] stopping preload:"
                            f" only {ram:.1f} GB RAM / {disk:.1f} GB disk free"
                        )
                        return


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    
def make_loaders(
    img_dir,
    seg_dir,
    img_size,
    segmentation_guided,
    batch_sizes,
    num_workers=4,
):
    train_ds = NRRDDataset(
        img_dir,
        seg_dir,
        split="train",
        img_size=img_size,
        segmentation_guided=segmentation_guided,
    )
    val_ds = NRRDDataset(
        img_dir,
        seg_dir,
        split="val",
        img_size=img_size,
        segmentation_guided=segmentation_guided,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_sizes["train"],
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_sizes["val"],
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader
