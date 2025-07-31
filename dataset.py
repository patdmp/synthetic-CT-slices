import os
import psutil, shutil
from monai.data import Dataset, DataLoader

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, EnsureTyped
)

# ---------------- resource thresholds -----------------
MIN_FREE_RAM_GB   = 1.0
MIN_FREE_DISK_GB  = 1.0
CHECK_EVERY_N_SLICES = 50
TARGET_SPACING = (0.8, 0.8, 3.0)
# ------------------------------------------------------

def _enough_resources():
    avail_ram_gb  = psutil.virtual_memory().available / 2**30
    cache_root    = os.path.expanduser("~/.cache")
    avail_disk_gb = shutil.disk_usage(cache_root).free / 2**30
    return (avail_ram_gb > MIN_FREE_RAM_GB and avail_disk_gb > MIN_FREE_DISK_GB), avail_ram_gb, avail_disk_gb

def pad_crop_2d(t, size=256):
    # t: (1,H,W)
    _, H, W = t.shape
    # pad (bottom/right) if smaller than target
    pad_h = max(0, size - H)
    pad_w = max(0, size - W)
    if pad_h > 0 or pad_w > 0:
        t = F.pad(t, (0, pad_w, 0, pad_h))
    # center crop to exactly size×size
    H2, W2 = t.shape[1:]
    start_h = (H2 - size) // 2
    start_w = (W2 - size) // 2
    t = t[:, start_h:start_h + size, start_w:start_w + size]
    return t

class NRRDDataset(Dataset):
    def __init__(self,
                 img_dir=None,
                 seg_dir=None,
                 split="train",
                 img_size=256,
                 segmentation_guided=True):
        super().__init__()
        self.samples = []

        seg_types = os.listdir(seg_dir) if segmentation_guided else []
        seg_keys  = [f"seg_{t}" for t in seg_types]

        # --- build list of volume paths ---
        if img_dir is not None:
            vol_paths = [
                os.path.join(img_dir, split, f)
                for f in os.listdir(os.path.join(img_dir, split))
                if f.endswith(".nrrd")
            ]
        else:
            vol_paths = [
                os.path.join(seg_dir, seg_types[0], split, f)
                for f in os.listdir(os.path.join(seg_dir, seg_types[0], split))
                if f.endswith(".nrrd")
            ]

        # --- dictionary transform applied once per volume ---
        # keys list: image + all segmentations
        all_keys = (["image"] if img_dir is not None else []) + seg_keys

        volume_xform = Compose([
            LoadImaged(keys=all_keys),
            EnsureChannelFirstd(keys=all_keys),
            Orientationd(keys=all_keys, axcodes="RAS"),
            Spacingd(
                keys=all_keys,
                pixdim=TARGET_SPACING,
                mode=("bilinear",) + ("nearest",) * len(seg_keys) if img_dir is not None
                     else ("nearest",) * len(seg_keys)
            ),
            # scale only the image
            ScaleIntensityRanged(
                keys="image",
                a_min=-1000, a_max=1000,
                b_min=-1.0, b_max=1.0, clip=True
            ) if img_dir is not None else (lambda x: x),
            EnsureTyped(keys=all_keys),
        ])

        # --- preload volumes and slice ---
        slice_counter = 0
        for vol_path in vol_paths:
            # assemble dictionary for this volume
            data_dict = {}
            if img_dir is not None:
                data_dict["image"] = vol_path
            for t in seg_types:
                data_dict[f"seg_{t}"] = os.path.join(seg_dir, t, split, os.path.basename(vol_path))

            data = volume_xform(data_dict)   # after this: tensors (1,H,W,D)

            # depth along last axis
            ref_key = "image" if img_dir is not None else seg_keys[0]
            depth = data[ref_key].shape[-1]

            stem = os.path.splitext(os.path.basename(vol_path))[0]
            for z in range(depth):
                record = {}
                if img_dir is not None:
                    img_slice = data["image"][..., z]
                    img_slice = pad_crop_2d(img_slice, size=img_size)
                    record["images"] = img_slice
                if segmentation_guided:
                    for k in seg_keys:
                        m_slice = data[k][..., z]
                        m_slice = pad_crop_2d(m_slice, size=img_size)
                        record[k] = m_slice
                record["image_filenames"] = f"{stem}_axial_{z:04d}"
                self.samples.append(record)

                slice_counter += 1
                if slice_counter % CHECK_EVERY_N_SLICES == 0:
                    ok, ram, disk = _enough_resources()
                    if not ok:
                        print(f"[NRRDDataset] stopping preload: only {ram:.1f} GB RAM / {disk:.1f} GB disk free")
                        return

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def make_loaders(img_dir, seg_dir, img_size, segmentation_guided, batch_sizes, num_workers=4):
    train_ds = NRRDDataset(img_dir, seg_dir, split="train",
                           img_size=img_size, segmentation_guided=segmentation_guided)
    val_ds = NRRDDataset(img_dir, seg_dir, split="val",
                         img_size=img_size, segmentation_guided=segmentation_guided)

    train_loader = DataLoader(train_ds, batch_size=batch_sizes["train"],
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_sizes["val"],
                            shuffle=False, num_workers=num_workers)
    return train_loader, val_loader
