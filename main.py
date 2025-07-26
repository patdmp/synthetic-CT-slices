import os

import torch

import monai
from monai.config import print_config

import diffusers
from diffusers.optimization import get_cosine_schedule_with_warmup
import torch.nn as nn

# custom imports
from utils import TrainConfig
from dataset import make_loaders
from training import train_loop
from eval import evaluate_generation, evaluate_sample_many

def main():
    #--- Environment Setup -----------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', device)

    # logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    # print_config()

    #--- Set Hyperparameters ----------------------------------------------
    cfg = TrainConfig(
        mode="train",
        model_type="DDIM",
        output_dir="runs",
        img_dir="data/img",
        seg_dir="data/seg",
        dataset="AVT",
        num_epochs=400,
        image_size=256,
        train_batch_size=8,
        eval_batch_size=4,
        segmentation_guided=True,
        segmentation_channel_mode="single",
        num_segmentation_classes=2,
    )
    # cfg = TrainConfig(
    #     mode="eval_many",
    #     model_type="DDIM",
    #     output_dir="runs/ddim-AVT_dongyang-256-segguided-20250725-045637/checkpoint-epoch400",
    #     dataset="AVT_dongyang",
    #     image_size=256,
    #     eval_batch_size=8,
    #     eval_sample_size=2000,
    #     img_dir="data_dongyang/img",
    #     seg_dir="data_dongyang/seg",
    #     segmentation_guided=True,
    #     num_segmentation_classes=2,
    # )
    os.makedirs(cfg.output_dir, exist_ok=True)
    

    #--- Load Dataset ---------------------------------------------
    train_loader, val_loader = make_loaders(
        img_dir=cfg.img_dir,
        seg_dir=cfg.seg_dir,
        img_size=cfg.image_size,
        segmentation_guided=cfg.segmentation_guided,
        batch_sizes={"train": cfg.train_batch_size, "val": cfg.eval_batch_size},
        num_workers=4,
    )

    batch = next(iter(train_loader))
    print("Batch tensor keys :", batch.keys())
    print("Batch 'images'    :", batch["images"].shape)      # (B, 1, 256, 256)


    #--- Define Model, Optimizer, Scheduler -----------------------------
    in_channels = cfg.num_img_channels
    if cfg.segmentation_guided:
        assert cfg.num_segmentation_classes is not None
        assert cfg.num_segmentation_classes > 1, "must have at least 2 segmentation classes (INCLUDING background)" 
        if cfg.segmentation_channel_mode == "single":
            in_channels += 1
        elif cfg.segmentation_channel_mode == "multi":
            in_channels = len(os.listdir(cfg.seg_dir)) + in_channels

    model = diffusers.UNet2DModel(
            sample_size=cfg.image_size,  # the target image resolution
            in_channels=in_channels,  # the number of input channels, 3 for RGB images
            out_channels=cfg.num_img_channels,  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channes for each UNet block
            down_block_types=(
                "DownBlock2D",  # a regular ResNet downsampling block
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",  # a regular ResNet upsampling block
                "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D"
            ),
        )

    if (cfg.mode == "train" and cfg.resume_epoch is not None) or "eval" in cfg.mode:
        if cfg.mode == "train":
            print("resuming from model at training epoch {}".format(cfg.resume_epoch))
        elif "eval" in cfg.mode:
            print("loading saved model...")
        model = model.from_pretrained(os.path.join(cfg.output_dir, 'unet'), use_safetensors=True)

    model = nn.DataParallel(model)
    model.to(device)

    if cfg.model_type == "DDPM":
        noise_scheduler = diffusers.DDPMScheduler(num_train_timesteps=1000)
    elif cfg.model_type == "DDIM":
        noise_scheduler = diffusers.DDIMScheduler(num_train_timesteps=1000)


    #--- Training ----------------------------------------------------
    if cfg.mode == "train":
        # training setup
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=cfg.lr_warmup_steps,
            num_training_steps=(len(train_loader) * cfg.num_epochs),
        )

        # train
        train_loop(
            cfg, 
            model, 
            noise_scheduler, 
            optimizer, 
            train_loader, 
            val_loader, 
            lr_scheduler, 
            device=device
            )
    elif cfg.mode == "eval":
        evaluate_generation(
            cfg, 
            model, 
            noise_scheduler,
            val_loader, 
            eval_mask_removal=cfg.eval_mask_removal,
            eval_blank_mask=cfg.eval_blank_mask,
            device=device
            )

    elif cfg.mode == "eval_many":
        """
        generate many images and save them to a directory, saved individually
        """
        evaluate_sample_many(
            cfg.eval_sample_size,
            cfg,
            model,
            noise_scheduler,
            val_loader,
            device=device
            )

    else:
        raise ValueError("mode \"{}\" not supported.".format(cfg.mode))
    
if __name__ == "__main__":
    main()