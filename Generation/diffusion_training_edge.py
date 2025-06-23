import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import shutil
import tempfile
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from monai import transforms
from monai.apps import MedNISTDataset
from monai.config import print_config
from monai.data import CacheDataset, DataLoader
from monai.utils import first, set_determinism
from monai.transforms import (Compose, LoadImage, ToTensor, ScaleIntensity,
                              EnsureChannelFirst, Resize, SpatialCrop, RandFlip)
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from monai.inferers import DiffusionInferer
from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler
from functions import NiFTIDataset, extract_slices_single


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.enabled = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)


# Training configuration
n_epochs = 100
val_interval = 5
bs = 16

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

# TensorBoard setup
log_dir = r"C:\NTNU\RepoThesis\tensorboard_logs\diffusion_edge_bravo_64" + timestamp
writer = SummaryWriter(log_dir)


"""def get_progressive_crop(epoch, max_epochs):
    #Progressive crop sizing during training
    if epoch < max_epochs * 0.3:
        return SpatialCrop(roi_start=[59, 43], roi_end=[195, 211])  # 136x168
    elif epoch < max_epochs * 0.7:
        return SpatialCrop(roi_start=[51, 35], roi_end=[203, 219])  # 152x184
    else:
        return SpatialCrop(roi_start=[43, 27], roi_end=[211, 227])  # 168x200"""


# Data setup
data_dir = r"C:\NTNU\MedicalDataSets\brainmetshare-3\train"
transform = Compose([LoadImage(image_only=True),
                     EnsureChannelFirst(),
                     ToTensor(),
                     ScaleIntensity(minv=0.0, maxv=1.0),
                     #RandFlip(0.5),
                     Resize(spatial_size=(64, 64, -1)),
                     #SpatialCrop(roi_start=[27, 27], roi_end=[227, 227]),  # 200x200
                     #SpatialCrop(roi_start=[43, 27], roi_end=[211, 227]),
                     ])

dataset = NiFTIDataset(data_dir=data_dir, mr_sequence="bravo", transform=transform)
train_dataset = extract_slices_single(dataset)
train_data_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, pin_memory=True)
device = torch.device("cuda")


# Model setup
model = DiffusionModelUNet(spatial_dims=2,
                           in_channels=1,
                           out_channels=1,
                           channels=(128, 256, 256),
                           attention_levels=(False, True, True),
                           num_res_blocks=1,
                           num_head_channels=256)

model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = DDPMScheduler(num_train_timesteps=1000, schedule='cosine')
inferer = DiffusionInferer(scheduler)
scaler = GradScaler('cuda')


def generate_validation_samples(epoch, num_samples=4):
    """Generate and log validation samples to TensorBoard - image generation"""
    model.eval()
    try:
        with torch.no_grad():
            noise = torch.randn((num_samples, 1, 64, 64), device=device)
            with autocast(device_type="cuda", enabled=True):
                image = inferer.sample(
                    input_noise=noise,
                    diffusion_model=model,
                    scheduler=scheduler
                )
            image_normalized = torch.clamp(image, 0, 1)
            writer.add_images('Generated_Annotation_Masks', image_normalized, epoch)
            del noise, image, image_normalized
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"Warning: Failed to generate validation samples at epoch {epoch}: {e}")
        # Still clean up any allocated tensors
        if 'noise' in locals():
            del noise
        if 'image' in locals():
            del image
        torch.cuda.empty_cache()
    model.train()


# Training loop
total_start = time.time()

for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(enumerate(train_data_loader), total=len(train_data_loader), ncols=70)
    progress_bar.set_description(f"Epoch {epoch}")

    for step, batch in progress_bar:
        images = batch.to(device)
        optimizer.zero_grad(set_to_none=True)

        with autocast('cuda', enabled=True):
            noise = torch.randn_like(images).to(device)
            timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps,
                                      (images.shape[0],), device=images.device).long()
            noise_pred = inferer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps)
            loss = F.mse_loss(noise_pred.float(), noise.float())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        #scaler.step(optimizer_sr)
        scaler.update()
        #loss.backward()
        #optimizer.step()
        #optimizer_sr.step()
        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=f"{epoch_loss/(step+1):.4f}")

    # Calculate and log average loss
    avg_epoch_loss = epoch_loss / len(train_data_loader)
    writer.add_scalar('Loss/Training', avg_epoch_loss, epoch)
    print(f"Epoch {epoch}: Average Loss = {avg_epoch_loss:.6f}")

    # Validation and model saving
    if (epoch + 1) % val_interval == 0:
        print(f"Generating validation samples...")
        generate_validation_samples(epoch)

        model.eval()
        path = (r"C:\NTNU\RepoThesis\trained_model\diffusion_edge\diffusion_edge_bravo_64"
                + str(bs) + "_Epoch" + str(epoch) + "_of_" + str(n_epochs))
        torch.save(model.state_dict(), path)
        print(f"✓ Saved model at epoch {epoch}")

total_time = time.time() - total_start
print(f"Training completed in {total_time:.2f} seconds ({total_time / 3600:.2f} hours)")
writer.close()
print("✓ Training completed! Check TensorBoard for metrics.")