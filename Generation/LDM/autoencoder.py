"""
This script trains AutoencoderKL model on a dataset of brain MRI images.
"""
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
from monai.config import print_config
from monai.data import CacheDataset, DataLoader
from monai.utils import first, set_determinism
from monai.transforms import (Compose, LoadImage, ToTensor, ScaleIntensity,
                              EnsureChannelFirst, Resize, SpatialCrop, RandFlip)
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from monai.networks.layers import Act
from tqdm import tqdm
from monai.networks.nets import AutoencoderKL, PatchDiscriminator
from torch.nn import L1Loss  # NEW: L1Loss import
from monai.losses import PatchAdversarialLoss, PerceptualLoss  # NEW: Tutorial loss functions
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
log_dir = r"C:\NTNU\RepoThesis\tensorboard_logs\autoencoder_bravo_" + timestamp
writer = SummaryWriter(log_dir)

# Data setup
data_dir = r"C:\NTNU\MedicalDataSets\brainmetshare-3\train"
transform = Compose([LoadImage(image_only=True),
                     EnsureChannelFirst(),
                     ToTensor(),
                     ScaleIntensity(minv=0.0, maxv=1.0),
                     # RandFlip(0.5),
                     Resize(spatial_size=(128, 128, -1)),
                     # SpatialCrop(roi_start=[27, 27], roi_end=[227, 227]),  # 200x200
                     # SpatialCrop(roi_start=[43, 27], roi_end=[211, 227]),
                     ])

dataset = NiFTIDataset(data_dir=data_dir, mr_sequence="bravo", transform=transform)
train_dataset = extract_slices_single(dataset)
train_data_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, pin_memory=True)
device = torch.device("cuda")

# Model setup
autoencoder = AutoencoderKL(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    channels=(128, 256, 384, 512),
    latent_channels=8,
    num_res_blocks=1,
    norm_num_groups=32,
    attention_levels=(False, False, False, True),
)

discriminator = PatchDiscriminator(
    spatial_dims=2,
    num_layers_d=4,
    channels=64,
    in_channels=1,
    out_channels=1,
    kernel_size=4,
    activation=(Act.LEAKYRELU, {"negative_slope": 0.2}),
    norm="BATCH",
    bias=False,
    padding=1,
)

autoencoder.to(device)
discriminator.to(device)

perceptual_loss = PerceptualLoss(
    spatial_dims=2,
    network_type="radimagenet_resnet50",
    #is_fake_3d=False,
    cache_dir=r"/model_cache",
    #pretrained=True,
    #channel_wise=False,
)
perceptual_loss.to(device)
l1_loss = L1Loss()
adv_loss = PatchAdversarialLoss(criterion="least_squares")

optimizer_ae = torch.optim.AdamW(autoencoder.parameters(), lr=5e-5)
optimizer_disc = torch.optim.AdamW(discriminator.parameters(), lr=2e-4)
scaler = GradScaler('cuda')

# Loss weights
kl_weight = 1e-6
adv_weight = 0.01
perceptual_weight = 0.001


def generate_validation_samples(epoch, num_samples=4):
    """Generate and log validation samples to TensorBoard - reconstruction"""
    autoencoder.eval()
    try:
        with torch.no_grad():
            # Get a batch of real images for reconstruction
            batch = next(iter(train_data_loader))
            if len(batch) > num_samples:
                batch = batch[:num_samples]

            images = batch.to(device)
            with autocast(device_type="cuda", enabled=True):
                reconstruction, z_mu, z_sigma = autoencoder(images)

            # Log original and reconstructed images
            writer.add_images('Original_Images', torch.clamp(images, 0, 1), epoch)
            writer.add_images('Reconstructed_Images', torch.clamp(reconstruction, 0, 1), epoch)
            del images, reconstruction, z_mu, z_sigma
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"Warning: Failed to generate validation samples at epoch {epoch}: {e}")
        torch.cuda.empty_cache()
    autoencoder.train()


# Training loop
total_start = time.time()

for epoch in range(n_epochs):
    autoencoder.train()
    discriminator.train()
    epoch_loss = 0
    gen_epoch_loss = 0
    disc_epoch_loss = 0
    epoch_recon_loss = 0
    epoch_kl_loss = 0
    epoch_perceptual_loss = 0
    progress_bar = tqdm(enumerate(train_data_loader), total=len(train_data_loader), ncols=70)
    progress_bar.set_description(f"Epoch {epoch}")

    for step, batch in progress_bar:
        images = batch.to(device)

        # Train Autoencoder
        optimizer_ae.zero_grad(set_to_none=True)

        with autocast('cuda', enabled=True):
            reconstruction, z_mu, z_sigma = autoencoder(images)

            # Reconstruction loss (L1 from tutorial)
            recons_loss = l1_loss(reconstruction.float(), images.float())

            # KL divergence loss
            kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

            # Perceptual loss (from tutorial)
            p_loss = perceptual_loss(reconstruction.float(), images.float())

            # Adversarial loss - Generator tries to fool discriminator
            logits_fake = discriminator(reconstruction.contiguous().float())[-1]
            generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)

            # Total generator loss
            loss_g = recons_loss + kl_weight * kl_loss + perceptual_weight * p_loss + adv_weight * generator_loss

        scaler.scale(loss_g).backward()
        scaler.step(optimizer_ae)
        scaler.update()

        # Train Discriminator
        optimizer_disc.zero_grad(set_to_none=True)

        with autocast('cuda', enabled=True):
            # Discriminator loss on fake images
            logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
            loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)

            # Discriminator loss on real images
            logits_real = discriminator(images.contiguous().detach())[-1]
            loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)

            # Combined discriminator loss
            discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
            loss_d = adv_weight * discriminator_loss

        scaler.scale(loss_d).backward()
        scaler.step(optimizer_disc)
        scaler.update()

        epoch_loss += recons_loss.item()
        gen_epoch_loss += generator_loss.item()
        disc_epoch_loss += discriminator_loss.item()
        epoch_recon_loss += recons_loss.item()
        epoch_kl_loss += kl_loss.item()
        epoch_perceptual_loss += p_loss.item()

        progress_bar.set_postfix(loss=f"{epoch_loss / (step + 1):.4f}")

    # Calculate and log average losses
    epoch_recon_loss_avg = epoch_loss / len(train_data_loader)
    epoch_gen_loss_avg = gen_epoch_loss / len(train_data_loader)
    epoch_disc_loss_avg = disc_epoch_loss / len(train_data_loader)
    avg_recon_loss = epoch_recon_loss / len(train_data_loader)
    avg_kl_loss = epoch_kl_loss / len(train_data_loader)
    avg_perceptual_loss = epoch_perceptual_loss / len(train_data_loader)

    writer.add_scalar('Loss/Reconstruction', epoch_recon_loss_avg, epoch)
    writer.add_scalar('Loss/Generator', epoch_gen_loss_avg, epoch)
    writer.add_scalar('Loss/Discriminator', epoch_disc_loss_avg, epoch)
    writer.add_scalar('Loss/KL_Divergence', avg_kl_loss, epoch)
    writer.add_scalar('Loss/Perceptual', avg_perceptual_loss, epoch)


    # Validation and model saving
    if (epoch + 1) % val_interval == 0:
        newpath = r'C:\NTNU\RepoThesis\trained_model\autoencoder' + timestamp
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        msgs = [
            f"epoch {epoch:d}/{n_epochs:d}: ",
            f"recons loss: {epoch_recon_loss_avg:.4f}, "
            f"gen_loss: {epoch_gen_loss_avg:.4f}, "
            f"disc_loss: {epoch_disc_loss_avg:.4f}"
        ]
        print("".join(msgs))
        print(f"Generating validation samples...")
        generate_validation_samples(epoch)

        autoencoder.eval()
        path_ae = (newpath + r"\autoencoder_edge_bravo_"
                   + str(bs) + "_Epoch" + str(epoch) + "_of_" + str(n_epochs))
        path_disc = (newpath + r"\discriminator_edge_bravo_"
                     + str(bs) + "_Epoch" + str(epoch) + "_of_" + str(n_epochs))

        torch.save(autoencoder.state_dict(), path_ae)
        torch.save(discriminator.state_dict(), path_disc)
        print(f"✓ Saved models at epoch {epoch}")

total_time = time.time() - total_start
print(f"Training completed in {total_time:.2f} seconds ({total_time / 3600:.2f} hours)")
writer.close()
print("✓ Training completed! Check TensorBoard for metrics.")