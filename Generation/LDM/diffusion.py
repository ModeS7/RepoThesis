import sys
import os
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
from monai.data import DataLoader
from monai.transforms import (Compose, LoadImage, ToTensor, ScaleIntensity,
                              EnsureChannelFirst, Resize)
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from monai.inferers import DiffusionInferer
from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler
from monai.networks.nets import AutoencoderKL

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from functions import NiFTIDataset, extract_slices_single

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.enabled = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch._dynamo.config.cache_size_limit = 32


# Training configuration
n_epochs = 100
val_interval = 5
bs = 16
latent = True
image_type = "seg"

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
sufix = image_type + "_" + timestamp
if latent:
    sufix = image_type + "_latent_" + timestamp

# TensorBoard setup
log_dir = r"/home/mode/NTNU/RepoThesis/tensorboard_logs/diffusion_" + sufix
writer = SummaryWriter(log_dir)

# Data setup
if not latent:
    data_dir = r"/home/mode/NTNU/MedicalDataSets/brainmetshare-3/train"
    transform = Compose([LoadImage(image_only=True),
                         EnsureChannelFirst(),
                         ToTensor(),
                         ScaleIntensity(minv=0.0, maxv=1.0),
                         Resize(spatial_size=(64, 64, -1)),
                         ])

    dataset = NiFTIDataset(data_dir=data_dir, mr_sequence=image_type, transform=transform)
    train_dataset = extract_slices_single(dataset)
    model = DiffusionModelUNet(spatial_dims=2,
                               in_channels=1,
                               out_channels=1,
                               channels=(32, 64, 128),
                               attention_levels=(False, True, True),
                               num_res_blocks=1,
                               num_head_channels=256)

if latent:
    class LatentDataset(torch.utils.data.Dataset):
        def __init__(self, latent_path):
            self.latents = np.load(latent_path)
            print(f"Loaded {len(self.latents)} latent representations with shape {self.latents.shape[1:]}")
        def __len__(self):
            return len(self.latents)
        def __getitem__(self, idx):
            # Convert to tensor and ensure float32
            latent = torch.from_numpy(self.latents[idx]).float()
            return latent

    latent_path = r"/home/mode/NTNU/MedicalDataSets/LDM_latent_data/seg_latents.npy"
    train_dataset = LatentDataset(latent_path)
    if image_type == "seg":
        model = DiffusionModelUNet(spatial_dims=2,
                                   in_channels=1,
                                   out_channels=1,
                                   channels=(32, 64),
                                   attention_levels=(True, True),
                                   num_res_blocks=2,
                                   num_head_channels=32)
    elif image_type == "bravo":
        model = DiffusionModelUNet(spatial_dims=2,
                                   in_channels=8,
                                   out_channels=8,
                                   channels=(32, 64),
                                   attention_levels=(False, True, True),
                                   num_res_blocks=1,
                                   num_head_channels=256)
train_data_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True,
                              pin_memory=True, num_workers=4)

device = torch.device("cuda")
model.to(device)
opt_model = torch.compile(model, mode="default")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
optimizer_sr = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)
scheduler = DDPMScheduler(num_train_timesteps=1000, schedule='cosine')
inferer = DiffusionInferer(scheduler)
scaler = GradScaler('cuda')

def generate_validation_samples(epoch, num_samples=4):
    """Generate and log validation samples to TensorBoard - supports both latent and pixel diffusion"""
    model.eval()

    try:
        with torch.no_grad():
            if latent:
                # Latent diffusion mode
                noise_shape = (num_samples, 1, 32, 32)
                log_prefix = "Latent"

                if image_type == "bravo":
                    autoencoder = AutoencoderKL(
                        spatial_dims=2,
                        in_channels=1,
                        out_channels=1,
                        channels=(128, 256),
                        latent_channels=8,
                        num_res_blocks=2,
                        norm_num_groups=32,
                        attention_levels=(False, False),
                    )
                elif image_type == "seg":
                    autoencoder = AutoencoderKL(
                        spatial_dims=2,
                        in_channels=1,
                        out_channels=1,
                        channels=(128, 256),
                        latent_channels=1,
                        num_res_blocks=2,
                        norm_num_groups=32,
                        attention_levels=(False, False),
                    )
                # Load and prepare autoencoder
                auto_path = r"/trained_model/autoencoder_20250624-022945/autoencoder_edge_bravo_16_Epoch99_of_100"
                autoencoder.load_state_dict(torch.load(auto_path))
                autoencoder.to(device)
                autoencoder.eval()

            else:
                # Pixel diffusion mode
                noise_shape = (num_samples, 1, 64, 64)  # Confirm channel count
                log_prefix = "Pixel"

            # Generate noise and samples
            noise = torch.randn(noise_shape, device=device)

            #with autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            samples = inferer.sample(
                input_noise=noise,
                diffusion_model=model,
                scheduler=scheduler
                )

            if latent:
                # Decode latent samples to images
                decoded_images = autoencoder.decode_stage_2_outputs(samples)
                decoded_images_normalized = torch.clamp(decoded_images, 0, 1)

                # Log both latents and decoded images
                writer.add_images(f'Generated_{log_prefix}_Ch0',
                                  torch.clamp(samples[:, 0:1], -3, 3), epoch)
                writer.add_images(f'Generated_Decoded_Images',
                                  decoded_images_normalized, epoch)
            else:
                # Direct image samples
                samples_normalized = torch.clamp(samples.float(), 0, 1)
                writer.add_images(f'Generated_{log_prefix}_Images',
                                      samples_normalized, epoch)

            # Clean up
            del noise, samples
            if latent and 'decoded_images' in locals():
                del decoded_images, decoded_images_normalized
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"Warning: Failed to generate validation samples at epoch {epoch}: {e}")
        # Emergency cleanup
        for var_name in ['noise', 'samples', 'decoded_images', 'decoded_images_normalized']:
            if var_name in locals():
                del locals()[var_name]
        torch.cuda.empty_cache()

    finally:
        model.train()

# Training loop
total_start = time.time()
for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(enumerate(train_data_loader), total=len(train_data_loader), ncols=100)
    progress_bar.set_description(f"Epoch {epoch}")

    for step, batch in progress_bar:
        if hasattr(batch, 'as_tensor'):
            images = batch.as_tensor().to(device)
        else:
            images = batch.to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast('cuda', enabled=True, dtype=torch.bfloat16):
            noise = torch.randn_like(images).to(device)
            timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps,
                                      (images.shape[0],), device=images.device).long()
            noise_pred = inferer(inputs=images, diffusion_model=opt_model, noise=noise, timesteps=timesteps)
            loss = F.mse_loss(noise_pred.float(), noise.float())

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=f"{epoch_loss/(step+1):.4f}")

    optimizer_sr.step()

    # Calculate and log average loss
    avg_epoch_loss = epoch_loss / len(train_data_loader)
    writer.add_scalar('Loss/Training', avg_epoch_loss, epoch)
    print(f"Epoch {epoch}: Average Loss = {avg_epoch_loss:.6f}")

    # Validation and model saving
    if (epoch + 1) % val_interval == 0:
        newpath = r'/home/mode/NTNU/RepoThesis/trained_model/diffusion' + sufix
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        print(f"Generating validation samples...")
        generate_validation_samples(epoch)

        model.eval()
        path = (newpath + r"/diffusion_edge_bravo_"
                + str(bs) + "_Epoch" + str(epoch) + "_of_" + str(n_epochs))
        torch.save(model.state_dict(), path)
        print(f"✓ Saved model at epoch {epoch}")

total_time = time.time() - total_start
print(f"Training completed in {total_time:.2f} seconds ({total_time / 3600:.2f} hours)")
writer.close()
print("✓ Training completed! Check TensorBoard for metrics.")