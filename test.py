"""
Test script for testing out new features and configurations.
"""
import os
import time
from datetime import datetime
import torch
import torch.nn.functional as F
from monai.data import CacheDataset, DataLoader
from monai.transforms import (Compose, LoadImage, ToTensor, ScaleIntensity,
                              EnsureChannelFirst, Resize)
from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from monai.inferers import DiffusionInferer
from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW

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
scale = 4
bs = 16 * scale

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = r"/home/mode/NTNU/RepoThesis/tensorboard_logs/Test_diffusion_edge_bravo_64_fp16_" + timestamp
writer = SummaryWriter(log_dir)

data_dir = r"/home/mode/NTNU/MedicalDataSets/brainmetshare-3/train"
transform = Compose([LoadImage(image_only=True),
                     EnsureChannelFirst(), ToTensor(),
                     ScaleIntensity(minv=0.0, maxv=1.0),
                     Resize(spatial_size=(64, 64, -1))
                     ])

dataset = NiFTIDataset(data_dir=data_dir, mr_sequence="bravo", transform=transform)
train_dataset = extract_slices_single(dataset)
train_data_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, pin_memory=True)
device = torch.device("cuda")

model = DiffusionModelUNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    channels=(128, 256, 256),
    attention_levels=(False, True, True),
    num_res_blocks=1,
    num_head_channels=256
).to(device)

# Compile the model for performance optimization. Made training at
# least 33% faster or more:"default", "reduce-overhead", "max-autotune"
opt_model = torch.compile(model, mode="default")
optimizer = AdamW(model.parameters(), lr=1e-4 * scale)
lr_scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5 * scale)
scheduler = DDPMScheduler(num_train_timesteps=1000, schedule='cosine')
inferer = DiffusionInferer(scheduler)

def generate_validation_samples(epoch, num_samples=4):
    """Generate and log validation samples to TensorBoard"""
    model.eval()
    try:
        with torch.no_grad():
            noise = torch.randn((num_samples, 1, 64, 64), device=device)
            with autocast(device_type="cuda", enabled=True, dtype=torch.float16):
                samples = inferer.sample(input_noise=noise, diffusion_model=opt_model, scheduler=scheduler)
                samples_normalized = torch.clamp(samples.float(), 0, 1)
                writer.add_images(f'Generated_Images', samples_normalized, epoch)
            del noise, samples
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"Warning: Failed to generate validation samples at epoch {epoch}: {e}")
        torch.cuda.empty_cache()
    finally:
        model.train()


total_start = time.time()
for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(enumerate(train_data_loader), total=len(train_data_loader), ncols=100)
    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in progress_bar:
        # Saw 5% improvement with this + have to have for torch.compile to work.
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
        lr_scheduler.step()
        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=f"{epoch_loss / (step + 1):.4f}")

    avg_epoch_loss = epoch_loss / len(train_data_loader)
    writer.add_scalar('Loss/Training', avg_epoch_loss, epoch)
    print(f"Epoch {epoch}: Loss = {avg_epoch_loss:.6f}")
    if (epoch + 1) % val_interval == 0:
        newpath = f'/home/mode/NTNU/RepoThesis/trained_model/Test_diffusion_edge_64_fp16_{timestamp}'
        os.makedirs(newpath, exist_ok=True)
        print(f"Generating validation samples...")
        generate_validation_samples(epoch)
        model.eval()
        path = f"{newpath}/diffusion_edge_bravo_{bs}_Epoch{epoch}_of_{n_epochs}"
        torch.save(model.state_dict(), path)
        print(f"✓ Saved model at epoch {epoch})")

total_time = time.time() - total_start
print(f"Training completed in {total_time:.2f} seconds ({total_time / 3600:.2f} hours)")
writer.close()
print("✓ Training completed! Check TensorBoard for metrics.")