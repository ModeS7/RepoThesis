import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from monai.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
import torch
import time
from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from torch.utils.data import ConcatDataset
#from torchao.optim import AdamW8bit
from monai.transforms import (Compose, LoadImage, ToTensor, ScaleIntensity,
                              EnsureChannelFirst, Resize, SpatialCrop)
from functions import NiFTIDataset, extract_slices_single


torch.backends.cuda.matmul.allow_tf32 = True  # allow TF32 for faster training
torch.backends.cudnn.allow_tf32 = True  # allow TF32 for faster training
torch.backends.cudnn.benchmark = True # enable benchmark mode for faster training
torch.backends.cudnn.deterministic = False  # allow non-deterministic algorithms for faster training
torch.backends.cudnn.enabled = True  # enable cuDNN for faster training

#torch.backends.cuda.matmul.allow_bf16 = True  # allow bfloat16 for faster training
#torch.backends.cudnn.allow_bf16 = True  # allow bfloat16 for faster training

n_epochs = 100
val_interval = 5
bs = 16

data_dir = r"C:\NTNU\MedicalDataSets\brainmetshare-3\train"
transform = Compose([LoadImage(image_only=True),
                     EnsureChannelFirst(),
                     ToTensor(),
                     ScaleIntensity(minv=0.0, maxv=1.0),
                     SpatialCrop(roi_start=[44, 28], roi_end=[211, 227]),   # 200x168
                     Resize(spatial_size=(100, 84, -1))])

bravo_dataset = NiFTIDataset(data_dir=data_dir, mr_sequence="bravo", transform=transform)
train_dataset = extract_slices_single(bravo_dataset)
train_data_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
device = torch.device("cuda")

model = DiffusionModelUNet(spatial_dims=2,
                           in_channels=1,
                           out_channels=1,
                           num_channels=(128, 256, 256),
                           attention_levels=(False, True, True),
                           num_res_blocks=1,
                           num_head_channels=256)

model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-5)
#optimizer = AdamW8bit(model.parameters(), lr=2.5e-5)
#optimizer = torch.compile(optimizer, mode="reduce-overhead", fullgraph=True)
scheduler = DDPMScheduler(num_train_timesteps=1000)
inferer = DiffusionInferer(scheduler)
scaler = GradScaler('cuda')
total_start = time.time()

for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(enumerate(train_data_loader), total=len(train_data_loader), ncols=70)
    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in progress_bar:
        # Now batch is just the BRAVO images, no need to slice
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
        scaler.update()
        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_epoch_loss = epoch_loss / len(train_data_loader)
    print(f"Epoch {epoch}: Average Loss = {avg_epoch_loss:.6f}")
    if (epoch + 1) % val_interval == 0:
        model.eval()
        path = (r"C:\NTNU\RepoThesis\trained_model\Testing\diffusion_edge_bravo_"
                + str(bs) + "_Epoch" + str(epoch) + "_of_" + str(n_epochs))
        torch.save(model.state_dict(), path)