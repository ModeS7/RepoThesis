"""
This script generates latent representations from a trained AutoencoderKL model
for a dataset of brain MRI images.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
from monai.data import DataLoader
from monai.networks.nets import AutoencoderKL
from functions import NiFTIDataset, extract_slices_single
from monai.transforms import (Compose, LoadImage, ToTensor, ScaleIntensity,
                              EnsureChannelFirst, Resize)
from tqdm import tqdm

bs = 16

# Set device
device = torch.device("cuda")

# Load trained autoencoder
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

# Load your trained model weights
model_path = r"/trained_model/autoencoder_20250624-022945/autoencoder_edge_bravo_16_Epoch99_of_100"  # Update with your model path
autoencoder.load_state_dict(torch.load(model_path))
autoencoder.to(device)
autoencoder.eval()

# Data setup (same as your original script)
data_dir = r"C:\NTNU\MedicalDataSets\brainmetshare-3\train"
transform = Compose([LoadImage(image_only=True),
                     EnsureChannelFirst(),
                     ToTensor(),
                     ScaleIntensity(minv=0.0, maxv=1.0),
                     Resize(spatial_size=(128, 128, -1)),
                     ])

dataset = NiFTIDataset(data_dir=data_dir, mr_sequence="bravo", transform=transform)
train_dataset = extract_slices_single(dataset)
train_data_loader = DataLoader(train_dataset, batch_size=bs, shuffle=False, pin_memory=True)

# Generate latent representations
latent_representations = []
print("Generating latent space images...")

with torch.no_grad():
    for batch in tqdm(train_data_loader):
        images = batch.to(device)

        # Encode to latent space
        z_mu, z_sigma = autoencoder.encode(images)
        # Use the mean (you could also sample)
        latent_z = z_mu

        latent_representations.append(latent_z.cpu().numpy())

# Concatenate all latent representations
all_latents = np.concatenate(latent_representations, axis=0)
print(f"Generated {all_latents.shape[0]} latent representations with shape {all_latents.shape[1:]}")

# Save latent representations
output_path = r"C:\NTNU\MedicalDataSets\latent_data\bravo_latents.npy"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
np.save(output_path, all_latents)
print(f"✓ Saved latent representations to {output_path}")