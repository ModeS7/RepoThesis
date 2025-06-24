"""
This script generates new latent space images using a trained latent diffusion model
and decodes them into pixel space using a trained autoencoder. The generated images are saved as NIfTI files.
"""
import torch
import numpy as np
import os
from torch.amp import autocast
from tqdm import tqdm
from monai.inferers import DiffusionInferer
from monai.networks.nets import DiffusionModelUNet, AutoencoderKL
from monai.networks.schedulers import DDPMScheduler
import nibabel as nib

# Set device
device = torch.device("cuda")

# Load trained latent diffusion model
print("Loading latent diffusion model...")
latent_diffusion_model = DiffusionModelUNet(
    spatial_dims=2,
    in_channels=8,
    out_channels=8,
    channels=(128, 256, 256),
    attention_levels=(False, True, True),
    num_res_blocks=1,
    num_head_channels=256
)

# Load trained model weights
latent_model_path = r"/trained_model/diffusion_edge20250624-130041/diffusion_edge_bravo_latent16_Epoch99_of_100"  # Update path
latent_diffusion_model.load_state_dict(torch.load(latent_model_path))
latent_diffusion_model.to(device)
latent_diffusion_model.eval()

# Load trained autoencoder for decoding
print("Loading autoencoder for decoding...")
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

# Load autoencoder weights
autoencoder_path = r"/trained_model/autoencoder_20250624-022945/autoencoder_edge_bravo_16_Epoch99_of_100"  # Update path
autoencoder.load_state_dict(torch.load(autoencoder_path))
autoencoder.to(device)
autoencoder.eval()

# Setup scheduler and inferer
scheduler = DDPMScheduler(num_train_timesteps=1000, schedule='cosine')
inferer = DiffusionInferer(scheduler)

# Generation parameters
num_images = 15000  # Number of images to generate
batch_size = 128  # Generate in batches to manage memory
output_dir = r"C:\NTNU\MedicalDataSets\LDM_gen"

# Create output directory
os.makedirs(output_dir, exist_ok=True)

print(f"Generating {num_images} images...")

image_counter = 0
num_batches = (num_images + batch_size - 1) // batch_size

with torch.no_grad():
    for batch_idx in tqdm(range(num_batches), desc="Generating batches"):
        # Calculate batch size for this iteration
        current_batch_size = min(batch_size, num_images - batch_idx * batch_size)

        # Generate noise in latent space (8 channels, 16x16)
        noise = torch.randn((current_batch_size, 8, 16, 16), device=device)

        with autocast(device_type="cuda", enabled=False):
            # Generate latent samples using diffusion model
            latent_samples = inferer.sample(
                input_noise=noise,
                diffusion_model=latent_diffusion_model,
                scheduler=scheduler
            )

            # Decode latent samples to pixel space
            decoded_images = autoencoder.decode_stage_2_outputs(latent_samples)

        # Convert to numpy and save immediately
        decoded_images_cpu = decoded_images.cpu().numpy()

        # Save individual images from this batch as NIfTI files
        for i in range(current_batch_size):
            image = decoded_images_cpu[i, 0]  # Remove channel dimension (shape: 128, 128)

            # Normalize image to [0, 1] range and ensure float32
            image_normalized = np.clip(image, 0, 1).astype(np.float32)

            # Add a third dimension for NIfTI (making it 128x128x1)
            image_3d = np.expand_dims(image_normalized, axis=-1)

            # Create NIfTI image and save
            nifti_image = nib.Nifti1Image(image_3d, np.eye(4))
            nib.save(nifti_image, os.path.join(output_dir, f"generated_image_{image_counter:04d}.nii.gz"))

            image_counter += 1

        # Clear GPU memory
        del noise, latent_samples, decoded_images, decoded_images_cpu
        torch.cuda.empty_cache()

print(f"✓ Generated {image_counter} images saved as NIfTI files to {output_dir}")