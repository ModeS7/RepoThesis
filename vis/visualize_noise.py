#!/usr/bin/env python3
"""
Single image diffusion visualization script.
Shows how noise is progressively added to one image at timesteps: 0,100,200,300,400,500,600,700,800,900,1000
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from monai.data import Dataset
from monai.transforms import Compose, LoadImage, ToTensor, ScaleIntensity, EnsureChannelFirst, Resize
from generative.networks.schedulers import DDPMScheduler

# CONFIGURATION - CHANGE THESE VALUES
DATA_DIR = r"C:\NTNU\MedicalDataSets\brainmetshare-3\train"
MODE = "bravo"  # Change to "bravo" or "seg"
PATIENT_INDEX = 13  # Which patient folder to use
SLICE_INDEX = 76  # Which slice from the 3D volume
DEVICE = torch.device("cpu")  # Use CPU - much simpler for this task!


def load_single_image(data_dir, mode, patient_index, slice_index):
    """Load a single 2D slice from specified patient and slice index."""

    # Get patient folders
    patient_folders = sorted(os.listdir(data_dir))
    if patient_index >= len(patient_folders):
        raise ValueError(f"Patient index {patient_index} >= number of patients {len(patient_folders)}")

    patient_folder = patient_folders[patient_index]
    nifti_path = os.path.join(data_dir, patient_folder, f"{mode}.nii.gz")

    # Setup transform
    transform = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        ToTensor(),
        ScaleIntensity(minv=0.0, maxv=1.0),
        Resize(spatial_size=(128, 128, -1))
    ])

    # Load 3D volume
    volume = transform(nifti_path)
    print(f"Loaded volume shape: {volume.shape}")

    # Extract single slice
    if slice_index >= volume.shape[-1]:
        print(f"Warning: slice_index {slice_index} >= volume depth {volume.shape[-1]}")
        slice_index = volume.shape[-1] // 2  # Use middle slice
        print(f"Using middle slice: {slice_index}")

    image_slice = volume[0, :, :, slice_index]  # Remove channel and take slice

    # For segmentation masks, make binary
    if mode == "seg":
        image_slice = (image_slice > 0.01).float()

    print(f"Selected slice shape: {image_slice.shape}")
    print(f"Image range: [{image_slice.min():.3f}, {image_slice.max():.3f}]")
    print(f"Non-zero pixels: {torch.sum(image_slice > 0).item()}")

    return image_slice, patient_folder


def visualize_diffusion_timesteps(image, save_name):
    """Visualize diffusion process at specified timesteps."""

    timesteps = [0, 99, 199, 299, 399, 499, 599, 699, 799, 899, 999]

    # Add batch and channel dimensions if needed
    if len(image.shape) == 2:
        image = image.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    elif len(image.shape) == 3:
        image = image.unsqueeze(0)  # Add batch dim

    print(f"Image tensor shape: {image.shape}")

    # Initialize scheduler
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    print(f"Scheduler timesteps: 0 to {scheduler.num_train_timesteps - 1}")

    # Create visualization
    fig, axes = plt.subplots(2, 6, figsize=(20, 7))
    axes = axes.flatten()

    print(f"Visualizing {len(timesteps)} timesteps...")

    for i, t in enumerate(timesteps):
        if t == 0:
            # Original image
            noisy_image = image
        else:
            # Add noise at timestep t
            noise = torch.randn_like(image)
            timestep_tensor = torch.tensor([t], dtype=torch.long)
            noisy_image = scheduler.add_noise(
                original_samples=image,
                noise=noise,
                timesteps=timestep_tensor
            )

        # Convert to numpy for visualization
        img_np = noisy_image[0, 0].numpy()

        # Plot
        axes[i].imshow(img_np, cmap='gray')
        axes[i].set_title(f'Step {t}', fontsize=12)
        axes[i].axis('off')

        # Print statistics for first few timesteps
        if i < 4:
            print(f"Timestep {t}: min={img_np.min():.3f}, max={img_np.max():.3f}, std={img_np.std():.3f}")

    # Hide the last empty subplot
    axes[-1].axis('off')

    plt.suptitle(f'Diffusion Process: {save_name}', fontsize=16, y=0.98)
    plt.tight_layout()

    # Save figure
    save_path = f"diffusion_{save_name}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved as: {save_path}")

    plt.show()
    return fig


def main():
    """Main execution function."""

    print(f"Configuration:")
    print(f"- Data directory: {DATA_DIR}")
    print(f"- Mode: {MODE}")
    print(f"- Patient index: {PATIENT_INDEX}")
    print(f"- Slice index: {SLICE_INDEX}")
    print(f"- Device: {DEVICE} (CPU is perfect for this task!)")
    print("-" * 50)

    try:
        # Load single image
        image, patient_name = load_single_image(
            DATA_DIR, MODE, PATIENT_INDEX, SLICE_INDEX
        )

        # Create save name
        save_name = f"{MODE}_patient{PATIENT_INDEX}_slice{SLICE_INDEX}"

        print(f"Processing patient: {patient_name}")

        # Visualize diffusion process
        fig = visualize_diffusion_timesteps(image, save_name)

        print("Done!")

    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    main()