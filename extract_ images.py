#!/usr/bin/env python3
"""
Extract joint 2-channel BRAVO+Segmentation images from train dataset.
Only extracts slices with pathology. Saves as 2-channel NIfTI files.
"""

import numpy as np
import nibabel as nib
import torch
from pathlib import Path
from tqdm import tqdm
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, ToTensor, ScaleIntensity, Resize, SpatialCrop


def setup_transforms():
    """Setup MONAI transforms"""
    return Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        ToTensor(),
        ScaleIntensity(minv=0.0, maxv=1.0),
        #Resize(spatial_size=(128, 128, -1)),
        #SpatialCrop(roi_start=[44, 28], roi_end=[211, 227])
    ])


def extract_images(data_dir, output_dir):
    """Extract joint BRAVO+segmentation images with pathology"""

    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = setup_transforms()
    patient_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])

    print(f"Processing {len(patient_dirs)} patients...")

    extracted_count = 0

    for patient_dir in tqdm(patient_dirs):
        bravo_path = patient_dir / "bravo.nii.gz"
        seg_path = patient_dir / "seg.nii.gz"

        if not (bravo_path.exists() and seg_path.exists()):
            continue

        try:
            # Load and transform images
            bravo_img = transform(str(bravo_path))
            seg_img = transform(str(seg_path))

            if bravo_img.shape != seg_img.shape:
                continue

            # Process each slice
            for slice_idx in range(bravo_img.shape[-1]):
                bravo_slice = bravo_img[0, :, :, slice_idx].numpy()
                seg_slice = seg_img[0, :, :, slice_idx].numpy()

                # Make binary and check for pathology
                seg_binary = (seg_slice > 0.01).astype(np.float32)

                if np.sum(seg_binary) > 0:  # Has pathology
                    # Stack channels and save
                    joint_data = np.stack([bravo_slice, seg_binary], axis=-1)

                    filename = f"joint_{extracted_count:06d}_{patient_dir.name}_s{slice_idx:03d}.nii.gz"
                    nifti_img = nib.Nifti1Image(joint_data, np.eye(4))
                    nib.save(nifti_img, output_dir / filename)

                    extracted_count += 1

        except Exception as e:
            print(f"Error with {patient_dir.name}: {e}")
            continue

    print(f"Extracted {extracted_count} images with pathology")
    return extracted_count


class JointDataset(torch.utils.data.Dataset):
    """Simple dataset for joint BRAVO+Segmentation images"""

    def __init__(self, data_dir, separate_channels=False):
        self.data_dir = Path(data_dir)
        self.separate_channels = separate_channels
        self.files = sorted(list(self.data_dir.glob("*.nii.gz")))
        print(f"Found {len(self.files)} joint images")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load joint image [H, W, 2]
        joint_data = nib.load(self.files[idx]).get_fdata()

        # Extract channels
        bravo = torch.from_numpy(joint_data[:, :, 0]).float().unsqueeze(0)
        seg = torch.from_numpy(joint_data[:, :, 1]).float().unsqueeze(0)

        if self.separate_channels:
            return bravo, seg
        else:
            return torch.cat([bravo, seg], dim=0)  # [2, H, W]


def create_dataloader(data_dir, batch_size=16, separate_channels=False):
    """Create dataloader for joint images"""
    dataset = JointDataset(data_dir, separate_channels)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


def verify_data(data_dir):
    """Quick verification of extracted data"""
    data_dir = Path(data_dir)
    files = list(data_dir.glob("*.nii.gz"))

    if not files:
        return False

    # Check sample file
    sample = nib.load(files[0]).get_fdata()
    print(f"Found {len(files)} files")
    print(f"Sample shape: {sample.shape} (should be [H, W, 2])")
    print(f"BRAVO range: [{sample[:, :, 0].min():.3f}, {sample[:, :, 0].max():.3f}]")
    print(f"Seg values: {np.unique(sample[:, :, 1])}")
    return True


if __name__ == "__main__":
    # Paths
    TRAIN_DIR = r"C:\NTNU\MedicalDataSets\brainmetshare-3\train"
    OUTPUT_DIR = r"C:\NTNU\MedicalDataSets\brainmetshare-3_extracted_joint_images"

    # Extract if needed
    if not verify_data(OUTPUT_DIR):
        print("Extracting images...")
        extract_images(TRAIN_DIR, OUTPUT_DIR)
        verify_data(OUTPUT_DIR)

    # Test dataloader
    print("\nTesting dataloader:")
    loader = create_dataloader(OUTPUT_DIR, batch_size=4)
    batch = next(iter(loader))
    print(f"Batch shape: {batch.shape}")
    print(f"BRAVO channel: [{batch[:, 0].min():.3f}, {batch[:, 0].max():.3f}]")
    print(f"Seg channel: {torch.unique(batch[:, 1])}")

    print(f"\nReady! Use create_dataloader('{OUTPUT_DIR}') in your training script.")