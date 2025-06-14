#!/usr/bin/env python3
"""
Script to create a union visualization of all brain tissue across the dataset.
White pixels = areas where brain tissue appears in any image
Black pixels = areas that are always empty
"""

import os
import numpy as np
import nibabel as nib
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image


def analyze_individual_crop_sizes(data_dir, threshold=0.01):
    """
    Analyze each image individually to find minimal crop sizes.

    Args:
        data_dir: Path to dataset directory
        threshold: Minimum intensity to consider as brain tissue

    Returns:
        dict: Statistics about individual crop requirements
    """

    data_dir = Path(data_dir)
    patient_dirs = [d for d in data_dir.iterdir() if d.is_dir()]

    print(f"Analyzing individual crop requirements for {len(patient_dirs)} patients...")

    all_crop_sizes = []  # List of (height, width) tuples
    slice_info = []  # Detailed info for each slice

    for i, patient_dir in enumerate(patient_dirs):
        nifti_path = patient_dir / "bravo.nii.gz"

        if not nifti_path.exists():
            continue

        try:
            nifti_img = nib.load(str(nifti_path))
            image_data = nifti_img.get_fdata()

            # Process each slice individually
            for slice_idx in range(image_data.shape[2]):
                slice_2d = image_data[:, :, slice_idx]

                if np.sum(slice_2d) > 0:  # Only process non-empty slices
                    # Find brain tissue in this slice
                    brain_mask = slice_2d > threshold

                    if np.any(brain_mask):
                        # Find bounding box for this individual slice
                        coords = np.where(brain_mask)
                        min_y, max_y = np.min(coords[0]), np.max(coords[0])
                        min_x, max_x = np.min(coords[1]), np.max(coords[1])

                        height = max_y - min_y + 1
                        width = max_x - min_x + 1

                        all_crop_sizes.append((height, width))
                        slice_info.append({
                            'patient': patient_dir.name,
                            'slice': slice_idx,
                            'height': height,
                            'width': width,
                            'max_dim': max(height, width),
                            'area': height * width
                        })

        except Exception as e:
            print(f"Error processing {nifti_path}: {e}")
            continue

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(patient_dirs)} patients")

    if not all_crop_sizes:
        print("No valid slices found!")
        return None

    # Calculate statistics
    heights = [size[0] for size in all_crop_sizes]
    widths = [size[1] for size in all_crop_sizes]
    max_dims = [max(h, w) for h, w in all_crop_sizes]

    # Count how many fit in different sizes
    fits_64 = sum(1 for dim in max_dims if dim <= 64)
    fits_96 = sum(1 for dim in max_dims if dim <= 96)
    fits_128 = sum(1 for dim in max_dims if dim <= 128)
    fits_160 = sum(1 for dim in max_dims if dim <= 160)
    fits_192 = sum(1 for dim in max_dims if dim <= 192)
    fits_224 = sum(1 for dim in max_dims if dim <= 224)
    fits_256 = sum(1 for dim in max_dims if dim <= 256)

    total_slices = len(all_crop_sizes)

    print(f"\nINDIVIDUAL SLICE ANALYSIS ({total_slices:,} slices)")
    print("=" * 60)

    print(f"Height statistics:")
    print(f"  Min: {np.min(heights)}")
    print(f"  Max: {np.max(heights)}")
    print(f"  Mean: {np.mean(heights):.1f}")
    print(f"  Median: {np.median(heights):.1f}")
    print(f"  95th percentile: {np.percentile(heights, 95):.1f}")

    print(f"\nWidth statistics:")
    print(f"  Min: {np.min(widths)}")
    print(f"  Max: {np.max(widths)}")
    print(f"  Mean: {np.mean(widths):.1f}")
    print(f"  Median: {np.median(widths):.1f}")
    print(f"  95th percentile: {np.percentile(widths, 95):.1f}")

    print(f"\nMaximum dimension statistics:")
    print(f"  Min: {np.min(max_dims)}")
    print(f"  Max: {np.max(max_dims)}")
    print(f"  Mean: {np.mean(max_dims):.1f}")
    print(f"  Median: {np.median(max_dims):.1f}")
    print(f"  95th percentile: {np.percentile(max_dims, 95):.1f}")

    print(f"\nCrop size compatibility:")
    print(f"  64x64:   {fits_64:6,} slices ({100 * fits_64 / total_slices:5.1f}%)")
    print(f"  96x96:   {fits_96:6,} slices ({100 * fits_96 / total_slices:5.1f}%)")
    print(f"  128x128: {fits_128:6,} slices ({100 * fits_128 / total_slices:5.1f}%)")
    print(f"  160x160: {fits_160:6,} slices ({100 * fits_160 / total_slices:5.1f}%)")
    print(f"  192x192: {fits_192:6,} slices ({100 * fits_192 / total_slices:5.1f}%)")
    print(f"  224x224: {fits_224:6,} slices ({100 * fits_224 / total_slices:5.1f}%)")
    print(f"  256x256: {fits_256:6,} slices ({100 * fits_256 / total_slices:5.1f}%)")

    # Find optimal sizes for different coverage levels
    coverage_levels = [95, 99, 99.5, 100]
    print(f"\nOptimal crop sizes for different coverage levels:")
    for coverage in coverage_levels:
        required_size = np.percentile(max_dims, coverage)
        # Round up to next multiple of 32 (common in deep learning)
        optimal_size = int(np.ceil(required_size / 32) * 32)
        lost_slices = sum(1 for dim in max_dims if dim > required_size)
        print(f"  {coverage:5.1f}% coverage: {optimal_size:3d}x{optimal_size:3d} (loses {lost_slices:4,} slices)")

    return {
        'total_slices': total_slices,
        'heights': heights,
        'widths': widths,
        'max_dims': max_dims,
        'slice_info': slice_info,
        'fit_counts': {
            64: fits_64, 96: fits_96, 128: fits_128,
            160: fits_160, 192: fits_192, 224: fits_224, 256: fits_256
        }
    }


def create_brain_union_image(data_dir, output_filename="brain_union.png", threshold=0.01):
    """
    Create a union image showing all areas where brain tissue appears.

    Args:
        data_dir: Path to dataset directory
        output_filename: Name for output image
        threshold: Minimum intensity to consider as brain tissue
    """

    data_dir = Path(data_dir)
    patient_dirs = [d for d in data_dir.iterdir() if d.is_dir()]

    print(f"Processing {len(patient_dirs)} patients...")

    # We'll determine the maximum dimensions needed
    max_height = 0
    max_width = 0

    # First pass: find maximum dimensions
    print("First pass: determining image dimensions...")
    for i, patient_dir in enumerate(patient_dirs):
        nifti_path = patient_dir / "bravo.nii.gz"

        if not nifti_path.exists():
            continue

        try:
            nifti_img = nib.load(str(nifti_path))
            image_data = nifti_img.get_fdata()

            # Check each slice
            for slice_idx in range(image_data.shape[2]):
                slice_2d = image_data[:, :, slice_idx]
                if np.sum(slice_2d) > 0:  # Only process non-empty slices
                    max_height = max(max_height, slice_2d.shape[0])
                    max_width = max(max_width, slice_2d.shape[1])

        except Exception as e:
            print(f"Error processing {nifti_path}: {e}")
            continue

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(patient_dirs)} patients")

    print(f"Maximum dimensions found: {max_height} x {max_width}")

    # Create union image (initialize as all black)
    union_image = np.zeros((max_height, max_width), dtype=np.uint8)

    # Second pass: create union of all brain regions
    print("Second pass: creating union image...")
    total_slices_processed = 0

    for i, patient_dir in enumerate(patient_dirs):
        nifti_path = patient_dir / "bravo.nii.gz"

        if not nifti_path.exists():
            continue

        try:
            nifti_img = nib.load(str(nifti_path))
            image_data = nifti_img.get_fdata()

            # Process each slice
            for slice_idx in range(image_data.shape[2]):
                slice_2d = image_data[:, :, slice_idx]

                if np.sum(slice_2d) > 0:  # Only process non-empty slices
                    # Find brain tissue in this slice
                    brain_mask = slice_2d > threshold

                    # Add to union (pad if necessary)
                    h, w = slice_2d.shape
                    union_image[:h, :w] = np.maximum(union_image[:h, :w], brain_mask.astype(np.uint8) * 255)

                    total_slices_processed += 1

        except Exception as e:
            print(f"Error processing {nifti_path}: {e}")
            continue

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(patient_dirs)} patients")

    print(f"Total slices processed: {total_slices_processed}")

    # Find the actual bounding box of brain tissue
    brain_coords = np.where(union_image > 0)
    if len(brain_coords[0]) > 0:
        min_y, max_y = np.min(brain_coords[0]), np.max(brain_coords[0])
        min_x, max_x = np.min(brain_coords[1]), np.max(brain_coords[1])

        brain_height = max_y - min_y + 1
        brain_width = max_x - min_x + 1

        # Count white pixels
        white_pixel_count = np.sum(union_image > 0)
        total_pixels = union_image.shape[0] * union_image.shape[1]
        white_percentage = (white_pixel_count / total_pixels) * 100

        print(f"\nBrain tissue bounding box:")
        print(f"  Position: ({min_y}, {min_x}) to ({max_y}, {max_x})")
        print(f"  Dimensions: {brain_height} x {brain_width}")
        print(f"  Fits in 128x128: {brain_height <= 128 and brain_width <= 128}")
        print(f"  Fits in 256x256: {brain_height <= 256 and brain_width <= 256}")
        print(f"\nPixel statistics:")
        print(f"  White pixels (brain tissue): {white_pixel_count:,}")
        print(f"  Total pixels: {total_pixels:,}")
        print(f"  Brain coverage: {white_percentage:.1f}%")

        # Create visualization with bounding box
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

        # Full image
        ax1.imshow(union_image, cmap='gray')
        ax1.set_title(f'Full Union Image\n{max_height} x {max_width}')
        ax1.grid(True, alpha=0.3)

        # Add 128x128 grid overlay if useful
        if max_height > 128 or max_width > 128:
            for i in range(0, max_height, 128):
                ax1.axhline(y=i, color='red', alpha=0.5, linewidth=1)
            for i in range(0, max_width, 128):
                ax1.axvline(x=i, color='red', alpha=0.5, linewidth=1)

        # Cropped to bounding box
        cropped = union_image[min_y:max_y + 1, min_x:max_x + 1]
        ax2.imshow(cropped, cmap='gray')
        ax2.set_title(f'Cropped to Brain Tissue\n{brain_height} x {brain_width}')

        # Add target size references
        if brain_height <= 128 and brain_width <= 128:
            ax2.text(brain_width // 2, brain_height // 2, '✓ Fits 128x128',
                     ha='center', va='center', color='green', fontsize=12, weight='bold',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        print(f"\nSaved visualization to '{output_filename}'")

        # Also save the raw union image
        union_pil = Image.fromarray(union_image)
        raw_filename = output_filename.replace('.png', '_raw.png')
        union_pil.save(raw_filename)
        print(f"Saved raw union image to '{raw_filename}'")

        # Save cropped version
        cropped_pil = Image.fromarray(cropped)
        cropped_filename = output_filename.replace('.png', '_cropped.png')
        cropped_pil.save(cropped_filename)
        print(f"Saved cropped version to '{cropped_filename}'")

        return {
            'max_dimensions': (max_height, max_width),
            'brain_bbox': (min_y, min_x, max_y, max_x),
            'brain_dimensions': (brain_height, brain_width),
            'fits_128': brain_height <= 128 and brain_width <= 128,
            'total_slices': total_slices_processed,
            'white_pixels': white_pixel_count,
            'total_pixels': total_pixels,
            'brain_coverage_percent': white_percentage
        }

    else:
        print("No brain tissue found!")
        return None


def main():
    # Update this path to your dataset location
    data_dir = r"C:\NTNU\MedicalDataSets\brainmetshare-3\train"

    if not os.path.exists(data_dir):
        print(f"Error: Dataset directory not found: {data_dir}")
        print("Please update the data_dir variable to point to your dataset")
        return

    print("Creating brain tissue union visualization...")
    print("This will show all areas where brain tissue appears across the entire dataset")
    print("White pixels = brain tissue present in any image")
    print("Black pixels = always empty")
    print()

    # First analysis: Union of all images
    union_result = create_brain_union_image(data_dir)

    if union_result:
        print("\n" + "=" * 50)
        print("UNION ANALYSIS SUMMARY")
        print("=" * 50)
        print(f"Dataset max dimensions: {union_result['max_dimensions'][0]} x {union_result['max_dimensions'][1]}")
        print(f"Brain tissue requires: {union_result['brain_dimensions'][0]} x {union_result['brain_dimensions'][1]}")
        print(f"Can fit in 128x128: {'YES ✓' if union_result['fits_128'] else 'NO ✗'}")
        print(f"Total slices analyzed: {union_result['total_slices']}")
        print(f"White pixels (brain): {union_result['white_pixels']:,}")
        print(f"Total pixels: {union_result['total_pixels']:,}")
        print(f"Brain coverage: {union_result['brain_coverage_percent']:.1f}%")

        if union_result['fits_128']:
            print("\n🎉 Union analysis: You can crop to 128x128 without losing brain tissue!")
        else:
            recommended_size = max(union_result['brain_dimensions'])
            # Round up to next power of 2 or common size
            if recommended_size <= 160:
                recommended_size = 160
            elif recommended_size <= 192:
                recommended_size = 192
            elif recommended_size <= 224:
                recommended_size = 224
            elif recommended_size <= 256:
                recommended_size = 256

            print(f"\n⚠️  Union analysis: Brain tissue needs at least {recommended_size}x{recommended_size}")

    print("\n" + "=" * 60)

    # Second analysis: Individual image requirements
    individual_result = analyze_individual_crop_sizes(data_dir)

    if individual_result:
        # Recommendations based on individual analysis
        fit_128_percent = 100 * individual_result['fit_counts'][128] / individual_result['total_slices']
        fit_160_percent = 100 * individual_result['fit_counts'][160] / individual_result['total_slices']
        fit_192_percent = 100 * individual_result['fit_counts'][192] / individual_result['total_slices']

        print(f"\nRECOMMENDATIONS:")
        print(f"================")

        if fit_128_percent >= 95:
            print(f"✅ Use 128x128: covers {fit_128_percent:.1f}% of slices")
            print("   Very few slices would be lost")
        elif fit_128_percent >= 90:
            print(f"⚠️  128x128 covers {fit_128_percent:.1f}% of slices")
            print("   Consider if losing some slices is acceptable")
        else:
            print(f"❌ 128x128 only covers {fit_128_percent:.1f}% of slices")

        if fit_160_percent >= 99:
            print(f"✅ Use 160x160: covers {fit_160_percent:.1f}% of slices")
        elif fit_192_percent >= 99:
            print(f"✅ Use 192x192: covers {fit_192_percent:.1f}% of slices")

        print(f"\nSTRATEGY OPTIONS:")
        print(f"=================")
        print(f"1. Fixed 128x128 cropping: Keep {fit_128_percent:.1f}% of slices")
        print(f"2. Fixed 160x160 cropping: Keep {fit_160_percent:.1f}% of slices")
        print(f"3. Adaptive cropping: Different sizes per image")
        print(f"4. Current resizing: Keep all slices but lose spatial resolution")


if __name__ == "__main__":
    main()