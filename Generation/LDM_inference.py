# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Code for sampling both synthetic annotation mask and synthetic BARVO image
num_images = 15000

###importing necessary libraries########
import torch
import torch.nn.functional as F
from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet, AutoencoderKL
from generative.networks.schedulers import DDPMScheduler
from torch.cuda.amp import autocast
from tqdm import tqdm

from skimage.measure import label, regionprops
from monai.config import print_config
import nibabel as nib
import numpy as np

print_config()

device = torch.device("cuda")

#################################################
'''Loading autoencoder for latent space operations'''

autoencoder = AutoencoderKL(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    num_channels=(128, 128, 256),
    latent_channels=4,  # Latent space channels
    num_res_blocks=2,
    attention_levels=(False, False, False),
    with_encoder_nonlocal_attn=False,
    with_decoder_nonlocal_attn=False,
)

autoencoder.to(device)
autoencoder.eval()

# Calculate latent space dimensions
sample_input = torch.randn(1, 1, 128, 128).to(device)
with torch.no_grad():
    latent_sample = autoencoder.encode_stage_2_inputs(sample_input)
latent_shape = latent_sample.shape
print(f"Latent shape: {latent_shape}")
latent_height, latent_width = latent_shape[2], latent_shape[3]

#################################################
'''Loading diffusion model trained to generate synthetic annotation masks in latent space'''

model = DiffusionModelUNet(
    spatial_dims=2,
    in_channels=latent_shape[1],  # Use latent channels
    out_channels=latent_shape[1],  # Output latent channels
    num_channels=(128, 256, 256),
    attention_levels=(False, True, True),
    num_res_blocks=1,
    num_head_channels=256,
)

# Load the checkpoint containing both models (updated to match training code)
modelname = r"C:\NTNU\RepoThesis\trained_model\ldm_syn_seg_256_Epoch4_of_5"  # Updated to match training format
checkpoint = torch.load(modelname)

# Extract and load the diffusion model state dict
model.load_state_dict(checkpoint['diffusion_model'])
model.to(device)
model.eval()

# Extract and load the autoencoder state dict
autoencoder.load_state_dict(checkpoint['autoencoder'])
print("Loaded diffusion model and autoencoder from checkpoint")

###################################################
scheduler = DDPMScheduler(num_train_timesteps=1000)
inferer = DiffusionInferer(scheduler)

####################################################
'''Loading diffusion model trained to generate bravo images from annotation masks in latent space'''

mr_model = DiffusionModelUNet(
    spatial_dims=2,
    in_channels=latent_shape[1] + 1,  # Latent channels + 1 for conditioning mask
    out_channels=latent_shape[1],  # Output latent channels
    num_channels=(128, 256, 256),
    attention_levels=(False, True, True),
    num_res_blocks=1,
    num_head_channels=256,
)

# Load the checkpoint for the conditional model (updated to match training format)
modelname_mr = r"C:\NTNU\RepoThesis\trained_model\ldm_conditional_syn_bravo_from_seg_256_Epoch4_of_5"  # Updated to match training format
checkpoint_mr = torch.load(modelname_mr)

# Extract and load the diffusion model state dict
mr_model.load_state_dict(checkpoint_mr['diffusion_model'])
mr_model.to(device)
mr_model.eval()

print("Loaded conditional diffusion model from checkpoint")

# Create noise in latent space instead of image space
noise_mr_latent = torch.randn((1, latent_shape[1], latent_height, latent_width)).to(device)
scheduler.set_timesteps(num_inference_steps=1000)
progress_bar = tqdm(scheduler.timesteps)


def sample_img(mask_image, counter):
    """Sample MRI image from mask using latent diffusion model"""
    mask_image = torch.unsqueeze(torch.Tensor(mask_image), dim=0)
    mask = torch.unsqueeze(mask_image, dim=0).to(device)

    # Resize mask to match latent spatial dimensions if needed
    if mask.shape[2] != latent_height or mask.shape[3] != latent_width:
        mask = F.interpolate(mask, size=(latent_height, latent_width), mode='nearest')

    current_latent = noise_mr_latent.clone()  # Start with pure noise in latent space
    combined = torch.cat((current_latent, mask), dim=1)  # Concatenate noise and synthetic annotation mask

    for t in progress_bar:  # Go through the denoising process
        with autocast(enabled=False):
            with torch.no_grad():
                prediction_t = mr_model(combined, timesteps=torch.Tensor((t,)).to(current_latent.device))
                current_latent, _ = scheduler.step(prediction_t, t, current_latent)
                combined = torch.cat((current_latent, mask), dim=1)

                if t % 1000 == 0:  # After 1000 denoising steps --> decode and save image
                    # Decode from latent space to image space
                    with torch.no_grad():
                        sampled_image = autoencoder.decode_stage_2_outputs(current_latent)

                    sampled_image = sampled_image.cpu()
                    sampled_image = torch.unsqueeze(sampled_image, dim=4)

                    # Resize mask back to image resolution for saving
                    mask_resized = F.interpolate(mask, size=(128, 128), mode='nearest')
                    mask_resized = mask_resized.cpu()
                    mask_resized = torch.unsqueeze(mask_resized, dim=4)

                    combined_to_save = torch.cat((sampled_image[0, 0], mask_resized[0, 0]), dim=2)
                    print(combined_to_save.shape)

    # Only save image if there is any signal where the synthetic lesion is placed
    num_pixels_correct = 0
    num_pixels = 0
    brain_mask = make_binary(combined_to_save[:, :, 0], threshold=0.005)
    for i in range(brain_mask.shape[0]):
        for j in range(brain_mask.shape[1]):
            if (combined_to_save[i, j, 1] == 1.0):
                num_pixels += 1
            if (combined_to_save[i, j, 1] == 1.0) and brain_mask[i, j] > 0:
                num_pixels_correct += 1

    if num_pixels == num_pixels_correct:
        nifti_image = nib.Nifti1Image(np.array(combined_to_save), np.eye(4))
        nib.save(nifti_image, fr"C:\NTNU\MedicalDataSets\synthetic_dataset_ldm\{counter}.nii.gz")

    return sampled_image


##############Sampling##############################

def make_binary(image, threshold):
    image_binary = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] > threshold:
                image_binary[i][j] = 1.0
    return image_binary


#############Saving images########################
bolk = 70  # Save images in batches for efficiency
k = 0
#########################################

for i in range(0, num_images, bolk):
    # Generate noise in latent space instead of image space
    noise_latent = torch.randn((bolk, latent_shape[1], latent_height, latent_width)).to(device)
    scheduler.set_timesteps(num_inference_steps=1000)

    # Sample synthetic annotation masks in latent space
    with torch.no_grad():
        latent_masks = inferer.sample(input_noise=noise_latent, diffusion_model=model, scheduler=scheduler)

        # Decode latent masks to image space
        synthetic_masks = autoencoder.decode_stage_2_outputs(latent_masks)

    for j in range(len(synthetic_masks)):
        image = synthetic_masks[j, 0].detach().cpu().numpy()
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        summen = np.sum(np.abs(image))

        if summen < 2000:  # Sometimes the sampling process fails --> creates distorted images with a very high sum
            image = make_binary(image, threshold=0.01)

            labeled_image, count = label(image, return_num=True)
            objects = regionprops(labeled_image)
            object_areas = [obj["area"] for obj in objects]
            small_met = False
            for elem in object_areas:
                if elem < 40:  # Checking if there is at least one small metastasis in the image
                    small_met = True

            if small_met:
                # Sampling synthetic bravo image from synthetic annotation mask
                sampled_image = sample_img(image, counter=k)
                k += 1

        if j % 10 == 0:
            torch.cuda.empty_cache()

    # Clear cache after each batch
    torch.cuda.empty_cache()