import torch
import numpy as np
import nibabel as nib
from skimage.measure import label, regionprops
from torch.amp import autocast
from tqdm import tqdm
from monai.inferers import DiffusionInferer
from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler
from monai.config import print_config

torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.enabled = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.matmul.allow_tf32 = True
torch._dynamo.config.cache_size_limit = 32


num_images = 15000
batch_size = 200
k = 0
image_size = 128
device = torch.device("cuda")

def make_binary(image, threshold):
    """Convert image to binary using threshold"""
    image_binary = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] > threshold:
                image_binary[i][j] = 1.0
    return image_binary

def load_model(model_path, model_input, image_size):
    """Load trained diffusion model"""
    model = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=model_input,
        out_channels=1,
        channels=(128, 256, 256),
        attention_levels=(False, True, True),
        num_res_blocks=1,
        num_head_channels=256,
    )

    pre_trained_model = torch.load(model_path, map_location=device)
    model.load_state_dict(pre_trained_model, strict=False)
    model.to(device)
    model.eval()

    # Compile for inference optimization
    opt_model = torch.compile(model, mode="default")

    return opt_model

# Load segmentation mask generation model
seg_model_path = r"/home/mode/NTNU/RepoThesis/trained_model/Diffusion_seg_128_20250910-200202/Diffusion_Epoch199_of_200"
seg_model = load_model(seg_model_path, model_input=1, image_size=image_size)

# Load bravo image generation model
bravo_model_path = r"/home/mode/NTNU/RepoThesis/trained_model/Diffusion_bravo_128_20250910-081749/Diffusion_Epoch199_of_200"
bravo_model = load_model(bravo_model_path, model_input=2, image_size=image_size)

# Initialize scheduler and inferer (matching training script settings)
scheduler = DDPMScheduler(num_train_timesteps=1000, schedule='cosine')
inferer = DiffusionInferer(scheduler)

def sample_img(mask_image, counter):
    mask_image = torch.unsqueeze(torch.Tensor(mask_image), dim=0)
    mask = torch.unsqueeze(mask_image, dim=0).to(device)

    # Start with pure noise
    noise_mr = torch.randn((1, 1, image_size, image_size)).to(device)
    current_image = noise_mr

    scheduler.set_timesteps(num_inference_steps=1000)
    progress_bar = tqdm(scheduler.timesteps, desc=f"Generating image {counter}")

    for t in progress_bar:
        combined = torch.cat((current_image, mask), dim=1)

        with autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):

            with torch.no_grad():
                prediction_t = bravo_model(combined, timesteps=torch.Tensor((t,)).to(device))
                current_image, _ = scheduler.step(prediction_t, t, current_image)

        if t % 1000 == 0:
            sampled_image = current_image.cpu()
            sampled_image = torch.unsqueeze(sampled_image, dim=4)
            mask_cpu = mask.cpu()
            mask_cpu = torch.unsqueeze(mask_cpu, dim=4)
            combined_to_save = torch.cat((sampled_image[0, 0], mask_cpu[0, 0]), dim=2)

    # Quality check: ensure lesion placement is valid
    num_pixels_correct = 0
    num_pixels = 0

    brain_mask = make_binary(combined_to_save[:,:,0], threshold = 0.005)
    for i in range(brain_mask.shape[0]):
        for j in range(brain_mask.shape[1]):
            if (combined_to_save[i, j, 1] == 1.0):
                num_pixels += 1
            if (combined_to_save[i, j, 1] == 1.0) and brain_mask[i, j] > 0:
                num_pixels_correct += 1

    # Save only if lesion placement is valid
    if num_pixels == num_pixels_correct and num_pixels > 0:
        nifti_image = nib.Nifti1Image(np.array(combined_to_save), np.eye(4))
        nib.save(nifti_image, fr"/home/mode/NTNU/MedicalDataSets/mask_conditioned_synthesis_v2/{counter}.nii.gz")
        return True

    return False

for i in range(0,num_images,batch_size):
    current_batch_size = min(batch_size, num_images - i)

    # Generate noise for batch
    noise = torch.randn((current_batch_size, 1, image_size, image_size)).to(device)
    scheduler.set_timesteps(num_inference_steps=1000)

    # Generate synthetic annotation masks
    with autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        with torch.no_grad():
            images = inferer.sample(input_noise=noise, diffusion_model=seg_model, scheduler=scheduler)

    for j in range(len(images)):
        image = images[j, 0].detach().cpu().numpy()
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        summen = np.sum(np.abs(image))

        # Filter out distorted images
        if summen < 2000:
            image = make_binary(image, threshold=0.01)

            # Check for metastasis size requirements
            labeled_image, count = label(image, return_num=True)
            objects = regionprops(labeled_image)
            object_areas = [obj["area"] for obj in objects]
            small_met = any(elem < 40 for elem in object_areas)

            if small_met:
                # Generate corresponding bravo image
                success = sample_img(image, counter=k)
                if success:
                    k += 1
                    if k % 100 == 0:
                        print(f"Generated {k} valid image pairs")

        if j % 10 == 0:
            torch.cuda.empty_cache()

        # Clear cache after each batch
        torch.cuda.empty_cache()