import os
import sys
import time
import numpy as np
from datetime import datetime
import torch
import torch.nn.functional as F
from monai.data import DataLoader
from monai.transforms import (Compose, LoadImage, ToTensor, ScaleIntensity,
                              EnsureChannelFirst, Resize)
from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from monai.inferers import DiffusionInferer
from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler
from monai.losses import PerceptualLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from functions import NiFTIDataset, extract_slices_single, extract_slices, merge_data

torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.enabled = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.matmul.allow_tf32 = True
torch._dynamo.config.cache_size_limit = 32

def main(image_type):
    # Training configuration
    n_epochs = 200
    val_interval = 10
    bs = 32
    image_size = 128
    show_para = False
    # image_type = "seg"  # "seg" or "bravo"

    # Perceptual loss weight
    perceptual_weight = 0.001

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    sufix = image_type + "_" + str(image_size) + "_" + timestamp
    log_dir = r"/home/mode/NTNU/RepoThesis/tensorboard_logs/Diffusion_" + sufix
    writer = SummaryWriter(log_dir)
    data_dir = r"/home/mode/NTNU/MedicalDataSets/brainmetshare-3/train"
    transform = Compose([LoadImage(image_only=True),
                         EnsureChannelFirst(),
                         ToTensor(),
                         ScaleIntensity(minv=0.0, maxv=1.0),
                         Resize(spatial_size=(image_size, image_size, -1))
                         ])

    bravo_dataset = NiFTIDataset(data_dir=data_dir, mr_sequence="bravo", transform=transform)
    seg_dataset = NiFTIDataset(data_dir=data_dir, mr_sequence="seg", transform=transform)

    if image_type == "seg":
        model_input = 1  # Only segmentation masks
        train_dataset = extract_slices_single(seg_dataset)
    elif image_type == "bravo":
        model_input = 2  # Bravo images with segmentation masks
        # Merge bravo and seg datasets for training
        merged = merge_data(bravo_dataset, seg_dataset)
        train_dataset = extract_slices(merged)

    train_data_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True,
                                   pin_memory=True, num_workers=4)
    device = torch.device("cuda")
    model = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=model_input,
        out_channels=1,
        channels=(128, 256, 256),
        attention_levels=(False, True, True),
        num_res_blocks=1,
        num_head_channels=256
    ).to(device)

    # Initialize perceptual loss - same as autoencoder implementation
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cache_dir = os.path.join(project_root, "model_cache")
    perceptual_loss_fn = PerceptualLoss(
        spatial_dims=2,
        network_type="radimagenet_resnet50",
        cache_dir=cache_dir,
        pretrained=True,
    ).to(device)

    # Count model parameters
    if show_para:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model Parameter Summary:")
        print(f"  Total parameters: {total_params:,}")
        if trainable_params != total_params:
            print(f"  Trainable parameters: {trainable_params:,}")
        print("-" * 50)

    # Compile the model for performance optimization
    opt_model = torch.compile(model, mode="default")
    opt_perceptual_loss = torch.compile(perceptual_loss_fn, mode="reduce-overhead")

    optimizer = AdamW(model.parameters(), lr=1e-4)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)
    scheduler = DDPMScheduler(num_train_timesteps=1000, schedule='cosine')
    inferer = DiffusionInferer(scheduler)

    def predict_clean_image(noisy_image, noise_pred, timesteps, scheduler):
        """Predict clean image from noisy image and noise prediction"""
        # Get alpha values for the timesteps
        alphas_cumprod = scheduler.alphas_cumprod.to(device)
        alpha_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
        # Predict clean image: x_0 = (x_t - sqrt(1-alpha_t) * noise) / sqrt(alpha_t)
        predicted_clean = (noisy_image - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
        return torch.clamp(predicted_clean, 0, 1)

    def generate_validation_samples(epoch, num_samples=4, image_type="seg"):
        """Generate and log validation samples to TensorBoard"""
        model.eval()
        try:
            with torch.no_grad():
                if image_type == "seg":
                    model_input = torch.randn((num_samples, 1, image_size, image_size), device=device)
                elif image_type == "bravo":
                    indices = torch.randint(0, len(train_dataset), (num_samples,))
                    seg_masks = []
                    for i in indices:
                        data = train_dataset[i.item()]
                        tensor = torch.from_numpy(data).float() if isinstance(data, np.ndarray) else torch.tensor(
                            data).float()
                        seg_masks.append(tensor[1:2, :, :])

                    seg_masks = torch.stack(seg_masks).to(device)
                    bravo_noise = torch.randn_like(seg_masks, device=device)
                    model_input = torch.cat([bravo_noise, seg_masks], dim=1)

                with autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                    samples = inferer.sample(input_noise=model_input, diffusion_model=opt_model, scheduler=scheduler)

                samples_to_log = samples[:, 0:1, :, :] if image_type == "bravo" else samples
                samples_normalized = torch.clamp(samples_to_log.float(), 0, 1)
                writer.add_images(f'Generated_Images', samples_normalized, epoch)

        except Exception as e:
            print(f"Warning: Failed to generate validation samples at epoch {epoch}: {e}")
        finally:
            torch.cuda.empty_cache()
            model.train()

    total_start = time.time()
    for epoch in range(n_epochs):
        model.train()
        (epoch_loss, epoch_mse_loss, epoch_perceptual_loss) = 0, 0, 0
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
                if image_type == "seg":
                    noise = torch.randn_like(images).to(device)
                    timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps,
                                              (images.shape[0],), device=images.device).long()
                    noise_pred = inferer(inputs=images, diffusion_model=opt_model, noise=noise, timesteps=timesteps)
                    mse_loss = F.mse_loss(noise_pred.float(), noise.float())
                    noisy_images = scheduler.add_noise(original_samples=images, noise=noise, timesteps=timesteps)
                    predicted_clean = predict_clean_image(noisy_images, noise_pred, timesteps, scheduler)
                    p_loss = opt_perceptual_loss(predicted_clean.float(), images.float())

                if image_type == "bravo":
                    images_bravo = batch[:, 0, :, :].to(device)  # to synthesize: bravo
                    images_labels = batch[:, 1, :, :].to(device)  # conditioning: labels
                    images_bravo = torch.unsqueeze(images_bravo, 1)
                    images_labels = torch.unsqueeze(images_labels, 1)
                    noise = torch.randn_like(images_bravo).to(device)
                    timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps,
                                              (images_bravo.shape[0],), device=images.device).long()
                    noisy_bravo = scheduler.add_noise(original_samples=images_bravo,
                                                      noise=noise, timesteps=timesteps)
                    noisy_bravo_w_label = torch.cat((noisy_bravo, images_labels), dim=1)
                    noise_pred = opt_model(x=noisy_bravo_w_label, timesteps=timesteps)
                    mse_loss = F.mse_loss(noise_pred.float(), noise.float())
                    predicted_clean = predict_clean_image(noisy_bravo, noise_pred, timesteps, scheduler)
                    p_loss = opt_perceptual_loss(predicted_clean.float(), images_bravo.float())

                # Combined loss
                loss = mse_loss + perceptual_weight * p_loss

            loss.backward()
            optimizer.step()
            epoch_mse_loss += mse_loss.item()
            epoch_perceptual_loss += p_loss.item()
            optimizer.zero_grad(set_to_none=True)
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=f"{epoch_loss / (step + 1):.6f}")

        lr_scheduler.step()
        # Calculate average losses per effective batch
        avg_epoch_loss = epoch_loss
        avg_mse_loss = epoch_mse_loss / len(train_data_loader)
        avg_perceptual_loss = epoch_perceptual_loss / len(train_data_loader)
        # Log losses separately for comparison
        writer.add_scalar('Loss/Total', avg_epoch_loss, epoch)
        writer.add_scalar('Loss/MSE', avg_mse_loss, epoch)
        writer.add_scalar('Loss/Perceptual', avg_perceptual_loss, epoch)
        print(f"Epoch {epoch}: Total Loss = {avg_epoch_loss:.6f}, MSE = {avg_mse_loss:.6f}, "
              f"Perceptual = {avg_perceptual_loss:.6f}")

        if (epoch + 1) % val_interval == 0:
            newpath = f'/home/mode/NTNU/RepoThesis/trained_model/Diffusion_{sufix}'
            os.makedirs(newpath, exist_ok=True)
            print(f"Generating validation samples...")
            generate_validation_samples(epoch, image_type=image_type)
            model.eval()
            path = f"{newpath}/Diffusion_Epoch{epoch}_of_{n_epochs}"
            torch.save(model.state_dict(), path)
            print(f"✓ Saved model at epoch {epoch})")

    total_time = time.time() - total_start
    print(f"Training completed in {total_time:.2f} seconds ({total_time / 3600:.2f} hours)")
    writer.close()
    print("✓ Training completed! Check TensorBoard for metrics.")

if __name__ == "__main__":
    list = ["bravo", "seg"]
    for image_type in list:
        print(f"Starting training for {image_type} images...")
        main(image_type)
        print(f"Training for {image_type} images completed.\n")