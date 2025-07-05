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
from torch import GradScaler
from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from monai.inferers import DiffusionInferer
from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW, Adam

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from functions import NiFTIDataset, extract_slices_single, extract_slices, merge_data

"""torch.backends.cudnn.allow_tf32 = True
#torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.enabled = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')"""
torch._dynamo.config.cache_size_limit = 32

def main(image_type):
    # Training configuration
    n_epochs = 200
    val_interval = 5
    scale = 1
    bs = 16 * scale
    image_size = 128
    show_para = False  # Set to True to print model parameters
    #image_type = "seg"  # "seg" or "bravo"

    accumulation_schedule = {
        0: 1,  # First 30 epochs: accumulate every 1 steps (effective batch size: 16)
        #30: 2,  # Epochs 30-60: accumulate every 2 steps (effective batch size: 32)
        #60: 4,  # Epochs 60-100: accumulate every 4 steps (effective batch size: 64)
    }

    def get_accumulation_steps(epoch):
        """Get current accumulation steps based on epoch"""
        current_steps = 1
        for schedule_epoch in sorted(accumulation_schedule.keys()):
            if epoch >= schedule_epoch:
                current_steps = accumulation_schedule[schedule_epoch]
        return current_steps

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    sufix = image_type + "_" + str(image_size) + "_" + timestamp
    log_dir = r"/home/mode/NTNU/RepoThesis/tensorboard_logs/Diffusion_default_" + sufix
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
    if image_type == "bravo":
        model_input = 2 # Bravo images with segmentation masks
        # Merge bravo and seg datasets for training
        merged = merge_data(bravo_dataset, seg_dataset)
        train_dataset = extract_slices(merged)
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

    # Count model parameters
    if show_para:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model Parameter Summary:")
        print(f"  Total parameters: {total_params:,}")
        if trainable_params != total_params:
            print(f"  Trainable parameters: {trainable_params:,}")
        print("-" * 50)

    # Compile the model for performance optimization. Made training at
    # least 33% faster or more:"default", "reduce-overhead", "max-autotune"
    opt_model = torch.compile(model, mode="default")
    max_accumulation = max(accumulation_schedule.values())
    lr = np.sqrt(scale * max_accumulation)
    #optimizer = AdamW(model.parameters(), lr=1e-4 * lr)
    optimizer = Adam(model.parameters(), lr=2.5e-5)
    #lr_scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5 * lr)
    #scheduler = DDPMScheduler(num_train_timesteps=1000, schedule='cosine')
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    inferer = DiffusionInferer(scheduler)
    scaler = GradScaler('cuda')

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

                with autocast(device_type="cuda", enabled=True):
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
        current_accumulation_steps = get_accumulation_steps(epoch)
        epoch_loss, accumulated_loss, effective_batch_count = 0, 0, 0
        progress_bar = tqdm(enumerate(train_data_loader), total=len(train_data_loader), ncols=100)
        progress_bar.set_description(f"Epoch {epoch} (acc_steps: {current_accumulation_steps})")
        for step, batch in progress_bar:
            # Saw 5% improvement with this + have to have for torch.compile to work.
            if image_type == "seg":
                if hasattr(batch, 'as_tensor'):
                    images = batch[:, 1:2, :, :].as_tensor().to(device)
                else:
                    images = batch[:, 1:2, :, :].to(device)
            if image_type == "bravo":
                if hasattr(batch, 'as_tensor'):
                    images = batch.as_tensor().to(device)
                else:
                    images = batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            #with autocast('cuda', enabled=True, dtype=torch.bfloat16):
            with autocast('cuda', enabled=True):

                if image_type == "seg":
                    noise = torch.randn_like(images).to(device)
                    timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps,
                                              (images.shape[0],), device=images.device).long()
                    noise_pred = inferer(inputs=images, diffusion_model=opt_model, noise=noise, timesteps=timesteps)
                if image_type == "bravo":
                    images_bravo = batch[:, 0, :, :].to(device)  # to synthesize: bravo
                    images_labels = batch[:, 1, :, :].to(device)  # conditionning: labels
                    images_bravo = torch.unsqueeze(images_bravo, 1)
                    images_labels = torch.unsqueeze(images_labels, 1)
                    noise = torch.randn_like(images_bravo).to(device)
                    timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps,
                                              (images_bravo.shape[0],), device=images.device).long()
                    noisy_bravo = scheduler.add_noise(original_samples=images_bravo,
                                                      noise=noise, timesteps=timesteps)
                    noisy_bravo_w_label = torch.cat((noisy_bravo, images_labels), dim=1)
                    noise_pred = opt_model(x=noisy_bravo_w_label, timesteps=timesteps)

                loss = F.mse_loss(noise_pred.float(), noise.float())
                loss = loss / current_accumulation_steps

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            #loss.backward()
            accumulated_loss += loss.item()
            # Update weights conditionally based on current accumulation steps
            if (step + 1) % current_accumulation_steps == 0 or (step + 1) == len(train_data_loader):
                #optimizer.step()
                #optimizer.zero_grad(set_to_none=True)
                # Log accumulated loss
                epoch_loss += accumulated_loss
                accumulated_loss = 0
                effective_batch_count += 1
                # Update progress bar with effective batch loss
                if effective_batch_count > 0:
                    avg_loss = epoch_loss / effective_batch_count
                    progress_bar.set_postfix(
                        loss=f"{avg_loss:.6f}",
                    )

        #lr_scheduler.step()

        # Calculate average loss per effective batch
        avg_epoch_loss = epoch_loss / effective_batch_count if effective_batch_count > 0 else 0
        writer.add_scalar('Loss/Training', avg_epoch_loss, epoch)
        writer.add_scalar('Accumulation_Steps', current_accumulation_steps, epoch)
        print(f"Epoch {epoch}: Loss = {avg_epoch_loss:.6f}, Acc Steps = {current_accumulation_steps}")
        if (epoch + 1) % val_interval == 0:
            newpath = f'/home/mode/NTNU/RepoThesis/trained_model/Diffusion_default_{sufix}'
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
    list = ["bravo","seg"]
    for image_type in list:
        print(f"Starting training for {image_type} images...")
        main(image_type)
        print(f"Training for {image_type} images completed.\n")
