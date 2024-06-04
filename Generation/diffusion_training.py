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
############################################################################3
#This code was adapted from the MONAI consortsium's tutorial for training a 2D DDPM: 
#https://github.com/Project-MONAI/GenerativeModels/blob/main/tutorials/generative/2d_ddpm/2d_ddpm_tutorial.ipynb


# Code for training mask-conditioned diffusion model / training diffusion model for synthesizing annotation masks

'''loading necessary libraries'''
import nibabel as nib
import os
from monai.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from torch.utils.data import ConcatDataset
from monai.transforms import Compose, LoadImage, ToTensor, ScaleIntensity, EnsureChannelFirst, Resize, CenterSpatialCrop, CropForeground


model_input = 2 #model input is 2 for mask-conditioned synthesis, 1 for synthetic annotation mask synthesis

############Diffusion model for synthetic bravo images controlled with annotation masks###########################
#####Preprocessing##############
data_dir = "/Masteroppgave/Data/BrainMets/StanfordSkullStripped/train"
transform = Compose(
    [
        LoadImage(image_only = True), 
        EnsureChannelFirst(), 
        ToTensor(),
        ScaleIntensity(minv = 0.0, maxv = 1.0),
        Resize(spatial_size = (128, 128, -1)), 
    ]
)

'''OBS!! Hent denne inn fra helper_functions heller'''
class NiFTIDataset(Dataset):
    def __init__(self, data_dir, mr_sequence, transform = None):
        self.data_dir = data_dir
        self.data = os.listdir(data_dir) 
        self.data.sort()
        self.mr_sequence = mr_sequence
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        nifti_file = os.path.join(self.data_dir, self.data[index] +  "/" + self.mr_sequence + ".nii.gz")
        if self.transform is not None:
            nifti_file = self.transform(nifti_file)
        return nifti_file, self.data[index] 

bravo_dataset = NiFTIDataset(data_dir= data_dir,mr_sequence="bravo", transform = transform)
seg_dataset = NiFTIDataset(data_dir= data_dir,mr_sequence="seg", transform = transform)

def merge_data(dataset1, dataset2, dataset3):
    dataset_tuple = []
    for i in range(len(dataset1)):
        image_stack_1, image_name_1 = dataset1.__getitem__(index = i) #bravo
        image_stack_2, image_name_2 = dataset2.__getitem__(index = i) #seg

        if image_name_1 == image_name_2:
            combined = np.concatenate((image_stack_1, image_stack_2),axis = 0)
            dataset_tuple.append(combined)

    return Dataset(dataset_tuple)

def make_binary(image, threshold):
    #print(image.shape)
    image_binary = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] > threshold:
                image_binary[i][j] = 1.0
    return image_binary


def extract_slices(nifti_dataset): #make segmentations binary when time
    total_dataset = Dataset([])
    for i in range(len(nifti_dataset)):
            print(i)
            image_stack = nifti_dataset.__getitem__(index = i) 

            images = [image_stack[:,:,:,k] for k in range(image_stack.shape[3])] #each image has shape (2,128,128) , first element is bravo image, second is seg
            non_empty_images = []
            for image in images:
                if np.sum(image) > 1.0:
                    image[1] = make_binary(image[1], threshold = 0.01)
                    non_empty_images.append(image)
            total_dataset = ConcatDataset([total_dataset, images])
    return total_dataset 

merged = merge_data(bravo_dataset, seg_dataset) 
train_dataset = extract_slices(merged)

bs = 16
train_data_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
device = torch.device("cuda")

model = DiffusionModelUNet(
    spatial_dims=2,
    in_channels=model_input, 
    out_channels=1,
    num_channels=(128, 256, 256), 
    attention_levels=(False, True, True),
    num_res_blocks=1,
    num_head_channels=256,
)

model.to(device)

scheduler = DDPMScheduler(num_train_timesteps=1000)

optimizer = torch.optim.Adam(params=model.parameters(), lr=2.5e-5)

inferer = DiffusionInferer(scheduler)

'''Training'''

n_epochs = 200
val_interval = 25
#metric lists
epoch_accuracy_list = []
epoch_loss_list = []

scaler = GradScaler()
total_start = time.time()

for epoch in range(n_epochs):
    model.train()
    #Epoch metrics
    epoch_loss = 0
    epoch_accuracy = 0
    
    progress_bar = tqdm(enumerate(train_data_loader), total=len(train_data_loader), ncols=70)
    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in progress_bar: 
        images = batch.to(device)
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=True):
            if model_input == 1:
                noise = torch.randn_like(images).to(device)
                timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device).long()
                noise_pred = inferer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps)
            if model_input == 2: 
                images_bravo = batch[:,0,:,:].to(device) #to synthesize: bravo
                images_labels = batch[:,1,:,:].to(device)  #conditionning: labels
                images_bravo = torch.unsqueeze(images_bravo, 1)
                images_labels = torch.unsqueeze(images_labels, 1)
                noise = torch.randn_like(images_bravo).to(device)
                timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps, (images_bravo.shape[0],), device=images.device).long()
    
                noisy_bravo = scheduler.add_noise(original_samples = images_bravo, noise = noise, timesteps = timesteps)
                noisy_bravo_w_label = torch.cat((noisy_bravo, images_labels),dim = 1)
                noise_pred = model(x = noisy_bravo_w_label, timesteps = timesteps)

            loss = F.mse_loss(noise_pred.float(), noise.float())
      
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item() 

    if (epoch + 1) % val_interval == 0:
          model.eval()
          path = "/Masteroppgave/Trained_models/BrainMets/Segmentations/conditional_syn_bravo_from_seg_" + str(bs) + "_Epoch" + str(epoch) + "_of_" + str(n_epochs)
          #path = "/Masteroppgave/Trained_models/BrainMets/Segmentations/syn_seg_" + str(bs) + "_Epoch" + str(epoch) + "_of_" + str(n_epochs)
          torch.save(model.state_dict(), path)
        
            
            
            
        

