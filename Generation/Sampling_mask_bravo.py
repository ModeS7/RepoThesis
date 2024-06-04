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

#Code for sampling both synthetic annotation mask and synthetic BARVO image
num_images = 15000

###importing necessary libraries########
import torch
from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler
from torch.cuda.amp import autocast
from tqdm import tqdm

from skimage.measure import label,regionprops
from monai.config import print_config
import nibabel as nib
import numpy as np
print_config()
#################################################
'''Loading diffusion model trained to generate synthetic annotation masks'''

model = DiffusionModelUNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    num_channels=(128, 256, 256), 
    attention_levels=(False, True, True),
    num_res_blocks=1,
    num_head_channels=256,
)
device = torch.device("cuda")
modelname = "/Masteroppgave/Trained_models/BrainMets/Segmentations/correct_validation_data_16_Epoch149_of_200" 
pre_trained_model = torch.load(modelname)
model.load_state_dict(pre_trained_model, strict = False) 
model.to(device)
###################################################
scheduler = DDPMScheduler(num_train_timesteps=1000)
inferer = DiffusionInferer(scheduler)
####################################################
'''Loading diffusion model trained to generate bravo images from annotation masks'''

mr_model = DiffusionModelUNet(
    spatial_dims=2,
    in_channels=2,
    out_channels=1,
    num_channels=(128, 256, 256),
    attention_levels=(False, True, True),
    num_res_blocks=1,
    num_head_channels=256,
)
device = torch.device("cuda")
#modelname_mr = "/Masteroppgave/Trained_models/BrainMets/2_1_3_1/with_seg_bs16_Epoch149_of_200"
modelname_mr = "/Masteroppgave/Trained_models/BrainMets/Segmentations/conditional_syn_26mars_16_Epoch149_of_200"
pre_trained_model_mr = torch.load(modelname_mr)
mr_model.load_state_dict(pre_trained_model_mr, strict = False) 
mr_model.to(device)

noise_mr = torch.randn((1,1,128,128)).to(device)
scheduler.set_timesteps(num_inference_steps=1000)
progress_bar = tqdm(scheduler.timesteps)

def sample_img(mask_image, counter):
    mask_image = torch.unsqueeze(torch.Tensor(mask_image), dim = 0)
    mask = torch.unsqueeze(mask_image, dim = 0).to(device)
    current_image = noise_mr #start with pure noise
    combined = torch.cat((noise_mr, mask), dim = 1) #concatenate noise and synthetic annotation mask
    
    for t in progress_bar:  # go through the noising process
        with autocast(enabled=False):
            with torch.no_grad():
                prediction_t = mr_model(combined, timesteps=torch.Tensor((t,)).to(current_image.device))
                current_image, _ = scheduler.step(prediction_t,t,current_image)
                combined = torch.cat((current_image, mask), dim=1)
                if t % 1000 == 0: #after 1000 denoising steps --> save image
                    sampled_image = current_image.cpu()
                    sampled_image = torch.unsqueeze(sampled_image, dim = 4)
                    mask = mask.cpu()
                    mask = torch.unsqueeze(mask, dim = 4)
                    combined_to_save = torch.cat((sampled_image[0,0], mask[0,0]), dim = 2)
                    print(combined_to_save.shape)
    #Only save image if there is any signal where the synthetic lesion is placed 
    num_pixels_correct = 0
    num_pixels = 0
    brain_mask = make_binary(combined_to_save[:,:,0], threshold = 0.005)
    for i in range(brain_mask.shape[0]):
        for j in range(brain_mask.shape[1]):
            if (combined_to_save[i, j, 1] == 1.0):
                num_pixels += 1
            if (combined_to_save[i, j, 1] == 1.0) and brain_mask[i, j] > 0:
                num_pixels_correct += 1

    if num_pixels == num_pixels_correct:
        nifti_image = nib.Nifti1Image(np.array(combined_to_save),np.eye(4))
        nib.save(nifti_image, "/Masteroppgave/Synthetic_images/BrainMets/only_small_mets/file_2_" + str(counter)+ ".nii.gz")
    
    return sampled_image

##############Sampling##############################

def make_binary(image, threshold):
    #print(image.shape)
    image_binary = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] > threshold:
                image_binary[i][j] = 1.0
    return image_binary

#############Saving images########################
bolk = 100 #lagrer bildene bolkvis --> fortere
k = 0
#########################################

for i in range(0,num_images,bolk):
    noise = torch.randn((bolk, 1, 128, 128)) #Generating 100 images with the shape (1, 128, 128) from noise
    noise = noise.to(device)
    scheduler.set_timesteps(num_inference_steps=1000)

    # sample 100 synthetic annotation masks
    images = inferer.sample(input_noise = noise, diffusion_model = model, scheduler = scheduler)
    for j in range(len(images)):
        image = images[j, 0].detach().cpu().numpy()
        image = (image-np.min(image))/(np.max(image)-np.min(image))
        summen = 0
        summen = np.sum(np.abs(image))
        if summen < 2000:# Sometimes the sampling process fails --> creates distorted images with a very high sum. This is just to avoid these images. 
            image = make_binary(image, threshold = 0.01)
            
            labeled_image,count = label(image, return_num=True)
            objects = regionprops(labeled_image)
            object_areas = [obj["area"] for obj in objects]
            small_met = False 
            for elem in object_areas:
                if elem < 40: # Checking if there is at least one small metastasis in the image. Only sampling if this is the case. Set small_met to True to circumvent this part of the code. 
                    small_met = True

            if small_met: 
                #Sampling synthetic bravo image from synthetic annotation mask
                sampled_image = sample_img(image, counter = k)
                k+=1







