'''Parts of this code is adapted from the MONAI consortsium's tutorial for realism diversity metrics for generative models. 
https://github.com/Project-MONAI/GenerativeModels/blob/main/tutorials/generative/realism_diversity_metrics/realism_diversity_metrics.ipynb'''
from monai.data import Dataset, DataLoader 
from torch.utils.data import ConcatDataset
import os
import torch
import numpy as np
import monai
import matplotlib.pyplot as plt
from monai.config import print_config
import nibabel as nib
print_config()

from generative.metrics import MultiScaleSSIMMetric, SSIMMetric,  FIDMetric

from helper_functions import merge_data, extract_slices, transform_128, NiFTIDataset

'''Loading real data'''
data_dir_train = "Masteroppgave/Data/BrainMets/StanfordSkullStripped/train"
data_dir_test = "/Masteroppgave/Data/BrainMets/StanfordSkullStrippedTest"

bravo_dataset_train = NiFTIDataset(data_dir= data_dir_train,mr_sequence="bravo", transform = transform_128)
labels_dataset_train = NiFTIDataset(data_dir= data_dir_train,mr_sequence="seg", transform = transform_128)

bravo_dataset_val = NiFTIDataset(data_dir= data_dir_test,mr_sequence="bravo", transform = transform_128)
labels_dataset_val = NiFTIDataset(data_dir= data_dir_test,mr_sequence="seg", transform = transform_128)

merged = merge_data(bravo_dataset_train, labels_dataset_train)
merged_val = merge_data(bravo_dataset_val, labels_dataset_val)

train_dataset = extract_slices(merged, threshold = 1.0)
val_dataset = extract_slices(merged_val, threshold = 1.0)

'''Loading synthetic data'''
syn_data_dir = "Synthetic_images/BrainMets/mask_conditioned_synthesis"
syn_data_list = os.listdir(syn_data_dir)

syn_bravo_images = []
for i in range(len(val_dataset)):
    image = nib.load(syn_data_dir + "/" + syn_data_list[i]).get_fdata() #(128,128,2), first element is bravo image, second element is annotation
    image_MR = image[:,:,0]
    image_MR = torch.unsqueeze(torch.Tensor(image_MR), dim = 0)
    syn_bravo_images.append(image_MR)#(1, 128, 128)


val_loader = DataLoader(val_dataset, batch_size = 32, shuffle = True)
train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
syn_loader = DataLoader(syn_bravo_images, batch_size = 32, shuffle = True)

device = torch.device("cuda")

'''Find SSIM and MS-SSIM between loaders.In this case synthetic images have
been compared to other synthetic images in order to measure the synthetic dataset diversity. Change
based on the need. '''
ms_ssim = MultiScaleSSIMMetric(spatial_dims=2, data_range=1.0, kernel_size=4)
ssim = SSIMMetric(spatial_dims=2, data_range=1.0, kernel_size=4)

ms_ssim_scores = []
ssim_scores = []
for step, batch in list(enumerate(syn_loader)):
    for step_1, batch_1 in list(enumerate(syn_loader)):
        if len(batch) == len(batch_1):
            image = batch.to(device)
            image_syn = batch_1.to(device)
            ms_ssim_scores.append(ms_ssim(image, image_syn).cpu())
            ssim_scores.append(ssim(image, image_syn).cpu())

ms_ssim_recon_scores = torch.cat(ms_ssim_scores, dim=0)
ssim_recon_scores = torch.cat(ssim_scores, dim=0)

ms_ssim_1 = []
ssim_1 = []
for i in range(len(ms_ssim_recon_scores)):
    if ms_ssim_recon_scores[i] !=1.0:
        ms_ssim_1.append(ms_ssim_recon_scores[i].item())
        
for i in range(len(ssim_recon_scores)):
    if ssim_recon_scores[i] !=1.0:
        ssim_1.append(ssim_recon_scores[i].item())

print(np.mean(ms_ssim_1))
print(np.mean(ssim_1))

'''Find FID values. Load radimagenet model to use as inception model '''
radnet = torch.hub.load("Warvito/radimagenet-models", model="radimagenet_resnet50", verbose=True)
radnet.to(device)
radnet.eval()

def subtract_mean(x: torch.Tensor) -> torch.Tensor:
    mean = [0.406, 0.456, 0.485]
    x[:, 0, :, :] -= mean[0]
    x[:, 1, :, :] -= mean[1]
    x[:, 2, :, :] -= mean[2]
    return x

def spatial_average(x: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
    return x.mean([2, 3], keepdim=keepdim)

def get_features(image):
    # If input has just 1 channel, repeat channel to have 3 channels
    if image.shape[1]:
        image = image.repeat(1, 3, 1, 1)

    # Change order from 'RGB' to 'BGR'
    image = image[:, [2, 1, 0], ...]

    # Subtract mean used during training
    image = subtract_mean(image)

    # Get model outputs
    with torch.no_grad():
        feature_image = radnet.forward(image)
        # flattens the image spatially
        feature_image = spatial_average(feature_image, keepdim=False)

    return feature_image

def calculate_fid(dataloader1, dataloader2):
    images1_features = []
    images2_features = []
    
    for step, x in enumerate(dataloader1):
        images1 = x.to(device)

        # Get the features for the first set of images
        eval_feats1 = get_features(images1)
        images1_features.append(eval_feats1)
    
    for step, y in enumerate(dataloader2):
        images2 = y.to(device)

        # Get the features for the second set of images
        eval_feats2 = get_features(images2)
        images2_features.append(eval_feats2)
        
    eval_features1 = torch.vstack(images1_features)
    eval_features1 = eval_features1[0:400] #use sample size 400 to match earlier calculations
    eval_features2 = torch.vstack(images2_features)
    eval_features2 = eval_features2[0:400] #use sample size 400 to match earlier calculations
    
    fid = FIDMetric()
    fid_res = fid(eval_features1, eval_features2)
    return fid_res.item()

fids = []
for i in range(10):
    fid = calculate_fid(syn_loader, train_loader)
    fids.append(fid)

print(np.mean(fids))
