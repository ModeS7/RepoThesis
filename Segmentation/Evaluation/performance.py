'''Code to find performance metrics of the different models. Additionnally, it is possible to compare models to each other. As such, it is possible to investigate which areas 
are detected by one model and undetected by another. Dice per lesion, sensitivity and number of FP are found for cohorts 1-5. '''

from monai.data import Dataset, DataLoader, decollate_batch, CacheDataset
from torch.utils.data import ConcatDataset
from monai.transforms import Compose, LoadImage, ToTensor, ScaleIntensity, EnsureChannelFirst, Resize, CenterSpatialCrop, Activations,AsDiscrete, RandScaleIntensity, RandShiftIntensity, RandFlip, CropForeground, RandRotate90, RandSpatialCrop, RandRotate, RandAdjustContrast, RandHistogramShift, RandSpatialCrop, SpatialPad
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.inferers import sliding_window_inference
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from skimage.measure import label,regionprops
from skimage import measure
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import cv2

from helper_functions import NiFTIDataset, transform_train_mask, extract_slices, merge_data
from performance_functions import dice_per_lesion


data_dir_train = "Masteroppgave/Data/BrainMets/StanfordSkullStripped/train"
data_dir_test = "Masteroppgave/Data/BrainMets/StanfordSkullStrippedTest"

bravo_dataset_val = NiFTIDataset(data_dir= data_dir_test,mr_sequence="bravo", transform = transform_train_mask)
labels_dataset_val = NiFTIDataset(data_dir= data_dir_test,mr_sequence="seg", transform = transform_train_mask)

merged_val = merge_data(bravo_dataset_val, labels_dataset_val)
val_dataset = extract_slices(merged_val, 1.0)

'''Dividing validation dataset into cohorts'''

small_mets = []
medium_mets = []
only_small_mets = []
only_medium_mets = []
for i in range(len(val_dataset)):
    image = val_dataset.__getitem__(i) 
    labeled_image,count = label(image[1,:,:], return_num=True) 
    objects = regionprops(labeled_image)
    object_areas = [obj["area"] for obj in objects] 
    m = 0
    for elem in object_areas:
        if elem < 40 and len(object_areas) == 1 and m == 0: 
            only_medium_mets.append(image)
            m = 1
        elif elem < 40 and m == 0: 
            medium_mets.append(image)
            m = 1      
    k = 0
    for elem in object_areas:
        if elem < 20 and len(object_areas) == 1 and k == 0:
            only_small_mets.append(image)
            k = 1
        elif elem < 20 and k == 0:
            small_mets.append(image)
            k = 1

small_mets = CacheDataset(data = small_mets + only_small_mets)
only_small_mets = CacheDataset(data = only_small_mets)
medium_mets = CacheDataset(data = medium_mets + only_medium_mets)
only_medium_mets = CacheDataset(data = only_medium_mets)

'''Load trained models'''

model_real_data = "Masteroppgave/Trained_models/BrainMets/Runs_to_evaluate/6april_best_modelUNet_20000_5em4_dropout_0.2"
model_1 = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
    dropout=0.2,
)
device = torch.device("cpu")
pre_trained_model = torch.load(model_real_data, map_location = "cpu")
model_1.load_state_dict(pre_trained_model, strict = False) 
model_1 = model_1.to(device)

model_for_testing ="Masteroppgave/Trained_models/BrainMets/Runs_to_evaluate/6april_best_modelUNet_25000_5em4_dropout_0.2"
model_2 = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
    dropout=0.2,
)
device = torch.device("cpu")
pre_trained_model_2 = torch.load(model_for_testing, map_location = "cpu")
model_2.load_state_dict(pre_trained_model_2, strict = False) 
model_2 = model_2.to(device)

model_1.eval()
model_2.eval()

def sensitivity(model_1, model_2, input_dataset):
    model_1_undetected = []
    model_2_undetected = []

    model_1_undetected_total = []
    model_2_undetected_total = []

    undetected_by_both = []

    dice_per_lesion_1 = []
    dice_per_lesion_2 = []

    fp_list_1 = 0
    fp_list_2 = 0

    num_mets = 0

    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([AsDiscrete(to_onehot=2)])
    for i in range(len(input_dataset)):
        image = input_dataset.__getitem__(i)
        image_MR = image[0]
        image_mask = image[1]

        bravo_input_val = torch.Tensor(image_MR).to(device)
        bravo_input_val = torch.unsqueeze(bravo_input_val, dim =0)
        bravo_input_val = torch.unsqueeze(bravo_input_val, dim =0)
        val_input_images = bravo_input_val

        val_labels = torch.Tensor(image_mask).to(device)
        val_labels = torch.unsqueeze(val_labels, dim = 0)
        val_labels = torch.unsqueeze(val_labels, dim = 0)

        roi_size = (160, 160)
        sw_batch_size = 1
        model_output = sliding_window_inference(val_input_images, roi_size, sw_batch_size, model_1)

        val_outputss = [post_pred(i) for i in decollate_batch(model_output)]
        val_labelss = [post_label(i) for i in decollate_batch(val_labels)]

        model_output_2 = sliding_window_inference(val_input_images, roi_size, sw_batch_size, model_2)

        val_outputss_2 = [post_pred(i) for i in decollate_batch(model_output_2)]
        
        #Find dice per lesion for both model outputs
        for i in range(len(val_outputss)):
            dice_per_lesion_list_1 = dice_per_lesion( y= val_labelss[i][1,:,:], y_pred = val_outputss[i][1,:,:])
            for elem in dice_per_lesion_list_1:
                dice_per_lesion_1.append(elem)
            
        for j in range(len(val_outputss_2)):
            dice_per_lesion_list_2 = dice_per_lesion( y= val_labelss[j][1,:,:], y_pred = val_outputss_2[j][1,:,:])
            for elem in dice_per_lesion_list_2:
                dice_per_lesion_2.append(elem)
        
        #image mask is true segmentation mask
        image_a_binary = np.array(val_labelss[0][1])


        #images b and c are predicted segmentation mask times true segmentation mask, thus ignoring FP. 
        image_b_binary = np.array(val_outputss[0][1]*val_labelss[0][1])
        image_c_binary = np.array(val_outputss_2[0][1]*val_labelss[0][1])
        
        #images d and e are predicted segmentation masks. To find FP later
        image_d_binary = np.array(val_outputss[0][1])
        image_e_binary = np.array(val_outputss_2[0][1])

        image_a_binary = np.uint8(image_a_binary)
        image_b_binary = np.uint8(image_b_binary)
        image_c_binary = np.uint8(image_c_binary)
        image_d_binary = np.uint8(image_d_binary)
        image_e_binary = np.uint8(image_e_binary)
        _, labels_a = cv2.connectedComponents(image_a_binary)
        _, labels_b = cv2.connectedComponents(image_b_binary)
        _, labels_c = cv2.connectedComponents(image_c_binary)
        _, labels_d = cv2.connectedComponents(image_d_binary)
        _, labels_e = cv2.connectedComponents(image_e_binary)
        unique_labels_a = np.unique(labels_a)
        unique_labels_b = np.unique(labels_d)
        unique_labels_c = np.unique(labels_e)

        #ignoring the backgroung label. Using unique labels_a to count how many true metastases there are.
        num_mets += len(unique_labels_a) - 1
        # First: finding the number of FPs
        # Identify clusters in B that do not overlap with any cluster in A
        clusters_non_overlap_b = []
        for labell in unique_labels_b:
            if labell != 0 and np.sum(np.logical_and(labels_b == labell, labels_a > 0)) == 0:
                clusters_non_overlap_b.append(labell)
            
        fp_list_1 += len(clusters_non_overlap_b)

        clusters_non_overlap_c = []
        for labell in unique_labels_c:
            if labell != 0 and np.sum(np.logical_and(labels_c == labell, labels_a > 0)) == 0:
                clusters_non_overlap_c.append(labell)
           
        fp_list_2 += len(clusters_non_overlap_c)
        ##########################################################

        # Identify clusters in A that do not overlap with any cluster in B
        clusters_non_overlap_a = []
        for labell in unique_labels_a:
            if labell != 0 and np.sum(np.logical_and(labels_a == labell, labels_b > 0)) == 0:
                clusters_non_overlap_a.append(labell)
        # Identify clusters in A that do not overlap with any cluster in C
        clusters_non_overlap_a_2 = []
        for labell in unique_labels_a:
            if labell != 0 and np.sum(np.logical_and(labels_a == labell, labels_c > 0)) == 0:
                clusters_non_overlap_a_2.append(labell)
                    

        new_image = np.zeros_like(image_a_binary)
        new_image_2 = np.zeros_like(image_a_binary)

            
        #Add clusters from A that do not overlap with any cluster in B to the new image. 
        #This is the undetected metastases
        for labell in clusters_non_overlap_a:
            new_image[labels_a == labell] = 1

        #Same for image C
        for labell in clusters_non_overlap_a_2:
            new_image_2[labels_a == labell] = 1

        labeled_image_1,count_1= measure.label(new_image, return_num=True)
        objects_1 = regionprops(labeled_image_1)
        object_areas_1 = [obj["area"] for obj in objects_1]

        for elem in object_areas_1:
            model_1_undetected_total.append(elem)

        labeled_image_2,count_2 = measure.label(new_image_2, return_num=True)
        objects_2 = regionprops(labeled_image_2)
        object_areas_2 = [obj["area"] for obj in objects_2]

        for elem in object_areas_2:
            model_2_undetected_total.append(elem)
                
        #if all mets are detected by model 2, but not by model 1
        if np.sum(new_image) != 0 and np.sum(new_image_2) == 0:
            for elem in object_areas_1:
                model_1_undetected.append(elem)
        #if all mets are detected by model 1, but not by model 2
        if np.sum(new_image_2) != 0 and np.sum(new_image) == 0:
            for elem in object_areas_2:
                model_2_undetected.append(elem)
        #if both models have undetected mets
        if np.sum(new_image_2) != 0 and np.sum(new_image) != 0:

            labeled_image_multi,_= measure.label(new_image_2 * new_image, return_num=True) #
            objects_multi = regionprops(labeled_image_multi)
            object_areas_multi = [obj["area"] for obj in objects_multi]

            for elem in object_areas_multi:
                undetected_by_both.append(elem)
            #if there are more undetected mets by model 2 than model 1, model 2 have undetected mets that model 1 has detected
            if count_1 < count_2: 
                labeled_image_F,_ = measure.label(new_image_2 - new_image, return_num=True) 
                objects_F = regionprops(labeled_image_F)
                object_areas_F = [obj["area"] for obj in objects_F]

                for elem in object_areas_F:
                    model_2_undetected.append(elem)
            #if there are more undetected mets by model 1 than model 2, model 1 has undetected mets that model 2 has detected
            if count_1 > count_2: 
                labeled_image_G,_ = measure.label(new_image - new_image_2, return_num=True) #
                objects_G = regionprops(labeled_image_G)
                object_areas_G = [obj["area"] for obj in objects_G]

                for elem in object_areas_G:
                    model_1_undetected.append(elem)

       
       
    return model_1_undetected_total, model_2_undetected_total, model_1_undetected, model_2_undetected, undetected_by_both, num_mets, dice_per_lesion_1, dice_per_lesion_2, fp_list_1, fp_list_2

datasets = [only_small_mets, small_mets, only_medium_mets, medium_mets, val_dataset]

for dataset in datasets:
    model_1_undetected_total, model_2_undetected_total, model_1_undetected, model_2_undetected, undetected_by_both, num_mets, dice_per_lesion_1, dice_per_lesion_2, fp_list_1, fp_list_2 = sensitivity(model_1, model_2, dataset)
    
