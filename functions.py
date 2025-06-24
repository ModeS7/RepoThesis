import os
from monai.data import Dataset
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import ConcatDataset
from torch.amp import autocast

class NiFTIDataset(Dataset):
    def __init__(self, data_dir, mr_sequence, transform=None):
        self.data_dir = data_dir
        self.data = os.listdir(data_dir)
        self.data.sort()
        self.mr_sequence = mr_sequence
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        nifti_file = os.path.join(self.data_dir, self.data[index] +
                                  "/" + self.mr_sequence + ".nii.gz")
        if self.transform is not None:
            nifti_file = self.transform(nifti_file)
        return nifti_file, self.data[index]


def merge_data(dataset1, dataset2):
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
            #print(i)
            image_stack = nifti_dataset.__getitem__(index = i)

            images = [image_stack[:,:,:,k] for k in range(
                image_stack.shape[3])] #each image has shape (2,128,128),
            # first element is bravo image, second is seg
            non_empty_images = []
            for image in images:
                if np.sum(image) > 1.0:
                    image[1] = make_binary(image[1], threshold = 0.01)
                    non_empty_images.append(image)
            total_dataset = ConcatDataset([total_dataset, non_empty_images])
    return total_dataset


def extract_slices_single(nifti_dataset):
    total_dataset = Dataset([])
    for i in range(len(nifti_dataset)):
        image_stack, _ = nifti_dataset.__getitem__(index=i)
        images = [image_stack[:, :, :, k] for k in range(image_stack.shape[3])]
        non_empty_images = []
        for image in images:
            if np.sum(image) > 1.0:  # Threshold to remove empty slices
                non_empty_images.append(image)
        total_dataset = ConcatDataset([total_dataset, non_empty_images])
    return total_dataset