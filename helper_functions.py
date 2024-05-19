import os
import numpy as np
from torch.utils.data import ConcatDataset
from monai.data import Dataset
from monai.transforms import Compose, LoadImage, ToTensor, ScaleIntensity, EnsureChannelFirst, Resize

'''NiFTIDataset class inherits from the monai Dataset class. If transform is not None, the specified transform
is applied to the specifiec mr_sequence. Important: mr_sequence must be either "bravo","t1_pre", "t1_gd", "flair" or "seg". '''
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
    

transform_train_mask = Compose(
    [
        LoadImage(image_only = True), 
        EnsureChannelFirst(), 
        ToTensor(),
        ScaleIntensity(minv = 0.0, maxv = 1.0),
        Resize(spatial_size = (256, 256, -1)), 
    ]
)

'''Function to merge bravo images with corresponding labels'''
def merge_data(dataset1, dataset2):
    dataset_tuple = []
    for i in range(len(dataset1)):
        image_stack_1, image_name_1 = dataset1.__getitem__(index = i)
        image_stack_2, image_name_2 = dataset2.__getitem__(index = i)
        
        if image_name_1 == image_name_2:
            combined = np.concatenate((image_stack_1,image_stack_2),axis = 0)
            dataset_tuple.append(combined)

    return Dataset(dataset_tuple)

'''Function to make labels (synthetic) binary. Threshold used is 0.01.'''
def make_binary(image, threshold):
    image_binary = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] > threshold:
                image_binary[i][j] = 1.0
    return image_binary

'''Function to extract slices from 3D volume. Only slices containing pathology is extracted. thres = threshold
to chech whether the slice contains pathology.'''
def extract_slices(nifti_dataset, thres):
    total_dataset = Dataset([])
    for i in range(len(nifti_dataset)):
            image_stack = nifti_dataset.__getitem__(index = i) 
             
            images_all = [image_stack[:,:,:,k] for k in range(image_stack.shape[3])]
            images_list = []

            for element in images_all:
                if np.sum(element[1,:,:]) > thres:
                    element[1,:,:] = make_binary(element[1,:,:], threshold=0.01)
                    images_list.append(element)
            images = Dataset(images_list)
            total_dataset = ConcatDataset([total_dataset, images])
    return total_dataset 

    
