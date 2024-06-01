'''Code for training the segmentation model'''

from monai.data import Dataset, DataLoader, decollate_batch, CacheDataset
from torch.utils.data import ConcatDataset
from monai.transforms import Compose, LoadImage, ToTensor, ScaleIntensity, EnsureChannelFirst, Resize, CenterSpatialCrop, Activations,AsDiscrete, RandScaleIntensity, RandShiftIntensity, RandFlip, CropForeground, RandRotate90, RandSpatialCrop, RandRotate, RandAdjustContrast, RandHistogramShift, RandSpatialCrop, SpatialPad
from monai.networks.nets import UNet
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss
from monai.networks.layers import Norm
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from skimage.measure import label,regionprops
from skimage import measure
from torch.utils.tensorboard import SummaryWriter
import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import nibabel as nib
import cv2
data_dir_train = "/cluster/home/annambek/Masteroppgave/Data/BrainMets/StanfordSkullStripped/train"
data_dir_test = "/cluster/home/annambek/Masteroppgave/Data/BrainMets/StanfordSkullStrippedTest"

#OBS TEST, må være extract slices som bare tar kreft-slicer
from Evaluation.dice_per_lesion import dice_per_lesion
from helper_functions import merge_data, make_binary, extract_slices 



model_name = "5000__syn__cuttoff25_max3__binary_5em4_dropout0.2"

class NiFTIDataset(Dataset):
    def __init__(self, data_dir, mr_sequence, transform = None):
        self.data_dir = data_dir
        self.data = os.listdir(data_dir) 
       # self.data = np.array(self.data) #+ "/" + mr_sequence + "nii.gz"
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

#Transform to be applied to bravo image before merge with label
transform_train_image = Compose( 
    [
        LoadImage(image_only = True), 
        EnsureChannelFirst(), 
        ToTensor(),
        ScaleIntensity(minv = 0.0, maxv = 1.0), #Scale intensity to range [0,1]
        Resize(spatial_size = (256, 256, -1)), #Resize the images to 256x256
        RandScaleIntensity(factors=0.5, prob=0.1), #Randomly scale the intensity with 0.1 probability
        RandShiftIntensity(offsets=0.1, prob=0.1), #Randomly shift the intensity with 0.1 probability
        RandAdjustContrast(gamma=(0.5, 2), prob=0.1), #Randomly adjust the contrast with 0.1 probability
        RandHistogramShift( num_control_points=(5, 15), prob=0.1), #Randomly shift the histogram with the probability 0.1
    ]
)

#Transform to be applied to the annotation masks + for bravo validation aswell 
transform_train_mask = Compose(
    [
        LoadImage(image_only = True), 
        EnsureChannelFirst(), 
        ToTensor(),
        ScaleIntensity(minv = 0.0, maxv = 1.0),
        Resize(spatial_size = (256, 256, -1)), 
    ]
)

#Transform to be applied after the bravo-seg merge
transform_after_merge_train = Compose([ 
    RandFlip(prob=0.25, spatial_axis=0),
    RandFlip(prob=0.25, spatial_axis=1),
    RandRotate90(prob=0.5, spatial_axes=(0, 1)),
    RandRotate(range_x=0.4, padding_mode='zeros'),
])

bravo_dataset = NiFTIDataset(data_dir= data_dir_train,mr_sequence="bravo", transform = transform_train_image)
labels_dataset = NiFTIDataset(data_dir= data_dir_train,mr_sequence="seg", transform = transform_train_mask)

bravo_dataset_val = NiFTIDataset(data_dir= data_dir_test,mr_sequence="bravo", transform = transform_train_mask)
labels_dataset_val = NiFTIDataset(data_dir= data_dir_test,mr_sequence="seg", transform = transform_train_mask)

merged = merge_data(bravo_dataset, labels_dataset)
merged_val = merge_data(bravo_dataset_val, labels_dataset_val)

train_dataset = extract_slices(merged, 1.0, "Training")
val_dataset = extract_slices(merged_val, 1.0, "Validation") 

####Load synthetic data#####

syn_data_dir = "/cluster/home/annambek/Masteroppgave/Synthetic_images/BrainMets/mask_conditioned_synthesis"
syn_data_list = os.listdir(syn_data_dir)

syn_data = []
for i in range(5000):
    image = nib.load(syn_data_dir + "/" + syn_data_list[i]).get_fdata()
    image_MR = cv2.resize(image[:,:,0], dsize=(256,256),interpolation=cv2.INTER_CUBIC)
    image_mask = cv2.resize(image[:,:,1], dsize=(256,256),interpolation=cv2.INTER_CUBIC)
    image_MR = np.expand_dims(image_MR, axis = 0)
    image_mask = np.expand_dims(image_mask, axis = 0)
    image = np.concatenate((image_MR,image_mask), axis = 0)
    image = image.astype("float32")
    syn_data.append(image)

synthetic_dataset = CacheDataset(syn_data, transform=transform_after_merge_train)

bs = 16
train_ds = CacheDataset(data = train_dataset, transform = transform_after_merge_train)
train_ds_concat = ConcatDataset([train_ds, synthetic_dataset])

train_loader = DataLoader(train_ds_concat, batch_size = bs, shuffle = True) 
val_loader = DataLoader(val_dataset, batch_size = bs, shuffle = True)

device = torch.device("cuda")

model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
    dropout=0.2,
).to(device)

#post-processing of true segmentation mask / predicted label. Make into one-hot format
post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
post_label = Compose([AsDiscrete(to_onehot=2)])

#Loss function a weighting between dice loss and cross entropy. 
loss_function = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)

#Adam optimizier, lr5e-4 / lr1e-4
optimizer = torch.optim.Adam(model.parameters(), 5e-4)

dice_metric = DiceMetric(include_background=False, reduction="mean")
scaler = torch.cuda.amp.GradScaler()

####Training#######################################################
num_epochs = 1500
val_interval = 5
best_metric = -1 
best_recall = -1
metric_values = [] 
best_dice_per_lesion = -1
writer = SummaryWriter(comment = model_name) #writing to tensorboard

for epoch in range(num_epochs):
    epoch_loss = 0
    model.train()
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in progress_bar: 
        input_images = batch[:,0,:,:].to(device)
        input_images = torch.unsqueeze(input_images, dim =1)
        #print("Shape of input images:", input_images.shape)
        labels = batch[:,1,:,:].to(device)
        labels = torch.unsqueeze(labels, dim = 1)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs_seg = model(input_images) 
            loss = loss_function(outputs_seg, labels) 
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
        progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
    #Write epoch loss to tensorboard
    writer.add_scalar("Loss/train", epoch_loss / (step + 1), epoch)

    if (epoch + 1) % val_interval == 0:
        model.eval()
        precision_list = []
        recall_list = []
        fn_list = []
        dice_per_lesion_list = []
        with torch.no_grad():
            progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), ncols=70)
            progress_bar.set_description(f"Epoch {epoch}")
            for step, batch in progress_bar: 
                val_input_images = batch[:,0,:,:].to(device) #shape now: [B,H,W]. 
                val_input_images = torch.unsqueeze(val_input_images, dim = 1) # shape --> [B,C,H,W]
                val_labels = batch[:,1,:,:].to(device)
                val_labels = torch.unsqueeze(val_labels, dim = 1)

                #get model predicitions
                val_output = model(val_input_images)

                #post-processing
                val_output_1 = [post_pred(i) for i in decollate_batch(val_output.cpu())] 
                val_labels_1 = [post_label(i) for i in decollate_batch(val_labels.cpu())]
                
                dice_metric(y_pred=val_output_1, y=val_labels_1)
                #for each predicted segmentation mask in val_output_1 (length 16)
                for j in range(len(val_output_1)):
                    #find precision between flattened tensors
                    precision = precision_score(torch.flatten(val_labels_1[j][1]), torch.flatten(val_output_1[j][1]))
                    precision_list.append(precision)
                  
                    recall = recall_score(torch.flatten(val_labels_1[j][1]), torch.flatten(val_output_1[j][1]))
                    recall_list.append(recall)
                  
                    tn, fp, fn, tp = confusion_matrix(torch.flatten(val_labels_1[j][1]), torch.flatten(val_output_1[j][1])).ravel()
                    fn_list.append(fn / (fn + tp)) 

                    # find dice per lesion
                    dice_liste = dice_per_lesion( y= val_labels_1[j][1,:,:], y_pred = val_output_1[j][1,:,:])
                    for elem in dice_liste:
                        dice_per_lesion_list.append(elem)

            metric = dice_metric.aggregate().item()

            #write metrics to tensorboard
            writer.add_scalar("DiceMetric/Validation", metric, epoch)
            writer.add_scalar("Precision/Validation", np.mean(precision_list), epoch)
            writer.add_scalar("Recall/Validation", np.mean(recall_list), epoch)
            writer.add_scalar("fn/Validation", np.mean(fn_list), epoch)
            writer.add_scalar("Dice per lesion/Validation", np.mean(dice_per_lesion_list), epoch)
            dice_metric.reset()
            metric_values.append(metric)

            recall = np.mean(recall_list)
            dice_per_lesion_value = np.mean(dice_per_lesion_list)

            #Save model
            if metric > best_metric:
                best_metric = metric
                path = "/cluster/home/annambek/Masteroppgave/Trained_models/BrainMets/UNet/best_dice_model" + model_name
                torch.save(model.state_dict(), path)

            if recall > best_recall:
                best_recall = recall
                path = "/cluster/home/annambek/Masteroppgave/Trained_models/BrainMets/UNet/best_recall_model" + model_name
                torch.save(model.state_dict(), path)

            if dice_per_lesion_value  > best_dice_per_lesion:
                best_dice_per_lesion = dice_per_lesion_value
                path = "/cluster/home/annambek/Masteroppgave/Trained_models/BrainMets/UNet/best_dice_per_lesion_model" + model_name
                torch.save(model.state_dict(), path)



    

    




    
