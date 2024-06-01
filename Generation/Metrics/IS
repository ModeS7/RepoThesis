'''Code for finding the inception score. This code was written following examples from the pytorch-ignite framework
https://pytorch.org/ignite/generated/ignite.metrics.InceptionScore.html'''

from collections import OrderedDict
import torch
from torch import nn
from ignite.engine import *
from ignite.utils import *
import os
from monai.data import Dataset
import numpy as np
import nibabel as nib
def eval_step(engine, batch):
    return batch

default_evaluator = Engine(eval_step)

param_tensor = torch.zeros([1], requires_grad=True)
default_optimizer = torch.optim.SGD([param_tensor], lr=0.1)

def get_default_trainer():

    def train_step(engine, batch):
        return batch

    return Engine(train_step)

default_model = nn.Sequential(OrderedDict([
    ('base', nn.Linear(4, 2)),
    ('fc', nn.Linear(2, 1))
]))

manual_seed(666)

syn_dir = "C:\\Users\\amibe\\Documents\\Masteroppgave\\Inception_score\\train\\train"
syn_list = os.listdir(syn_dir)
syn_data = []
for i in range(len(syn_list)):
    image = nib.load(syn_dir + "/" + syn_list[i]).get_fdata()
    #print(image.shape)
    syn_data.append(image)

def preprocess_for_fid(images):
    images = np.array(images).astype('float32')
    images = torch.Tensor(images)
    #images = torch.unsqueeze(torch.tensor(images), 1)
    print(images.shape)
    images_3 = torch.concatenate((images, images, images), dim = 1)
    print(images_3.shape)
    #if images.shape[1]:
     #   images = images.repeat(1, 3, 1, 1)# --> (1, 3, 128,128)
    return images_3


images = preprocess_for_fid(syn_data)#(1, 128, 128)
#print(np.array(images).shape, ", # of images: ",len(images))


from ignite.metrics import InceptionScore
import torch

metric = InceptionScore()
metric.attach(default_evaluator, "is")
state = default_evaluator.run([images])
print(state.metrics["is"])

