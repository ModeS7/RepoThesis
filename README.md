# Thesis
README file for the master's thesis of Anna Milde Bekkevoll. My master's thesis is written as a part of my specialization in bipphysics and medical technology at NTNU in Trondheim. This repository is divided into Generation and Segmentation folders. 

The Generation folder includes the code used for training the diffusion models, and for sampling synthetic data from the trained models. Code for the metrics used to evaluate the generated
images is also found in this folder. This includes code for finding FID, IS and MS-SSIM / SSIM values. Different tutorials from the MONAI consortsium have been used when writing the code, and the link to these tutorials are provided
in the files. 

The Segmentation folder includes code for the training of the segmentation model used for the different segmentation experiments. Also a performance.py file is included that was used to test the model performance after training. 

Tutorials that have been used in these files: 
Diffusion model and training:
https://github.com/Project-MONAI/GenerativeModels/blob/main/tutorials/generative/2d_ddpm/2d_ddpm_tutorial.ipynb

FID, SSIM and MS-SSIM:
https://github.com/Project-MONAI/GenerativeModels/blob/main/tutorials/generative/realism_diversity_metrics/realism_diversity_metrics.ipynb

inception score:
https://pytorch.org/ignite/generated/ignite.metrics.InceptionScore.html
