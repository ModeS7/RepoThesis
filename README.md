# Thesis
README file for the master's thesis of Anna Milde Bekkevoll. My master's thesis is written as a part of my specialization in biophysics and medical technology at NTNU in Trondheim. This repository is divided into the folders "Generation" and "Segmentation". 

The "Generation" folder includes the code used for training the diffusion models, and for sampling synthetic data from the trained models. Code for the metrics used to evaluate the generated
images is also found in this folder. This includes code for finding FID, IS and MS-SSIM / SSIM values. Different tutorials from the MONAI consortsium have been used when writing the code, and the link to these tutorials are provided
in the files. 

The "Segmentation" folder includes code for the training of the segmentation model used for the different segmentation experiments. Also performance.py, dice_per_lesion.py and hausdorff_distance.py files are included that were used to test the model performance after training.
