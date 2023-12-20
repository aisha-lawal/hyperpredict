import sys
sys.path.append("../")
import pandas as pd
import torch
import numpy as np
from utils import SetParams, load, imgnorm, compute_metric, generate_grid_unit

#running 1 and 2 hidden layers
hyperpredict_df= pd.read_csv("../results/symnet_clapirn/sensitivity_analysis/sensitivity_analysis_image_0.5.csv")

#Run registration with optimal lambda
registration_model = "clapirn"
start_channel = 4
range_flow = 0.4
imgshape = (160, 192, 224)
imgshape_4 = (160 / 4, 192 / 4, 224 / 4)
imgshape_2 = (160 / 2, 192 / 2, 224 / 2)
pretrained_path = "../models/pretrained_models/"
registration_path = "clapirn.pth"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

registration = SetParams.set_registration(registration_model, start_channel, imgshape, imgshape_2, imgshape_4, range_flow, pretrained_path +registration_path) 

grid = generate_grid_unit(imgshape)
grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).to(device).float()

lr_accumbens = (16, 32)
lr_amygdala = (15, 31)
lr_caudate = (8, 27)
lr_hippocampus = (14, 30)
lr_pallidum = (10, 29)
lr_putamen = (9, 28)
lr_thalamus = (7, 26)

hyperpredict_df["target_dice"] = 0
hyperpredict_df["target_jac"] = 0

print(len(hyperpredict_df))
for i in range(len(hyperpredict_df)):
    moving_index = hyperpredict_df.loc[i, 'moving_index']
    fixed_index = hyperpredict_df.loc[i, 'fixed_index']
    lamda = hyperpredict_df.loc[i, 'lamda']
   
    moving_path =  "../data/oasis/testing/OASIS_OAS1_0"+str(moving_index)+"_MR1/aligned_norm.nii.gz"
    fixed_path =  "../data/oasis/testing/OASIS_OAS1_0"+str(fixed_index)+"_MR1/aligned_norm.nii.gz"
    moving_lbl_path = "../data/oasis/testing/OASIS_OAS1_0"+str(moving_index)+"_MR1/aligned_seg35.nii.gz"
    fixed_lbl_path = "../data/oasis/testing/OASIS_OAS1_0"+str(fixed_index)+"_MR1/aligned_seg35.nii.gz"


    fixed_img = load(fixed_path)
    moving_img  = load(moving_path)
    fixed_label = load(fixed_lbl_path)
    moving_label = load(moving_lbl_path)

    fixed_img, moving_img = imgnorm(fixed_img), imgnorm(moving_img)
    fixed_img = torch.from_numpy(fixed_img).float().to(device).unsqueeze(dim=0)
    moving_img = torch.from_numpy(moving_img).float().to(device).unsqueeze(dim=0)

    fixed_label = torch.from_numpy(fixed_label).float().to(device).unsqueeze(dim=0)
    moving_label = torch.from_numpy(moving_label).float().to(device).unsqueeze(dim=0)
    lamda = torch.tensor([lamda], dtype=fixed_img.dtype, device=fixed_img.device).unsqueeze(dim=0)

    deformation_field = registration(moving_img, fixed_img, lamda)
    target_dice, target_jac = compute_metric(moving_label, fixed_label, deformation_field, grid)
 

    hyperpredict_df.loc[i, "target_dice"] = target_dice.mean().item()
    hyperpredict_df.loc[i, "target_jac"] = target_jac.item()
    print(i)
hyperpredict_df.to_csv("../results/symnet_clapirn/sensitivity_analysis/sensitivity_analysis_image_0.5_target.csv")

