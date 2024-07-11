import sys
sys.path.append("../")
import pandas as pd
import torch
import numpy as np
from utils import SetParams, load, imgnorm, compute_metric, generate_grid_unit



# get optimal based on highest dice - label
hyperpredict_df = pd.read_csv("../results/symnet_clapirn/final_result/final_label_1.0_alpha.csv")
hyperpredict_df = hyperpredict_df.groupby(["pair_idx", "label"], as_index=False).apply(lambda x: x[x.predicted_dice == x.predicted_dice.max()]).reset_index(drop=True)
print("len of optimal", len(hyperpredict_df))

# #for comparison - image
# hyperpredict_df = pd.read_csv("../results/symnet_clapirn/final_result/xavier_0.25_seeded_labels_quick.csv")
# # hyperpredict_df = hyperpredict_df.groupby(["pair_idx"], as_index=False).apply(lambda x: x[x.predicted_dice == x.predicted_dice.max()]).reset_index(drop=True)
# print("len of optimal", len(hyperpredict_df))

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
hyperpredict_df["target_label"] = "No" #for label

labels = {"Thalamus": lr_thalamus, "Hippocampus": lr_hippocampus, "Amygdala": lr_amygdala, "Accumbens": lr_accumbens, "Putamen": lr_putamen, "Pallidum": lr_pallidum, "Caudate": lr_caudate}

print(len(hyperpredict_df))
for i in range(len(hyperpredict_df)):
    moving_index = hyperpredict_df.loc[i, 'moving_index']
    fixed_index = hyperpredict_df.loc[i, 'fixed_index']
    lamda = hyperpredict_df.loc[i, 'lamda']
    label = hyperpredict_df.loc[i, 'label'] #for label
    
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
    
    #image start
    # hyperpredict_df.loc[i, "target_dice"] = target_dice.mean().item()
    # hyperpredict_df.loc[i, "target_jac"] = target_jac.item()
    # df = pd.DataFrame(columns=["pair_idx", "moving_index", "fixed_index",  "predicted_dice", "lamda", "predicted_jac", "target_dice", "target_jac"])
    # df.loc[i] = hyperpredict_df.loc[i]
    # df.to_csv("../results/symnet_clapirn/final_result/sensitivity_analysis/xavier_2.0_seeded_target.csv", mode='a', header=False, index=False)
    # print(i)
    #image stop

    #label start
    target_dice = target_dice[0]
    for l in labels.keys():
        if l == label:
            target_label = l
            target_avg = (target_dice[labels[l][0]-1] + target_dice[labels[l][1]-1])/2


    hyperpredict_df.loc[i, "target_dice"] = target_avg.item()
    hyperpredict_df.loc[i, "target_jac"] = target_jac.item()
    hyperpredict_df.loc[i, "target_label"] = target_label
    df = pd.DataFrame(columns=["pair_idx", "moving_index", "fixed_index",  "predicted_dice", "lamda", "label", "predicted_jac", "target_dice", "target_jac", "target_label"])
    df.loc[i] = hyperpredict_df.loc[i]
    df.to_csv("../results/symnet_clapirn/final_result/final_label_1.0_alpha_target.csv", mode='a', header=False, index=False)
    print(i)
    #label stop

 
