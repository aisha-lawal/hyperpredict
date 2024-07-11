import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import nibabel as nib

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


plt.rcParams.update({'font.size': 16})
plt.rcParams["figure.figsize"] = (12,8)

maximum_nfv = 160 * 192 * 224
#get optimal based on highest dice
# hyperpredict_optimal_registration = pd.read_csv("../results/symnet_niftyreg/final_result/final_label_0.025_alpha_le.csv")
# hyperpredict_optimal_registration = hyperpredict_optimal_registration.groupby(["pair_idx", "label"], as_index=False).apply(lambda x: x[x.predicted_dice == x.predicted_dice.max()]).reset_index(drop=True)
# print("len of optimal", len(hyperpredict_optimal_registration))


hyperpredict_optimal_registration = pd.read_csv("../results/symnet_niftyreg/final_result/sensitivity_analysis/xavier_0.01_seeded.csv")
# hyperpredict_optimal_registration = hyperpredict_optimal_registration.groupby(["pair_idx"], as_index=False).apply(lambda x: x[x.predicted_dice == x.predicted_dice.max()]).reset_index(drop=True)
print("len of optimal", len(hyperpredict_optimal_registration))

def dice(im1, im2):
    im1 = im1.squeeze(0)
    im2 = im2.squeeze(0)
    unique_class = torch.unique(im2)
    dice = 0
    num_count = 0
    labels_dice_score = []
    for i in unique_class:
        if (i == 0) or ((im1 == i).sum() == 0) or ((im2 == i).sum() == 0):
            continue

        sub_dice = torch.sum(im2[im1 == i] == i) * 2.0 / (torch.sum(im1 == i) + torch.sum(im2 == i))
        dice += sub_dice
        labels_dice_score.append(sub_dice)
        num_count += 1

    return labels_dice_score

def load(img):
    img = nib.load(img).get_fdata()
    img = np.reshape(img, (1,) + img.shape)

    return img
def load_nii(fixed_image, moving_image,warped_image, fxd_lbl, mov_lbl, warped_label, jac_det, deformation_field, bspline):
    fixed_image = load(fixed_image)
    moving_image = load(moving_image)
    warped_image = load(warped_image)
    fixed_label = load(fxd_lbl)
    moving_label = load(mov_lbl)
    warped_label = load(warped_label)
    jac_det = load(jac_det)
    deformation_field = load(deformation_field)
    bspline = load(bspline)
    return torch.tensor(fixed_image, device = device, dtype=torch.float32), torch.tensor(moving_image, device = device, dtype=torch.float32), torch.tensor(warped_image, device = device, dtype=torch.float32), torch.tensor(fixed_label, device = device, dtype=torch.float32), torch.tensor(moving_label, device = device, dtype=torch.float32), torch.tensor(warped_label, device = device, dtype=torch.float32), torch.tensor(jac_det, device = device, dtype=torch.float32), torch.tensor(deformation_field, device =device, dtype=torch.float32)



def set_niftyreg():
        get_bspline = "reg_f3d -ref {} -flo {} -be {} -le {} -sx {} -cpp bspline.nii -res warped_image.nii"
        get_deformation_field = "reg_transform -ref {} -def bspline.nii deformation_field.nii"
        get_warp_seg = "reg_resample -ref {} -flo {} -inter 0 -trans deformation_field.nii -res warped_label.nii"
        get_jac_det = "reg_jacobian -ref {} -trans deformation_field.nii -jac jac_det_map.nii" 

        return get_bspline, get_deformation_field, get_warp_seg, get_jac_det

def pad_with_zeros(string):
    string = str(string)
    if len(string) < 4:
        num_zeros = 4 - len(string)
        padded_string = '0'* num_zeros + string
        return padded_string
    else:
        return string


def niftyreg(pair_idx, moving_index, fixed_index, fixed_label, bending_energy, linear_elasticity, spacing, task):
    moving_image = "../data/oasis/" +task +"/OASIS_OAS1_"+pad_with_zeros(moving_index)+"_MR1/aligned_norm.nii.gz"
    fixed_image = "../data/oasis/" +task +"/OASIS_OAS1_"+pad_with_zeros(fixed_index)+"_MR1/aligned_norm.nii.gz"
    mov_lbl = "../data/oasis/" +task +"/OASIS_OAS1_"+pad_with_zeros(moving_index)+"_MR1/aligned_seg35.nii.gz"
    fxd_lbl = "../data/oasis/" +task +"/OASIS_OAS1_"+pad_with_zeros(fixed_index)+"_MR1/aligned_seg35.nii.gz"
    
    warped_label = "warped_label.nii"
    jac_det = "jac_det_map.nii"
    deformation_field = "deformation_field.nii" 
    bspline = "bspline.nii" 
    warped_image = "warped_image.nii"
     
    get_bspline, get_deformation_field, get_warp_seg, get_jac_det = set_niftyreg()
    os.system(get_bspline.format(fixed_image, moving_image, bending_energy, linear_elasticity, spacing))
    os.system(get_deformation_field.format(fixed_image))
    os.system(get_warp_seg.format(fxd_lbl, mov_lbl))
    os.system(get_jac_det.format(fixed_image))
    
    # print("shape of everything from niftyreg before registration ", nib.load("warped_image.nii").get_fdata().shape, nib.load("warped_label.nii").get_fdata().shape, nib.load("jac_det_map.nii").get_fdata().shape, 
    #       nib.load("deformation_field.nii").get_fdata().shape, nib.load("bspline.nii").get_fdata().shape)

    fixed_image, moving_image, warped_image, fixed_label, moving_label, warped_label, jac_det, deformation_field = load_nii(fixed_image, moving_image,warped_image, fxd_lbl, mov_lbl, warped_label, jac_det, deformation_field, bspline)
    # print("shape of everything from niftyreg after registration ", fixed_image.shape, moving_image.shape, warped_image.shape, fixed_label.shape, moving_label.shape, warped_label.shape, jac_det.shape,
    #        deformation_field.shape, fixed_label.min(), fixed_label.max(), moving_label.min(), moving_label.max(), warped_label.min(), warped_label.max())
    folded_voxels = torch.count_nonzero(jac_det.squeeze(0)<0) 
    dice_score = dice(torch.floor(warped_label), torch.floor(fixed_label))

    return dice_score, folded_voxels.item()


lr_thalamus = (7, 26)
lr_hippocampus = (14, 30)
lr_cerebellum_WM = (5, 24)
lr_cerebellum_cortex = (6, 25)
lr_amygdala = (15, 31)
lr_cerebral_cortex = (2, 21)
lr_cerebral_WM = (1, 20)
lr_vessel = (18, 34)
lr_lateral_ventricle = (3, 22)
lr_accumbens = (16, 32)
lr_putamen = (9, 28)
lr_pallidum = (10, 29)
lr_caudate = (8, 27)

hyperpredict_optimal_registration["target_dice"] = 0
hyperpredict_optimal_registration["target_jac"] = 0
# hyperpredict_optimal_registration["target_label"] = "No" #for label

# labels = {"Thalamus": lr_thalamus, "Hippocampus": lr_hippocampus, "Amygdala": lr_amygdala, "Accumbens": lr_accumbens, "Putamen": lr_putamen, "Pallidum": lr_pallidum, "Caudate": lr_caudate, "Cerebellum WM": lr_cerebellum_WM,
#                    "Cerebellum Cortex": lr_cerebellum_cortex, "Cerebral Cortex": lr_cerebral_cortex, "Cerebral WM": lr_cerebral_WM, "Vessel": lr_vessel, "Lateral Ventricle": lr_lateral_ventricle}

labels = {"Thalamus": lr_thalamus, "Hippocampus": lr_hippocampus, "Amygdala": lr_amygdala, "Accumbens": lr_accumbens, "Putamen": lr_putamen, "Pallidum": lr_pallidum, "Caudate": lr_caudate}

for i in range(len(hyperpredict_optimal_registration)):
    # if i > 2:
    #     break
    moving_index = hyperpredict_optimal_registration.loc[i, 'moving_index']
    fixed_index = hyperpredict_optimal_registration.loc[i, 'fixed_index']
    print("moving and fixed indexes", moving_index, fixed_index)

    be = hyperpredict_optimal_registration.loc[i, 'be']
    sx = hyperpredict_optimal_registration.loc[i, 'sx']
    le = hyperpredict_optimal_registration.loc[i, 'le']
    # label = hyperpredict_optimal_registration.loc[i, 'label'] #for label
    fixed_label = load("../data/oasis/testing/OASIS_OAS1_"+pad_with_zeros(fixed_index)+"_MR1/aligned_seg35.nii.gz")
    target_dice, target_jac = niftyreg(i, moving_index, fixed_index, fixed_label, be, le, sx, "testing")

    #image start
    target_dice =  torch.stack(target_dice).mean().item() 
    hyperpredict_optimal_registration.loc[i, "target_jac"] = target_jac
    hyperpredict_optimal_registration.loc[i, "target_dice"] = target_dice
    df = pd.DataFrame(columns=["pair_idx", "moving_index", "fixed_index",  "predicted_dice", "be", "le", "sx", "predicted_jac", "target_dice", "target_jac"])
    df.loc[i] = hyperpredict_optimal_registration.loc[i]
    break
    df.to_csv("../results/symnet_niftyreg/final_result/sensitivity_analysis/xavier_0.01_seeded_target.csv", mode='a', header=False, index=False)
    print(i)
    #image stop

    # label start
    # for l in labels.keys():
    #     if l == label:
    #         target_label = l
    #         target_avg = (target_dice[labels[l][0]-1] + target_dice[labels[l][1]-1])/2


    # hyperpredict_optimal_registration.loc[i, "target_dice"] = target_avg.item()
    # hyperpredict_optimal_registration.loc[i, "target_jac"] = target_jac
    # hyperpredict_optimal_registration.loc[i, "target_label"] = target_label
    # df = pd.DataFrame(columns=["pair_idx", "moving_index", "fixed_index",  "predicted_dice", "be", "le", "sx", "label", "predicted_jac", "target_dice", "target_jac", "target_label"])
    # df.loc[i] = hyperpredict_optimal_registration.loc[i]
    # print(i)
    # df.to_csv("../results/symnet_niftyreg/final_result/final_label_0.025_alpha_le_target.csv", mode='a', header=False, index=False)
    #label stop




