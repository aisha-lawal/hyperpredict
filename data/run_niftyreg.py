import os
import nibabel as nib
import numpy as np
import torch
import random
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.utils import data as Data
import glob

from torch.utils import data as Data
from itertools import permutations

def imgnorm(img):
    max_val = np.max(img)
    min_val = np.min(img)
    norm_img = (img - min_val) / (max_val - min_val)
    return norm_img

class DatasetImgsLabels(Data.Dataset):
    def __init__(self, imgs, labels, encoder_model = "clapirn", data = "oasis"):
        super(DatasetImgsLabels, self).__init__()
        self.imgs = imgs 
        self.labels = labels 
        self.img_pair = list(permutations(self.imgs , 2))
        self.label_pair = list(permutations(self.labels, 2))
        self.data = data
        self.encoder_model = encoder_model

    def __getitem__(self, index):

        moving_img = load(self.img_pair[index][0])
        fixed_img = load(self.img_pair[index][1])

        moving_label= load(self.label_pair[index][0])
        fixed_label= load(self.label_pair[index][1])

        if self.data == "oasis":
            moving_index, fixed_index = self.img_pair[index][0].split("_")[2], self.img_pair[index][1].split("_")[2]
            return torch.from_numpy(imgnorm(moving_img)).float(), torch.from_numpy(imgnorm(fixed_img)).float(), torch.from_numpy(moving_label).float(), torch.from_numpy(fixed_label).float(), moving_index, fixed_index

        elif self.data == "abdominal_ct":
            moving_index, fixed_index = self.img_pair[index][0].split("_")[1], self.img_pair[index][1].split("_")[1]

            return torch.from_numpy(imgnorm(moving_img)).float(), torch.from_numpy(imgnorm(fixed_img)).float(), torch.from_numpy(moving_label).float(), torch.from_numpy(fixed_label).float(), moving_index, fixed_index

      

    def __len__(self):
        return len(self.img_pair)
 

def save_csv(idx, moving_index, fixed_index, dice_per_label, dice_mean, Jdet_mean, Jdet_std, count_folded_voxels, be, le, sx, columns, file_path):
    dice_per_label = dice_per_label.tolist()
    row =[idx, moving_index,fixed_index ,dice_per_label, dice_mean, Jdet_mean, Jdet_std, count_folded_voxels, be, le, sx]
    
    data = dict(zip(columns, row))
    df = pd.DataFrame(data = data)
    df.to_csv(file_path,  mode='a', index=False, header= False)

def dice3D(warped_lbl, fixed_lbl):
    batch = torch.ones(1, 35, device=device)
    for i in range(warped_lbl.shape[0]):
        dice = 0
        dice_per_label = []
        num_count = 0
        labels = 35
        fixed_label = fixed_lbl
        warped_label = warped_lbl
        fixed_label_unique = torch.unique(fixed_label)
        warped_label_unique = torch.unique(warped_label)
        for i in range(1, labels +1):
            if i in fixed_label_unique and i in warped_label_unique:
                sub_dice = torch.sum(fixed_label[warped_label == i] == i) * 2.0 / (torch.sum(warped_label == i) + torch.sum(fixed_label == i))
                dice += sub_dice
                num_count += 1
                dice_per_label.append(sub_dice.item())

            elif( i not in fixed_label_unique ) or (i not in warped_label_unique):
                dice_per_label.append(0.0)
        
        dice_per_label = torch.tensor(dice_per_label, device = device).unsqueeze(0)
        batch = torch.cat([batch, dice_per_label], dim=0)
    
    batch = batch[1:]

    return batch

def load(img):
    img = nib.load(img).get_fdata()
    img = np.reshape(img, (1,) + img.shape)

    return img
def load_nii(fxd_lbl, mov_lbl, warped_label, jac_det, deformation_field, bspline):

    fixed_label = load(fxd_lbl)
    moving_label = load(mov_lbl)
    warped_label = load(warped_label)
    jac_det = load(jac_det)
    deformation_field = load(deformation_field)
    bspline = load(bspline)
    return torch.tensor(fixed_label, device = device, dtype=torch.float32), torch.tensor(moving_label, device = device, dtype=torch.float32), torch.tensor(warped_label, device = device, dtype=torch.float32), torch.tensor(jac_det, device = device, dtype=torch.float32), torch.tensor(deformation_field, device =device, dtype=torch.float32)

def set_niftyreg():
        get_bspline = "reg_f3d -ref {} -flo {} -be {} -sx {} -le {} -cpp bspline.nii -res warped_image.nii"
        get_deformation_field = "reg_transform -ref {} -def bspline.nii deformation_field.nii"
        get_warp_seg = "reg_resample -ref {} -flo {} -inter 0 -trans deformation_field.nii -res warped_label.nii"
        get_jac_det = "reg_jacobian -ref {} -trans deformation_field.nii -jac jac_det_map.nii"

        return get_bspline, get_deformation_field, get_warp_seg, get_jac_det



def niftyreg(moving_index, fixed_index, fixed_label, be, le, sx, task):

    moving_image = "data/oasis/" +task +"/OASIS_OAS1_"+moving_index[0]+"_MR1/aligned_norm.nii.gz"
    fixed_image = "data/oasis/" +task +"/OASIS_OAS1_"+fixed_index[0]+"_MR1/aligned_norm.nii.gz"
    mov_lbl = "data/oasis/" +task +"/OASIS_OAS1_"+moving_index[0]+"_MR1/aligned_seg35.nii.gz"
    fxd_lbl = "data/oasis/" +task +"/OASIS_OAS1_"+fixed_index[0]+"_MR1/aligned_seg35.nii.gz"
    
    warped_label = "warped_label.nii"
    jac_det = "jac_det_map.nii"
    deformation_field = "deformation_field.nii" 
    bspline = "bspline.nii" 
    warped_image = "warped_image.nii"

     
    get_bspline, get_deformation_field, get_warp_seg, get_jac_det = set_niftyreg()
    os.system(get_bspline.format(fixed_image, moving_image, be, sx, le))
    os.system(get_deformation_field.format(fixed_image))
    os.system(get_warp_seg.format(fxd_lbl, mov_lbl))
    os.system(get_jac_det.format(fixed_image))
    
    fixed_label, moving_label, warped_label, jac_det, deformation_field = load_nii(fxd_lbl, mov_lbl, warped_label, jac_det, deformation_field, bspline)
    dice_per_label = dice3D(torch.floor(warped_label), torch.floor(fixed_label)) 

    print("dice before and after registration", dice3D(torch.floor(moving_label), torch.floor(fixed_label)).mean(),
           dice3D(torch.floor(warped_label), torch.floor(fixed_label)).mean())
    folded_voxels = torch.count_nonzero(jac_det.squeeze(0)<0)

    return dice_per_label, folded_voxels, jac_det.mean(), jac_det.std()


def generating_validation_data():
    datapath = "data/oasis/validation"
    imgs = sorted(glob.glob(datapath + '/OASIS_OAS1_*_MR1/aligned_norm.nii.gz'))
    labels = sorted(glob.glob(datapath + '/OASIS_OAS1_*_MR1/aligned_seg35.nii.gz'))
    testing_generator = Data.DataLoader(
                        DatasetImgsLabels(imgs, labels),
                        batch_size=batch_size,
                        shuffle=True, drop_last = True, num_workers=2)

    print("length of data is ", len(testing_generator)) 
    columns = ['index','moving_index','fixed_index','dice_per_label', 'dice_mean', 'Jdet_mean', 'Jdet_std','count_folded_voxels','be', 'le', 'sx']

    file_path = "data/oasis/niftyreg_validation_data_be_le.csv"
    for batch_idx, data in enumerate(testing_generator):
        _, _, _, fixed_label, moving_index, fixed_index = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4], data[5]
        be = torch.tensor(random.sample(bending_energy.tolist(), batch_size), device = device).reshape(batch_size, 1)
        le = torch.tensor(random.sample(linear_elasticity.tolist(), batch_size), device = device).reshape(batch_size, 1)
        sx = torch.tensor(random.sample(spacing, batch_size), device=device).unsqueeze(0)
        
        print("batch_idx is ", batch_idx,  moving_index, fixed_index, be, le, sx)
        dice_per_label, num_folded_voxels, jdet_mean, jdet_std = niftyreg(moving_index, fixed_index, fixed_label, be.item(), le.item(), sx.item(), "validation")

        save_csv(batch_idx, moving_index, fixed_index, dice_per_label, dice_per_label.mean().item(), jdet_mean.item(), jdet_std.item(), 
                 num_folded_voxels.item(), be.item(), le.item(), sx.item(), columns, file_path)

if __name__ == "__main__":
    
    mu = 0.0
    sigma = 0.7
    distrubution = torch.distributions.log_normal.LogNormal(mu, sigma)
    #bending energy
    low = torch.tensor([random.uniform(0.0, 0.01) for _ in range(300)])
    mid = torch.tensor([random.uniform(0.0, 0.1) for _ in range(1000)])
    high = torch.tensor([random.uniform(0.2, 1.25) for _ in range(1000)])


    be = distrubution.sample((2000,))
    be = (be - be.min())/ ((be.max())/1.75 - be.min())
    bending_energy = torch.cat((be, low, mid, high), dim=0)

    #linear elasticity
    le = distrubution.sample((2000,))
    le = (le - le.min())/ ((le.max())/1.75 - le.min())
    linear_elasticity = torch.cat((le, low, mid, high), dim=0)

    batch_size = 1
    spacing = [5.0]
    start_t = datetime.now()

    generating_validation_data()
    end_t = datetime.now()
    total_t = end_t - start_t
    print("Time: ", total_t.total_seconds())
