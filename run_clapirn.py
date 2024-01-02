import os
from argparse import ArgumentParser

import numpy as np
import torch
from datetime import datetime
import random

from utils import DatasetImgsLabels, SpatialTransform_unit, SpatialTransformNearest_unit, dice3D, generate_grid_unit
from models.registrations.clapirn import Lvl1, Lvl2, Lvl3
import glob
from torch.utils import data as Data
import pandas as pd




def JacboianDet(y_pred, sample_grid):
    batch_count_num_voxels = torch.rand(1, 1, device=device)
    batch_count_mean = torch.rand(1, 1, device=device)
    batch_count_std = torch.rand(1, 1, device=device)
    for i in range(y_pred.shape[0]):

        J = y_pred[i].unsqueeze(0)+ sample_grid
        #derivatives using finite difference method
        dy = J[:, 1:, :-1, :-1, :] - J[:, :-1, :-1, :-1, :]
        dx = J[:, :-1, 1:, :-1, :] - J[:, :-1, :-1, :-1, :]
        dz = J[:, :-1, :-1, 1:, :] - J[:, :-1, :-1, :-1, :]

        Jdet0 = dx[:,:,:,:,0] * (dy[:,:,:,:,1] * dz[:,:,:,:,2] - dy[:,:,:,:,2] * dz[:,:,:,:,1])
        Jdet1 = dx[:,:,:,:,1] * (dy[:,:,:,:,0] * dz[:,:,:,:,2] - dy[:,:,:,:,2] * dz[:,:,:,:,0])
        Jdet2 = dx[:,:,:,:,2] * (dy[:,:,:,:,0] * dz[:,:,:,:,1] - dy[:,:,:,:,1] * dz[:,:,:,:,0])

        Jdet = Jdet0 - Jdet1 + Jdet2
        count_num_folded_voxels = torch.count_nonzero(Jdet < 0)
        batch_count_num_voxels = torch.cat([batch_count_num_voxels, count_num_folded_voxels.unsqueeze(0).unsqueeze(0)], dim=0)
        batch_count_mean = torch.cat([batch_count_mean, Jdet.mean().unsqueeze(0).unsqueeze(0)], dim=0)
        batch_count_std = torch.cat([batch_count_std, Jdet.std().unsqueeze(0).unsqueeze(0)], dim=0)

    batch_count_num_voxels = batch_count_num_voxels[1:]
    batch_count_mean = batch_count_mean[1:]
    batch_count_std = batch_count_std[1:]
    return batch_count_num_voxels, batch_count_mean, batch_count_std

def save_csv(idx, moving_index, fixed_index, dice_per_label, dice_mean, Jdet_mean, Jdet_std, count_folded_voxels, hyper_param, columns, file_path):
    dice_per_label = dice_per_label.tolist()
    row =[idx, moving_index,fixed_index ,dice_per_label, dice_mean, Jdet_mean, Jdet_std, count_folded_voxels, hyper_param]
    
    data = dict(zip(columns, row))
    df = pd.DataFrame(data = data)
    df.to_csv(file_path,  mode='a', index=False, header= False)


def generating_training_data():

    datapath = "data/oasis/training"
    imgs = sorted(glob.glob(datapath + '/OASIS_OAS1_*_MR1/aligned_norm.nii.gz'))
    labels = sorted(glob.glob(datapath + '/OASIS_OAS1_*_MR1/aligned_seg35.nii.gz'))

    testing_generator = Data.DataLoader(
                        DatasetImgsLabels(imgs, labels),
                        batch_size=batch_size,
                        shuffle=True, drop_last = True, num_workers=2)

    columns = ['index','moving_index','fixed_index','dice_per_label', 'dice_mean', 'Jdet_mean', 'Jdet_std','count_folded_voxels','hyper_param']

    file_path = "data/oasis/clapirn_training_data.csv"
    


    model.to(device)
    
    for batch_idx, data in enumerate(testing_generator):
        moving_img, fixed_img, moving_label, fixed_label, moving_index, fixed_index = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4], data[5]
      

        print("batch_idx", batch_idx)
        with torch.no_grad():
            reg_code = torch.tensor(random.sample(lam.tolist(), batch_size), device = device).reshape(batch_size, 1)
            displacement_field = model(moving_img, fixed_img, reg_code)
            warped_label = transform_nearest(moving_label, displacement_field.permute(0, 2, 3, 4, 1), grid)
            dice_per_label = dice3D(torch.floor(warped_label), torch.floor(fixed_label))

            num_folded_voxels, jdet_mean, jdet_std = JacboianDet(displacement_field.permute(0, 2, 3, 4, 1), grid)

            for i in range(batch_size):
                save_csv(batch_idx, moving_index[i], fixed_index[i], dice_per_label[i].unsqueeze(0), dice_per_label[i].mean().item(), jdet_mean[i].item(), jdet_std[i].item(), num_folded_voxels[i].item(), reg_code[i].item(), columns, file_path)


def generating_validation_data():

    datapath = "data/oasis/validation"
    imgs = sorted(glob.glob(datapath + '/OASIS_OAS1_*_MR1/aligned_norm.nii.gz'))
    labels = sorted(glob.glob(datapath + '/OASIS_OAS1_*_MR1/aligned_seg35.nii.gz'))

    testing_generator = Data.DataLoader(
                        DatasetImgsLabels(imgs, labels),
                        batch_size=batch_size,
                        shuffle=True, drop_last = True, num_workers=2)

    print("length of data is ", len(testing_generator))


    columns = ['index','moving_index','fixed_index','dice_per_label', 'dice_mean', 'Jdet_mean', 'Jdet_std','count_folded_voxels','hyper_param']

    file_path = "data/oasis/clapirn_validation_data.csv"
    

    
    for batch_idx, data in enumerate(testing_generator):
        moving_img, fixed_img, moving_label, fixed_label, moving_index, fixed_index = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4], data[5]
      

        print("batch_idx", batch_idx)
        with torch.no_grad():
            reg_code = torch.tensor(random.sample(lam.tolist(), batch_size), device = device).reshape(batch_size, 1)
            displacement_field = model(moving_img, fixed_img, reg_code)
            warped_label = transform_nearest(moving_label, displacement_field.permute(0, 2, 3, 4, 1), grid)
            dice_per_label = dice3D(torch.floor(warped_label), torch.floor(fixed_label))
            num_folded_voxels, jdet_mean, jdet_std = JacboianDet(displacement_field.permute(0, 2, 3, 4, 1), grid)

            for i in range(batch_size):
                save_csv(batch_idx, moving_index[i], fixed_index[i], dice_per_label[i].unsqueeze(0), dice_per_label[i].mean().item(), jdet_mean[i].item(), jdet_std[i].item(), num_folded_voxels[i].item(), reg_code[i].item(), columns, file_path)

def generating_testing_data():
    print("generating testing data")
    datapath = "data/oasis/testing"
    imgs = sorted(glob.glob(datapath + '/OASIS_OAS1_*_MR1/aligned_norm.nii.gz'))
    labels = sorted(glob.glob(datapath + '/OASIS_OAS1_*_MR1/aligned_seg35.nii.gz'))

    testing_generator = Data.DataLoader(
                        DatasetImgsLabels(imgs, labels),
                        batch_size=batch_size,
                        shuffle=True, drop_last = True, num_workers=2)

    print("length of data is ", len(testing_generator))
    

    columns = ['index','moving_index','fixed_index','dice_per_label', 'dice_mean', 'Jdet_mean', 'Jdet_std','count_folded_voxels','hyper_param']

    file_path = "data/oasis/clapirn_testing_data.csv"
    
    for batch_idx, data in enumerate(testing_generator):
        moving_img, fixed_img, moving_label, fixed_label, moving_index, fixed_index = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4], data[5]
        print("batch_idx", batch_idx)
        with torch.no_grad():
            reg_code = torch.tensor(random.sample(lam.tolist(), batch_size), device = device).reshape(batch_size, 1)
            displacement_field = model(moving_img, fixed_img, reg_code)
            warped_label = transform_nearest(moving_label, displacement_field.permute(0, 2, 3, 4, 1), grid)
            dice_per_label = dice3D(torch.floor(warped_label), torch.floor(fixed_label))
            num_folded_voxels, jdet_mean, jdet_std = JacboianDet(displacement_field.permute(0, 2, 3, 4, 1), grid)

            for i in range(batch_size):
                save_csv(batch_idx, moving_index[i], fixed_index[i], dice_per_label[i].unsqueeze(0), dice_per_label[i].mean().item(), jdet_mean[i].item(), jdet_std[i].item(), num_folded_voxels[i].item(), reg_code[i].item(), columns, file_path)



        



if __name__ == "__main__":
    start_channel = 4
    modelpath = "models/pretrained_models/clapirn.pth"
    imgshape = (160, 192, 224)
    imgshape_4 = (160 / 4, 192 / 4, 224 / 4)
    imgshape_2 = (160 / 2, 192 / 2, 224 / 2) 
    mu = 0.0
    sigma = 0.7
    distrubution = torch.distributions.log_normal.LogNormal(mu, sigma)
    #bending energy
    zero = torch.zeros(20)
    low = torch.tensor([random.uniform(0.0, 0.01) for _ in range(300)])
    mid = torch.tensor([random.uniform(0.0, 0.1) for _ in range(1000)])
    high = torch.tensor([random.uniform(0.2, 1.25) for _ in range(1000)])


    lam = distrubution.sample((2000,))
    lam = (lam - lam.min())/ ((lam.max())/1.75 - lam.min())
    lam = torch.cat((zero, lam, low, mid, high), dim=0)



    # lam = torch.distributions.log_normal.LogNormal(mu, sigma)
    # lam = lam.sample((6000,))
    # lam = (lam - lam.min())/ ((lam.max()/1.75) - lam.min())
    # add_sample_zero = torch.zeros(500)
    # add_sample_low_low = torch.tensor([random.uniform(0.0, 0.01) for _ in range(4000)])
    # add_sample_low = torch.tensor([random.uniform(0.0, 0.1) for _ in range(8000)])
    # add_sample_high = torch.tensor([random.uniform(0.2, 1.25) for _ in range(8000)])
    # lam = torch.cat((lam, add_sample_zero, add_sample_low_low, add_sample_low, add_sample_high), dim=0)
        
    batch_size = 16
    range_flow = 0.4
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    model_lvl1 = Lvl1(2, 3, start_channel, is_train=True, imgshape=imgshape_4,
                                                                         range_flow=range_flow).to(device)
    model_lvl2 = Lvl2(2, 3, start_channel, is_train=True, imgshape=imgshape_2,
                                                                         range_flow=range_flow, model_lvl1=model_lvl1).to(device)

    model = Lvl3(2, 3, start_channel, is_train=False, imgshape=imgshape,
                                                                    range_flow=range_flow, model_lvl2=model_lvl2).to(device)


    transform = SpatialTransform_unit().to(device)
    transform_nearest = SpatialTransformNearest_unit().to(device)
    model.load_state_dict(torch.load(modelpath))

    for params in model_lvl1.parameters():
        params.requires_grad = False


    for param in model_lvl2.parameters():
        param.requires_grad = False

    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    transform.eval()
    transform_nearest.eval()
    model.to(device)

    grid = generate_grid_unit(imgshape)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).to(device).float()

   
    start_t = datetime.now()
    generating_training_data()
    generating_validation_data()
    generating_testing_data()
    end_t = datetime.now()
    total_t = end_t - start_t
    print("Time: ", total_t.total_seconds())
