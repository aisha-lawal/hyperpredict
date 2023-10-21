import torch
import numpy as np
import torch.utils.data as Data
from torch.utils.data import DataLoader
from itertools import permutations
import nibabel as nib
import glob
import torch.nn as nn
import pandas as pd
from models.encoders.clapirn import ConditionalLaplacianLvl1, ConditionalLaplacianLvl2, ConditionalLaplacianLvl3
from models.registrations.clapirn import Lvl1, Lvl2, Lvl3
from models.encoders.symnet import SYMNet
import os
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mse_loss = nn.MSELoss()

def load(img):
    img = nib.load(img).get_fdata()
    img = np.reshape(img, (1,) + img.shape)

    return img

def resize(img):
    img = F.interpolate(img, [224, 224, 224], mode = "nearest").squeeze(0).permute(1,2,3,0)
    return img

def load_with_header(img):
   
    image = nib.load(img)
    header, affine = image.header, image.affine
    image = image.get_fdata()
    image = np.reshape(image, (1,) + image.shape)
    return image, affine, header

def imgnorm(img):
    max_val = np.max(img)
    min_val = np.min(img)
    norm_img = (img - min_val) / (max_val - min_val)

    return norm_img

def save_nifti(image, save_as, affine, header =None):
    img = nib.Nifti1Image(image.cpu().numpy(), affine = affine.squeeze(0).cpu().numpy(), header = header)
    nib.save(img, save_as)


def save_csv(epoch, batch_idx, idx, moving_index, fixed_index, dice_per_label, dice_mean, Jdet_mean, Jdet_std, count_folded_voxels, hyper_param, columns, file_path):
    dice_per_label = [dice_per_label.cpu()]
    row =[epoch+1, batch_idx+1,idx.item(), moving_index,fixed_index ,dice_per_label, dice_mean, Jdet_mean, Jdet_std, count_folded_voxels , hyper_param]
    
    data = dict(zip(columns, row))
    df = pd.DataFrame(data = data)
    df.to_csv(file_path,  mode='a', index=False, header=True if batch_idx == 0 else False)


def load_nii(warped_label, jac_det, deformation_field, bspline):

    warped_label = load(warped_label)
    jac_det = load(jac_det)
    deformation_field = load(deformation_field)
    bspline = load(bspline)
    # print("shape of deformation field is ", warped_label.shape, deformation_field.shape, jac_det.shape, bspline.shape) # [1, 160, 192, 224] (1, 160, 192, 224, 1, 3) (1, 160, 192, 224)
    return torch.tensor(warped_label, device = device, dtype=torch.float32), torch.tensor(jac_det, device = device, dtype=torch.float32), torch.tensor(deformation_field, device =device, dtype=torch.float32)


def set_niftyreg():
        #compute bspline
        get_bspline = "reg_f3d -ref {} -flo {} -be {} -sx {} -cpp bspline.nii -res warped_image.nii"

        #convert the bspline to deformation field
        get_deformation_field = "reg_transform -ref {} -def bspline.nii deformation_field.nii"

        #warp segmentation
        get_warp_seg = "reg_resample -ref {} -flo {} -inter 0 -trans deformation_field.nii -res warped_label.nii"

        #compute jacobian_det
        get_jac_det = "reg_jacobian -ref {} -trans deformation_field.nii -jac jac_det_map.nii" #bspline or deformation field?

        return get_bspline, get_deformation_field, get_warp_seg, get_jac_det



def niftyreg(moving_index, fixed_index, fixed_label, bending_energy, spacing, task):

    moving_image = "data/oasis/" +task +"/OASIS_OAS1_"+moving_index[0]+"_MR1/aligned_norm.nii"
    fixed_image = "data/oasis/" +task +"/OASIS_OAS1_"+fixed_index[0]+"_MR1/aligned_norm.nii"
    mov_lbl = "data/oasis/" +task +"/OASIS_OAS1_"+moving_index[0]+"_MR1/aligned_seg35.nii"
    fxd_lbl = "data/oasis/" +task +"/OASIS_OAS1_"+fixed_index[0]+"_MR1/aligned_seg35.nii"
    
    warped_label = "warped_label.nii"
    jac_det = "jac_det_map.nii"
    deformation_field = "deformation_field.nii" 
    bspline = "bspline.nii" 
     
    #do registeration
    get_bspline, get_deformation_field, get_warp_seg, get_jac_det = set_niftyreg()
    os.system(get_bspline.format(fixed_image, moving_image, bending_energy, spacing))
    os.system(get_deformation_field.format(fixed_image))
    os.system(get_warp_seg.format(fxd_lbl, mov_lbl))
    os.system(get_jac_det.format(fixed_image))
   

    warped_label, jac_det, deformation_field = load_nii(warped_label, jac_det, deformation_field, bspline)
    #dice after registration
    # print("in niftyreg warped shapes ", warped_label.shape, fixed_label.shape)  #shape is [1, 144, 152, 192]
    dice_per_label = dice3D(torch.floor(warped_label), torch.floor(fixed_label).squeeze(0)) 
    print("in niftyreg check jac_det shape ", jac_det.shape)

    folded_voxels = torch.count_nonzero(jac_det<0) #neg value computation
    return dice_per_label, folded_voxels.unsqueeze(0).unsqueeze(0).float()



def compute_metric(moving_label, fixed_label, displacement_field, grid_unit):

    warped_label = transform_nearest(moving_label, displacement_field.permute(0, 2, 3, 4, 1),grid_unit)
    count_folded_voxels = JacobianDet(displacement_field.permute(0, 2, 3, 4, 1), grid_unit)
    dice_per_label = dice3D(torch.floor(warped_label), torch.floor(fixed_label))

    return dice_per_label, count_folded_voxels


def JacobianDet(y_pred, sample_grid):
    batch_count_num_voxels = torch.rand(1, 1, device=device)

    for i in range(y_pred.shape[0]):

        J = y_pred[i].unsqueeze(0)+ sample_grid

        dy = J[:, 1:, :-1, :-1, :] - J[:, :-1, :-1, :-1, :]
        dx = J[:, :-1, 1:, :-1, :] - J[:, :-1, :-1, :-1, :]
        dz = J[:, :-1, :-1, 1:, :] - J[:, :-1, :-1, :-1, :]

        Jdet0 = dx[:,:,:,:,0] * (dy[:,:,:,:,1] * dz[:,:,:,:,2] - dy[:,:,:,:,2] * dz[:,:,:,:,1])
        Jdet1 = dx[:,:,:,:,1] * (dy[:,:,:,:,0] * dz[:,:,:,:,2] - dy[:,:,:,:,2] * dz[:,:,:,:,0])
        Jdet2 = dx[:,:,:,:,2] * (dy[:,:,:,:,0] * dz[:,:,:,:,1] - dy[:,:,:,:,1] * dz[:,:,:,:,0])

        Jdet = Jdet0 - Jdet1 + Jdet2
        count_num_folded_voxels = torch.count_nonzero(Jdet < 0)
        batch_count_num_voxels = torch.cat([batch_count_num_voxels, count_num_folded_voxels.unsqueeze(0).unsqueeze(0)], dim=0)

    batch_count_num_voxels = batch_count_num_voxels[1:]
    return batch_count_num_voxels

class Identity(nn.Module):
  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x


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
    

class DatasetImgsLabelsRegistered(Data.Dataset):
    def __init__(self, registration_model, encoder_model = "clapirn", path = None, data = "oasis", task = None):
        super(DatasetImgsLabelsRegistered, self).__init__()
       
        self.data = data
        self.encoder_model = encoder_model
        self.registration_model = registration_model
        self.path = path
        self.task = task
        self.df = pd.read_csv(self.path, converters={'moving_index': str, 'fixed_index': str})



    def __getitem__(self, index):
      
        loaded_data = load_data(self.df.iloc[index],  self.task, self.registration_model)

        if self.registration_model == "clapirn":
            return loaded_data[0], loaded_data[1], loaded_data[2], loaded_data[3], loaded_data[4], loaded_data[5], loaded_data[6], loaded_data[7]
        
        elif self.registration_model == "niftyreg":
            return loaded_data[0], loaded_data[1], loaded_data[2], loaded_data[3], loaded_data[4], loaded_data[5], loaded_data[6], loaded_data[7], loaded_data[8]
      

    def __len__(self):
        return len(self.df)

def load_data(dataframe, task, registration_model):
    data_path = "data/oasis/"+task
    mov_img_path = data_path + '/OASIS_OAS1_'+ pad_with_zeros(dataframe['moving_index']) +'_MR1'+'/aligned_norm.nii.gz'
    fix_img_path = data_path + '/OASIS_OAS1_'+ pad_with_zeros(dataframe['fixed_index'])+'_MR1'+'/aligned_norm.nii.gz'
    dice_per_label = dataframe["dice_per_label"][2:-2]
    dice_per_label = np.fromstring(dice_per_label, dtype=np.float32, sep =',')
    dice_mean = torch.tensor(float(dataframe["dice_mean"]))
    jdet_mean = torch.tensor(float(dataframe["Jdet_mean"]))
    jdet_std = torch.tensor(float(dataframe["Jdet_std"]))
    count_folded_voxels = torch.tensor(int(dataframe["count_folded_voxels"]))
    if registration_model  == "clapirn":
        moving_img = imgnorm(load(mov_img_path))
        fixed_img = imgnorm(load(fix_img_path))
        hyper_param = torch.tensor(float(dataframe["hyper_param"]))
        return torch.from_numpy(moving_img).float(), torch.from_numpy(fixed_img).float(), torch.from_numpy(dice_per_label), dice_mean.unsqueeze(-1),jdet_mean.unsqueeze(-1), jdet_std.unsqueeze(-1), count_folded_voxels.unsqueeze(-1), hyper_param.unsqueeze(-1)
    
    elif registration_model == "niftyreg":
        moving_img = imgnorm(load(mov_img_path))
        fixed_img = imgnorm(load(fix_img_path))
        bending_energy = torch.tensor(float(dataframe["bending_energy"]))
        spacing = torch.tensor(float(dataframe["spacing"]))

        return torch.from_numpy(moving_img).float(), torch.from_numpy(fixed_img).float(), torch.from_numpy(dice_per_label), dice_mean.unsqueeze(-1),jdet_mean.unsqueeze(-1), jdet_std.unsqueeze(-1), count_folded_voxels.unsqueeze(-1), bending_energy.unsqueeze(-1), spacing.unsqueeze(-1)
    
        

    
class DatasetImages(Data.Dataset):
  def __init__(self, imgs, data = "oasis"):
        super(DatasetImages, self).__init__()

        self.imgs = imgs
        self.index_pair = list(permutations(imgs, 2))
        self.data = data

  def __len__(self):
        return len(self.index_pair)

  def __getitem__(self, index):
        moving_img = load(self.index_pair[index][0])
        fixed_img = load(self.index_pair[index][1])


        if self.data == "oasis":
            moving_index, fixed_index = self.img_pair[index][0].split("_")[2], self.img_pair[index][1].split("_")[2]
            
            return torch.from_numpy(imgnorm(moving_img)).float(), torch.from_numpy(imgnorm(fixed_img)).float(), moving_index, fixed_index
        
        elif self.data == "abdominal_ct":
            moving_index, fixed_index = self.img_pair[index][0].split("_")[1], self.img_pair[index][1].split("_")[1]

            return torch.from_numpy(imgnorm(moving_img)).float(), torch.from_numpy(imgnorm(fixed_img)).float(), moving_index, fixed_index

        else:
            moving_index, fixed_index = self.img_pair[index][0].split("_")[2], self.img_pair[index][1].split("_")[2]
            return torch.from_numpy(imgnorm(moving_img)).float(), torch.from_numpy(imgnorm(fixed_img)).float()

      
  

    
def weight_step_function(gaussian_loss, poisson_loss):
    """
    Adjust loss based on number of digit in larger loss
    """
    num_digits = len(str(int(max(gaussian_loss, poisson_loss))))
    weight = 1 * 10  ** (-num_digits)
    return weight

    
def pad_with_zeros(string):
    if len(string) < 4:
        num_zeros = 4 - len(string)
        padded_string = '0'* num_zeros + string
        return padded_string
    else:
        return string


class RegisteredDataset(Data.Dataset):
    def __init__(self, file_path, epoch) -> None:
        super(RegisteredDataset, self).__init__()
        self.file_path = file_path +str(epoch+1)+'.csv'
        self.df = pd.read_csv(self.file_path, converters={'moving_index': str, 'fixed_index': str})
    def __getitem__(self, index):
        loaded_data = load_data(self.df.iloc[index], self.file_path)
        return loaded_data

        
    def __len__(self):
        return len(self.df)
        
  
             


class SpatialTransform_unit(nn.Module):
    def __init__(self):
        super(SpatialTransform_unit, self).__init__()

    def forward(self, moving_img, flow, sample_grid):
        sample_grid = sample_grid + flow
        flow = torch.nn.functional.grid_sample(moving_img, sample_grid, mode='bilinear', padding_mode="border", align_corners=True)

        return flow

class SpatialTransformNearest_unit(nn.Module):
    """
    Takes in image (B, C, X, Y, Z), and grid of (B, X, Y, Z, 3), output is (B, C, X, Y, Z)
    """
    def __init__(self):
        super(SpatialTransformNearest_unit, self).__init__()

    def forward(self, moving_img, flow, sample_grid):
        sample_grid = sample_grid + flow
        flow = torch.nn.functional.grid_sample(moving_img, sample_grid, mode='nearest', padding_mode="border", align_corners=True)

        return flow



def jacobian_determinant_batch(batch_deformation_field):
    """
    Parameters: 
        deformation field: tensor with shape (B, 2, H, W)

    returns: a scalar(for each value in the batch) standard deviation of the 
    log jacobian determinant over the entire image(taking the mean)

    NB: torch.gradient is used, so central differences is used
    
    """
    batch_jdet = []
    for deformation_field in batch_deformation_field:
        x_component = deformation_field[0]
        y_component = deformation_field[1]

        J = torch.gradient(deformation_field )
        dx = J[1] # derivative of x component wrt y coordinate
        dy = J[2] # derivative of y component wrt x corrdinate

        Jdet = (dx[0,...] * dy[1,...] - dy[0,...] * dx[1,...]).mean()
        batch_jdet.append(Jdet)
    return torch.tensor(batch_jdet, device = device).unsqueeze(-1)



def num_folded_voxels(batch_deformation_field):
    """
    Parameters: 
        deformation field: tensor with shape (B, 2, H, W)

    returns: the number of folded voxels (gotten by computing the |J|, thresholding the values at an epsilon, and taking the sum)

    NB: torch.gradient is used, so central differences is used
    
    """
    batch_sum_folded_voxels = []
    for deformation_field in batch_deformation_field:
        x_component = deformation_field[0]
        y_component = deformation_field[1]

        J = torch.gradient(deformation_field )
        dx = J[1] # derivative of x component wrt y coordinate
        dy = J[2] # derivative of y component wrt x corrdinate

        Jdet = (dx[0,...] * dy[1,...] - dy[0,...] * dx[1,...])

        epsilon = 1e-3
        mask = Jdet > epsilon #mask negative |J|
        sum_folded_voxels = torch.count_nonzero(Jdet < 0)
        batch_sum_folded_voxels.append(sum_folded_voxels)
    return torch.tensor(batch_sum_folded_voxels, device = device).unsqueeze(-1)





def jacobian_determinant(disp):

    batch = torch.rand(1, 1, device=device)

    """
    Input B,H,W,D,C, output H,W,D
    
    """
    for i in range(disp.shape[0]):
        J = torch.gradient(disp[i])

        dx = J[0]
        dy = J[1]
        dz = J[2]

        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        Jdet = Jdet0 - Jdet1 + Jdet2
        count_num_folded_voxels = torch.count_nonzero(Jdet < 0)
        batch = torch.cat([batch, count_num_folded_voxels.unsqueeze(0).unsqueeze(0)], dim=0)
    batch = batch[1:]
    return batch

def dice3D_nifty(warped_label, fixed_label):
    dice = 0
    dice_per_label = []
    num_count = 0
    labels = 35
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
       
    dice_per_label = torch.tensor(dice_per_label, device = device)
    return dice_per_label, dice_per_label.mean()

def dice3D(warped_lbl, fixed_lbl):
    batch = torch.ones(1, 35, device=device)
    for i in range(warped_lbl.shape[0]):
        dice = 0
        dice_per_label = []
        num_count = 0
        labels = 35
        fixed_label = fixed_lbl[i]
        warped_label = warped_lbl[i]
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


def hausdorf_distance(moving_label, fixed_label):

    pass

transform_nearest = SpatialTransformNearest_unit().to(device)
transform = SpatialTransform_unit().to(device)

for param in transform.parameters():
    param.requires_grad = False

for param in transform_nearest.parameters():
    param.requires_grad = False

def normalize(loss, target):
    return (loss-target.min())/(target.max()- target.min())

def gaussian_nllLoss(input, target, var):
    #should be computed for each batch, take sample-wise mean, then batch-wise mean.
    log_var = torch.log(var) #using fixed variance, i can use a varying var and clamp at epsilon
    loss = (log_var + ((torch.pow(input - target, 2))/var)) * 0.5
    return loss.mean()

def poisson_nllLoss(input, target):
    loss = input - target * torch.log(input + 1e-8) #log_input = False
    return loss.mean()



#move to gpu
def generate_grid_unit(imgshape):
    x = (np.arange(imgshape[0]) - ((imgshape[0]-1)/2)) / (imgshape[0]-1) * 2
    y = (np.arange(imgshape[1]) - ((imgshape[1]-1)/2)) / (imgshape[1]-1) * 2
    z = (np.arange(imgshape[2]) - ((imgshape[2]-1)/2)) / (imgshape[2]-1) * 2
    grid = np.rollaxis(np.array(np.meshgrid(z, y, x)), 0, 4)
    grid = np.swapaxes(grid,0,2)
    grid = np.swapaxes(grid,1,2)
    return grid


class MaskedAutoEncoder():
    def __init__(self):
        super(MaskedAutoEncoder, self).__init__()
        pass

    def prepare_model(self, chkpt_dir, arch='mae_vit_large_patch16'):
        model = getattr(models_mae, arch)()
        checkpoint = torch.load(chkpt_dir, map_location= device)
        msg = model.load_state_dict(checkpoint['model'], strict=False)
        print(msg)
        return model
    

    def run_mae(self, image, model):
        image = image.unsqueeze(dim=0)
        image = torch.einsum('nhwc->nchw', image)
      
        encoding= model(image, mask_ratio=0.0)

        
        return encoding

    
  

    def generate_encoding(self, moving_image, fixed_image, model_mae): 
        mov_img = resize(moving_image.unsqueeze(0))
        fixd_img = resize(fixed_image.unsqueeze(0)) 

        stack_encodings = torch.ones(1, 196, device=device)
    

        for i in range(mov_img.shape[2]):
            ch1_ch2 = torch.cat((mov_img[:, :, i, :], fixd_img[:, :, i, :]), dim=-1)
            avg = torch.mean(ch1_ch2 , dim = -1).unsqueeze(-1)
            img =  torch.cat((ch1_ch2, avg), dim=-1) 

            encoding = self.run_mae(img, model_mae) 
            encoding = torch.mean(encoding, dim=2) 
            stack_encodings = torch.cat([stack_encodings, encoding], dim = 0)


        print("stach enc",stack_encodings.shape)
        stack_encodings = stack_encodings[1:]
        print(stack_encodings.shape)
        enc_rep = torch.mean(stack_encodings, dim = 1).unsqueeze(0)

        return enc_rep



class SetParams(nn.Module):

    """
    Extra preprocessing stuff
    Takes in encoder and registration models parameters and freezes weights
    Encoders are located in models/encoders/{encoder_model}
    Registration models are located in models/registrations/{registration_model}
    Returns encoder, regsitration and data for any method called
    
    """
    def __init__(self):
        super(SetParams, self).__init__()
        pass



    def set_encoder(encoder_model, encoder_path, start_channel, imgshape, imgshape_2, imgshape_4, range_flow):
        if encoder_model == "clapirn":

            model_lvl1 = ConditionalLaplacianLvl1(2, 3, start_channel, is_train=True, imgshape=imgshape_4,
                                                                         range_flow=range_flow).to(device)
            model_lvl2 = ConditionalLaplacianLvl2(2, 3, start_channel, is_train=True, imgshape=imgshape_2,
                                                                         range_flow=range_flow, model_lvl1=model_lvl1).to(device)

            encoder = ConditionalLaplacianLvl3(2, 3, start_channel, is_train=False, imgshape=imgshape,
                                                                    range_flow=range_flow, model_lvl2=model_lvl2).to(device)


            encoder.load_state_dict(torch.load(encoder_path, map_location= device))
            for params in encoder.parameters():
                params.requires_grad = False


            for param in model_lvl1.parameters():
                param.requires_grad = False

            for param in model_lvl2.parameters():
                param.requires_grad = False

            print('clapirn Model loaded. All keys matched successfully')

            
            return encoder
        
        elif encoder_model == "symnet":
            encoder = SYMNet(2, 3, start_channel = 7).to(device)

            encoder_path = "models/pretrained_models/symnet.pth"
            encoder.load_state_dict(torch.load(encoder_path, map_location= device))
            for params in encoder.parameters():
                params.requires_grad = False
            print('SymNet Model loaded. All keys matched successfully')
            return encoder

        elif encoder_model == "mae":
            chkpt_dir = encoder_path #replace with encoder path
            model_mae = MaskedAutoEncoder().prepare_model(chkpt_dir, 'mae_vit_large_patch16')
            for params in model_mae.parameters():
                params.requires_grad = False
            
            model_mae.to(device)
            print('MAE Model loaded.')

            return model_mae



    def set_registration(registration_model, start_channel, imgshape, imgshape_2, imgshape_4, range_flow, registration_path):
        if registration_model == "clapirn":

            model_lvl1 = Lvl1(2, 3, start_channel, is_train=True, imgshape=imgshape_4,
                                                                         range_flow=range_flow).to(device)
            model_lvl2 = Lvl2(2, 3, start_channel, is_train=True, imgshape=imgshape_2,
                                                                         range_flow=range_flow, model_lvl1=model_lvl1).to(device)

            registration = Lvl3(2, 3, start_channel, is_train=False, imgshape=imgshape,
                                                                    range_flow=range_flow, model_lvl2=model_lvl2).to(device)


            registration.load_state_dict(torch.load(registration_path, map_location = device))

            for params in registration.parameters():
                params.requires_grad = False

            for param in model_lvl1.parameters():
                param.requires_grad = False

            for param in model_lvl2.parameters():
                param.requires_grad = False

            return registration
        

    def set_oasis_data(data_path, train_batch_size, validation_batch_size, test_batch_size, encoder_model):
        train_imgs = sorted(glob.glob(data_path + "training/OASIS_OAS1_*_MR1/aligned_norm.nii.gz"))
        train_labels = sorted(glob.glob(data_path  + "training/OASIS_OAS1_*_MR1/aligned_seg35.nii.gz"))
        training_generator = DataLoader(dataset = DatasetImgsLabels(train_imgs, train_labels, encoder_model, data = "oasis"), batch_size = train_batch_size, shuffle = True , drop_last = True, num_workers = 16)


        imgs_val = sorted(glob.glob(data_path + "validation/OASIS_OAS1_*_MR1/aligned_norm.nii.gz"))
        labels_val = sorted(glob.glob(data_path  + "validation/OASIS_OAS1_*_MR1/aligned_seg35.nii.gz"))
        validation_generator = DataLoader(dataset = DatasetImgsLabels(imgs_val, labels_val, encoder_model, data = "oasis"), batch_size = validation_batch_size, shuffle = False , drop_last = True, num_workers = 16)


        imgs_test = sorted(glob.glob(data_path + "testing/OASIS_OAS1_*_MR1/aligned_norm.nii.gz"))
        labels_test = sorted(glob.glob(data_path  + "testing/OASIS_OAS1_*_MR1/aligned_seg35.nii.gz"))
        testing_generator = DataLoader(dataset = DatasetImgsLabels(imgs_test, labels_test, encoder_model, data = "oasis"), batch_size = test_batch_size, shuffle = True , drop_last = True, num_workers = 16)


        imgs_test_sub = sorted(glob.glob(data_path + "testing/sub/OASIS_OAS1_*_MR1/aligned_norm.nii.gz"))
        labels_test_sub = sorted(glob.glob(data_path  + "testing/sub/OASIS_OAS1_*_MR1/aligned_seg35.nii.gz"))
        testing_generator_sub = DataLoader(dataset = DatasetImgsLabels(imgs_test_sub, labels_test_sub, encoder_model, data = "oasis"), batch_size = test_batch_size, shuffle = True , drop_last = True, num_workers = 16)


        return training_generator, validation_generator, testing_generator, testing_generator_sub

    def set_oasis_data_registered(data_path, train_batch_size, validation_batch_size, test_batch_size, encoder_model, registration_model):
        train_path = data_path + registration_model + "_training_data.csv"
        training_generator = DataLoader(dataset = DatasetImgsLabelsRegistered(registration_model, encoder_model, train_path,  data = "oasis", task = "training"), batch_size = train_batch_size, shuffle = True , drop_last = True, num_workers = 16)

        validation_path = data_path + registration_model + "_validation_data.csv"
        validation_generator = DataLoader(dataset = DatasetImgsLabelsRegistered(registration_model, encoder_model, validation_path, data = "oasis", task="validation"), batch_size = validation_batch_size, shuffle = False , drop_last = True, num_workers = 16)

        testing_path = data_path + registration_model + "_testing_data.csv"
        testing_generator = DataLoader(dataset = DatasetImgsLabelsRegistered(registration_model, encoder_model,testing_path, data = "oasis", task ="testing"), batch_size = test_batch_size, shuffle = False , drop_last = True, num_workers = 16)


        return training_generator, validation_generator, testing_generator

    def set_oasis_data_niftyreg():
        pass


    def set_abdominal_ct_data(data_path, train_batch_size, test_batch_size):
        train_imgs = sorted(glob.glob(data_path + "training/AbdomenCTCT_*_0000.nii"))
        train_labels = sorted(glob.glob(data_path  + "training_labels/AbdomenCTCT_*_0000.nii"))
        training_generator = DataLoader(dataset = DatasetImgsLabels(train_imgs, train_labels, data = "abdominal_ct"), batch_size = train_batch_size, shuffle = True , drop_last = True, num_workers = 2)


        return training_generator
