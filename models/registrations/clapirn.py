

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Lvl1(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel, is_train=True, imgshape=(160, 192, 144), range_flow=0.4):
        super(Lvl1, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel

        self.range_flow = range_flow
        self.is_train = is_train

        self.imgshape = imgshape

        self.grid_1 = generate_grid_unit(self.imgshape)
        self.grid_1 = torch.from_numpy(np.reshape(self.grid_1, (1,) + self.grid_1.shape)).to(device).float()

        self.transform = SpatialTransform_unit().to(device)

        bias_opt = False

        self.input_encoder_lvl1 = self.input_feature_extract(self.in_channel, self.start_channel * 2, bias=bias_opt)

        self.down_conv = nn.Conv3d(self.start_channel * 2, self.start_channel * 4, 3, stride=2, padding=1, bias=bias_opt)

        self.resblock_group_lvl1 = self.resblock_seq(self.start_channel * 4, bias_opt=bias_opt)

        # self.up = torch.nn.Upsample(scale_factor=2, mode="trilinear")
        self.up = nn.ConvTranspose3d(self.start_channel * 4, self.start_channel * 4, 2, stride=2,
                               padding=0, output_padding=0, bias=bias_opt)

        self.down_avg = nn.AvgPool3d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

        self.output_lvl1 = self.outputs(self.start_channel * 6, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False)


    def resblock_seq(self, in_channels, bias_opt=False):
        layer = nn.ModuleList([
            PreActBlock_Conditional(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock_Conditional(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock_Conditional(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock_Conditional(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock_Conditional(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2)
            ]
        )
        return layer


    def input_feature_extract(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2),
                nn.Conv3d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias))
        return layer

    def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.ReLU())
        return layer

    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.Tanh())
        else:
            # layer = nn.Sequential(
            #     nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            #     nn.Tanh())
            layer = nn.Sequential(
                nn.Conv3d(in_channels, int(in_channels/2), kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2),
                nn.Conv3d(int(in_channels/2), out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.Softsign())
        return layer

    def forward(self, moving_img, fixed_img, lamda):

        cat_input = torch.cat((moving_img, fixed_img), 1)
        cat_input = self.down_avg(cat_input)
        cat_input_lvl1 = self.down_avg(cat_input)
        # print("in level 1 lamda shape is ", lamda, lamda.shape)
        down_y = cat_input_lvl1[:, 1:2, :, :, :]

        fea_e0 = self.input_encoder_lvl1(cat_input_lvl1)
        e0 = self.down_conv(fea_e0)

        # e0 = self.resblock_group_lvl1(e0)
        for i in range(len(self.resblock_group_lvl1)):
            if i % 2 == 0:
                e0 = self.resblock_group_lvl1[i](e0, lamda)
            else:
                e0 = self.resblock_group_lvl1[i](e0)

        e0 = self.up(e0)
        output_disp_e0_v = self.output_lvl1(torch.cat([e0, fea_e0], dim=1)) * self.range_flow
        warpped_inputx_lvl1_out = self.transform(moving_img, output_disp_e0_v.permute(0, 2, 3, 4, 1), self.grid_1)


        if self.is_train is True:
            return output_disp_e0_v, warpped_inputx_lvl1_out, down_y, output_disp_e0_v, e0
        else:
            return output_disp_e0_v


class Lvl2(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel, is_train=True, imgshape=(160, 192, 144), range_flow=0.4, model_lvl1=None):
        super(Lvl2, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel

        self.range_flow = range_flow
        self.is_train = is_train

        self.imgshape = imgshape

        self.model_lvl1 = model_lvl1

        self.grid_1 = generate_grid_unit(self.imgshape)
        self.grid_1 = torch.from_numpy(np.reshape(self.grid_1, (1,) + self.grid_1.shape)).to(device).float()

        self.transform = SpatialTransform_unit().to(device)

        bias_opt = False

        self.input_encoder_lvl1 = self.input_feature_extract(self.in_channel+3, self.start_channel * 2, bias=bias_opt)

        self.down_conv = nn.Conv3d(self.start_channel * 2, self.start_channel * 4, 3, stride=2, padding=1, bias=bias_opt)

        self.resblock_group_lvl1 = self.resblock_seq(self.start_channel * 4, bias_opt=bias_opt)

        self.up_tri = torch.nn.Upsample(scale_factor=2, mode="trilinear")
        self.up = nn.ConvTranspose3d(self.start_channel * 4, self.start_channel * 4, 2, stride=2,
                                     padding=0, output_padding=0, bias=bias_opt)

        self.down_avg = nn.AvgPool3d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

        self.output_lvl1 = self.outputs(self.start_channel * 6, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False)


    def unfreeze_modellvl1(self):
        # unFreeze model_lvl1 weight
        print("\nunfreeze model_lvl1 parameter")
        for param in self.model_lvl1.parameters():
            param.requires_grad = True

    def resblock_seq(self, in_channels, bias_opt=False):
        layer = nn.ModuleList([
            PreActBlock_Conditional(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock_Conditional(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock_Conditional(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock_Conditional(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock_Conditional(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2)
            ]
        )
        return layer

    def input_feature_extract(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                              bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2),
                nn.Conv3d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias))
        return layer

    def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.ReLU())
        return layer

    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.Tanh())
        else:
            # layer = nn.Sequential(
            #     nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            #     nn.Tanh())
            layer = nn.Sequential(
                nn.Conv3d(in_channels, int(in_channels/2), kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2),
                nn.Conv3d(int(in_channels/2), out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.Softsign())
        return layer

    def forward(self, moving_img, fixed_img, lamda):
        lvl1_disp, _, _, lvl1_v, lvl1_embedding = self.model_lvl1(moving_img, fixed_img, lamda)
        lvl1_disp_up = self.up_tri(lvl1_disp)

        x_down = self.down_avg(moving_img)
        y_down = self.down_avg(fixed_img)

        warpped_x = self.transform(x_down, lvl1_disp_up.permute(0, 2, 3, 4, 1), self.grid_1)

        cat_input_lvl2 = torch.cat((warpped_x, y_down, lvl1_disp_up), 1)

        fea_e0 = self.input_encoder_lvl1(cat_input_lvl2)
        e0 = self.down_conv(fea_e0)

        e0 = e0 + lvl1_embedding

        # e0 = self.resblock_group_lvl1(e0)
        for i in range(len(self.resblock_group_lvl1)):
            if i % 2 == 0:
                e0 = self.resblock_group_lvl1[i](e0, lamda)
            else:
                e0 = self.resblock_group_lvl1[i](e0)

        e0 = self.up(e0)
        output_disp_e0_v = self.output_lvl1(torch.cat([e0, fea_e0], dim=1)) * self.range_flow
        compose_field_e0_lvl1 = lvl1_disp_up + output_disp_e0_v
        warpped_inputx_lvl1_out = self.transform(moving_img, compose_field_e0_lvl1.permute(0, 2, 3, 4, 1), self.grid_1)

        if self.is_train is True:
            return compose_field_e0_lvl1, warpped_inputx_lvl1_out, y_down, output_disp_e0_v, lvl1_v, e0
        else:
            return compose_field_e0_lvl1


class Lvl3(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel, is_train=True, imgshape=(160, 192, 144), range_flow=0.4,
                 model_lvl2=None):
        super(Lvl3, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel

        self.range_flow = range_flow
        self.is_train = is_train

        self.imgshape = imgshape

        self.model_lvl2 = model_lvl2

        self.grid_1 = generate_grid_unit(self.imgshape)
        self.grid_1 = torch.from_numpy(np.reshape(self.grid_1, (1,) + self.grid_1.shape)).to(device).float()

        self.transform = SpatialTransform_unit().to(device)

        bias_opt = False

        self.input_encoder_lvl1 = self.input_feature_extract(self.in_channel+3, self.start_channel * 2, bias=bias_opt)

        self.down_conv = nn.Conv3d(self.start_channel * 2, self.start_channel * 4, 3, stride=2, padding=1, bias=bias_opt)

        self.resblock_group_lvl1 = self.resblock_seq(self.start_channel * 4, bias_opt=bias_opt)

        self.up_tri = torch.nn.Upsample(scale_factor=2, mode="trilinear")
        self.up = nn.ConvTranspose3d(self.start_channel * 4, self.start_channel * 2, 2, stride=2,
                                     padding=0, output_padding=0, bias=bias_opt)

        # self.down_avg = nn.AvgPool3d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

        self.output_lvl1 = self.outputs(self.start_channel * 4, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False)


    def unfreeze_modellvl2(self):
        # unFreeze model_lvl1 weight
        print("\nunfreeze model_lvl2 parameter")
        for param in self.model_lvl2.parameters():
            param.requires_grad = True

    def resblock_seq(self, in_channels, bias_opt=False):
        layer = nn.ModuleList([
            PreActBlock_Conditional(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock_Conditional(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock_Conditional(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock_Conditional(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock_Conditional(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2)
            ]
        )
        return layer

    def input_feature_extract(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                              bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2),
                nn.Conv3d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias))
        return layer

    def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.ReLU())
        return layer

    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.Tanh())
        else:
            # layer = nn.Sequential(
            #     nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            #     nn.Tanh())
            layer = nn.Sequential(
                nn.Conv3d(in_channels, int(in_channels/2), kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2),
                nn.Conv3d(int(in_channels/2), out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.Softsign())
        return layer

    def forward(self, moving_img, fixed_img, lamda):
        # compose_field_e0_lvl1, warpped_inputx_lvl1_out, down_y, output_disp_e0_v, lvl1_v, e0
        # lamda = torch.ones((batch_size, 1), device = moving_img.device) * 0.5
        lvl2_disp, _, _, lvl2_v, lvl1_v, lvl2_embedding = self.model_lvl2(moving_img, fixed_img, lamda)
        lvl2_disp_up = self.up_tri(lvl2_disp)
        warpped_x = self.transform(moving_img, lvl2_disp_up.permute(0, 2, 3, 4, 1), self.grid_1)

        cat_input = torch.cat((warpped_x, fixed_img, lvl2_disp_up), 1)

        fea_e0 = self.input_encoder_lvl1(cat_input)
        e0 = self.down_conv(fea_e0)

        e0 = e0 + lvl2_embedding

        for i in range(len(self.resblock_group_lvl1)):
            if i % 2 == 0:
                e0 = self.resblock_group_lvl1[i](e0, lamda)
            else:
                e0 = self.resblock_group_lvl1[i](e0)
        
        e0 = self.up(e0)
        output_disp_e0_v = self.output_lvl1(torch.cat([e0, fea_e0], dim=1)) * self.range_flow
        compose_field_e0_lvl1 = output_disp_e0_v + lvl2_disp_up

        warpped_inputx_lvl1_out = self.transform(moving_img, compose_field_e0_lvl1.permute(0, 2, 3, 4, 1), self.grid_1)

        if self.is_train is True:
            return compose_field_e0_lvl1, warpped_inputx_lvl1_out, fixed_img, output_disp_e0_v, lvl1_v, lvl2_v, e0
        else:
            return compose_field_e0_lvl1


class ConditionalInstanceNorm(nn.Module):
    def __init__(self, in_channel, latent_dim=64):
        super().__init__()

        self.norm = nn.InstanceNorm3d(in_channel)

        self.style = nn.Linear(latent_dim, in_channel * 2)

        self.style.bias.data[:in_channel] = 0
        self.style.bias.data[in_channel:] = 0

    def forward(self, input, latent_code):
        # style [batch_size, in_channels*2] => [batch_size, in_channels*2, 1, 1, 1]
        style = self.style(latent_code).unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        gamma, beta = style.chunk(2, dim=1)

        out = self.norm(input)
        # out = input
        out = (1. + gamma) * out + beta

        return out


class PreActBlock_Conditional(nn.Module):
    """Pre-activation version of the BasicBlock + Conditional instance normalization"""
    expansion = 1

    def __init__(self, in_planes, planes, num_group=4, stride=1, bias=False, latent_dim=64, mapping_fmaps=64):
        super(PreActBlock_Conditional, self).__init__()
        self.ai1 = ConditionalInstanceNorm(in_planes, latent_dim=latent_dim)
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.ai2 = ConditionalInstanceNorm(in_planes, latent_dim=latent_dim)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=bias)

        self.mapping = nn.Sequential(
            nn.Linear(1, mapping_fmaps),
            nn.LeakyReLU(0.2),
            nn.Linear(mapping_fmaps, mapping_fmaps),
            nn.LeakyReLU(0.2),
            nn.Linear(mapping_fmaps, mapping_fmaps),
            nn.LeakyReLU(0.2),
            nn.Linear(mapping_fmaps, latent_dim),
            nn.LeakyReLU(0.2)
        )

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=bias)
            )

    def forward(self, x, lamda):

        latent_fea = self.mapping(lamda)

        out = F.leaky_relu(self.ai1(x, latent_fea), negative_slope=0.2)

        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)

        out = self.conv2(F.leaky_relu(self.ai2(out, latent_fea), negative_slope=0.2))

        out += shortcut
        return out


class SpatialTransform_unit(nn.Module):
    def __init__(self):
        super(SpatialTransform_unit, self).__init__()

    def forward(self, x, flow, sample_grid):
        sample_grid = sample_grid+ flow
        flow = torch.nn.functional.grid_sample(x, sample_grid, mode='bilinear', padding_mode="border", align_corners=True)

        return flow


class SpatialTransformNearest_unit(nn.Module):
    def __init__(self):
        super(SpatialTransformNearest_unit, self).__init__()

    def forward(self, x, flow, sample_grid):

        sample_grid = sample_grid + flow
        flow = torch.nn.functional.grid_sample(x, sample_grid, mode='nearest', padding_mode="border", align_corners=True)
    #return batchsize, channel, depth, height, width
        return flow


def smoothloss(y_pred):
    dy = torch.abs(y_pred[:,:,1:, :, :] - y_pred[:,:, :-1, :, :])
    dx = torch.abs(y_pred[:,:,:, 1:, :] - y_pred[:,:, :, :-1, :])
    dz = torch.abs(y_pred[:,:,:, :, 1:] - y_pred[:,:, :, :, :-1])
    return (torch.mean(dx * dx)+torch.mean(dy*dy)+torch.mean(dz*dz))/3.0


def JacboianDet(y_pred, sample_grid):
    batch_count_num_voxels = torch.rand(1, 1, device=device)
    batch_count_mean = torch.rand(1, 1, device=device)
    batch_count_std = torch.rand(1, 1, device=device)
    for i in range(y_pred.shape[0]):

        J = y_pred[i].unsqueeze(0)+ sample_grid
        # print("in jacboianDet", y_pred[i].unsqueeze(0).shape, sample_grid.shape)

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





def neg_Jdet_loss(y_pred, sample_grid):
    neg_Jdet = -1.0 * JacboianDet(y_pred, sample_grid)
    selected_neg_Jdet = F.relu(neg_Jdet)

    # return selected_neg_Jdet
    return torch.mean(selected_neg_Jdet)


def mse_loss(input, target):
    y_true_f = input.view(-1)
    y_pred_f = target.view(-1)
    diff = y_true_f-y_pred_f
    mse = torch.mul(diff,diff).mean()   
    return mse


class NCC(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    """
    def __init__(self, win=5, eps=1e-5):
        super(NCC, self).__init__()
        self.win = win
        self.eps = eps
        self.w_temp = win

    def forward(self, I, J):
        ndims = 3
        win_size = self.w_temp

        # set window size
        if self.win is None:
            self.win = [5] * ndims
        else:
            self.win = [self.w_temp] * ndims

        weight_win_size = self.w_temp
        weight = torch.ones((1, 1, weight_win_size, weight_win_size, weight_win_size), device=I.device, requires_grad=False)
        conv_fn = F.conv3d

        # compute CC squares
        I2 = I*I
        J2 = J*J
        IJ = I*J

        # compute filters
        # compute local sums via convolution
        I_sum = conv_fn(I, weight, padding=int(win_size/2))
        J_sum = conv_fn(J, weight, padding=int(win_size/2))
        I2_sum = conv_fn(I2, weight, padding=int(win_size/2))
        J2_sum = conv_fn(J2, weight, padding=int(win_size/2))
        IJ_sum = conv_fn(IJ, weight, padding=int(win_size/2))

        # compute cross correlation
        win_size = np.prod(self.win)
        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross * cross / (I_var * J_var + self.eps)

        # return negative cc.
        return -1.0 * torch.mean(cc)


class multi_resolution_NCC(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    """
    def  __init__(self, win=None, eps=1e-5, scale=3):
        super(multi_resolution_NCC, self).__init__()
        self.num_scale = scale
        # self.similarity_metric = NCC(win=win)

        self.similarity_metric = []

        for i in range(scale):
            self.similarity_metric.append(NCC(win=win - (i*2)))
            # self.similarity_metric.append(Normalized_Gradient_Field(eps=0.01))

    def forward(self, I, J):
        total_NCC = []

        for i in range(self.num_scale):
            current_NCC = self.similarity_metric[i](I, J)
            total_NCC.append(current_NCC/(2**i))
            # print(scale_I.size(), scale_J.size())

            I = nn.functional.avg_pool3d(I, kernel_size=3, stride=2, padding=1, count_include_pad=False)
            J = nn.functional.avg_pool3d(J, kernel_size=3, stride=2, padding=1, count_include_pad=False)

        return sum(total_NCC)



def generate_grid_unit(imgshape):
    x = (np.arange(imgshape[0]) - ((imgshape[0]-1)/2)) / (imgshape[0]-1) * 2
    y = (np.arange(imgshape[1]) - ((imgshape[1]-1)/2)) / (imgshape[1]-1) * 2
    z = (np.arange(imgshape[2]) - ((imgshape[2]-1)/2)) / (imgshape[2]-1) * 2
    grid = np.rollaxis(np.array(np.meshgrid(z, y, x)), 0, 4)
    grid = np.swapaxes(grid,0,2)
    grid = np.swapaxes(grid,1,2)
    return grid





