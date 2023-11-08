import torch
import torch.nn as nn
import numpy as np
from utils import generate_grid_unit, SetParams, niftyreg, compute_metric, MaskedAutoEncoder
import lightning.pytorch as pl
import pandas as pd
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HyperPredictLightningModule(pl.LightningModule):
    def __init__(self, hyper_predict, registration_model, encoder_model, imgshape, enc, batch_size, encoding_type):
        super(HyperPredictLightningModule, self).__init__()
        self.mapping, self.linear_mapping, self.dice_mapping, self.jacobian_mapping = hyper_predict
        self.compute_loss = ComputeLoss(2).to(device)
        self.enc = enc.to(self.device)
        self.registration_model = registration_model
        self.encoder_model = encoder_model
        self.batch_size = batch_size
        self.imgshape = imgshape
        self.enc.eval()
        self.nfv_clapirn = 211473
        # self.nfv_niftyreg = 23189
        self.nfv_niftyreg = 194404

        self.encoding_type = encoding_type
        self.grid = generate_grid_unit(self.imgshape)
        self.grid = torch.from_numpy(np.reshape(self.grid, (1,) + self.grid.shape)).to(device).float()
        self.maximum_nfv = 160 * 192 * 224
        #add labels         

    def forward(self, batch):
        if self.registration_model == "clapirn":
            with torch.no_grad():
                moving_img, fixed_img, target_dice, dice_mean,jdet_mean, jdet_std, target_jac, lamda= batch
                
                if self.encoder_model == "clapirn":
                    enc_rep = self.enc(moving_img, fixed_img, self.batch_size)
                    
                elif self.encoder_model == "symnet":
                    enc_rep = self.enc(moving_img, fixed_img)

                print("hyperparameters going in, lamda before: ", lamda, "lamda after: ",  torch.where(lamda > 0, lamda.log(), (lamda + 1e-5).log()))
                mapped_log_lamda = self.mapping(torch.where(lamda > 0, lamda.log(), (lamda + 1e-5).log()))
                
                #global statistics
                if self.encoding_type == "mean_encoding":
                    global_mean = self.global_statistics(enc_rep, self.encoding_type)
                    enc_rep_cat = torch.cat([global_mean, mapped_log_lamda], dim= 1)

                elif self.encoding_type == "mean_min_max_encoding":
                    global_mean, global_max, global_min = self.global_statistics(enc_rep, self.encoding_type)
                    enc_rep_cat = torch.cat([global_mean, global_max, global_min, mapped_log_lamda], dim= 1) 

        elif self.registration_model == "niftyreg":
            with torch.no_grad():

                moving_img, fixed_img, target_dice, dice_mean,jdet_mean, jdet_std, target_jac, be,le, spacing= batch

                if self.encoder_model == "clapirn": 
                    enc_rep = self.enc(moving_img, fixed_img, self.batch_size)
                elif self.encoder_model == "symnet":
                    enc_rep = self.enc(moving_img, fixed_img)

                print("hyperparameters going in, bending energy before: ", be, "bending_energy after: ", torch.where(be > 0, be.log(), (be + 1e-6).log()),
                      "linear elasticity before: ", le, "linear_elasticity after: ", torch.where(le > 0, le.log(), (le + 1e-6).log()), "spacing before: ", spacing, "spacing after: ", spacing.log())
                mapped_sx = self.mapping(spacing.log())
                mapped_be = self.mapping(torch.where(be > 0, be.log(), (be + 1e-6).log()))
                mapped_le = self.mapping(torch.where(le > 0, le.log(), (le + 1e-6).log()))
                
                 #global statistics
                if self.encoding_type == "mean_encoding":
                    global_mean = self.global_statistics(enc_rep, self.encoding_type)
                    enc_rep_cat = torch.cat([global_mean, mapped_sx, mapped_be, mapped_le], dim= 1)

                elif self.encoding_type == "mean_min_max_encoding":
                    global_mean, global_max, global_min = self.global_statistics(enc_rep, self.encoding_type)
                    enc_rep_cat = enc_rep_cat = torch.cat([global_mean, global_max, global_min, mapped_sx, mapped_be, mapped_le], dim= 1)

        #linear mapping
        linear_map = self.linear_mapping(enc_rep_cat)

        #dice and jacobian mapping 
        predicted_dice = self.dice_mapping(linear_map)
        predicted_jac = self.jacobian_mapping(linear_map)
        return predicted_dice, target_dice, predicted_jac, target_jac
    

    def training_step(self, batch, batch_idx):
        print("in training step: batch_index", batch_idx)
        predicted_dice, target_dice, predicted_jac, target_jac  = self.forward(batch)
        train_loss, dice_loss, jac_loss = self.compute_loss(predicted_dice, target_dice, predicted_jac, target_jac, self.registration_model)
        
        print("training_loss ", train_loss, dice_loss, jac_loss)
        print("training values ", predicted_dice[0], target_dice[0], predicted_jac, target_jac / self.nfv_niftyreg if self.registration_model == "niftyreg" else target_jac / self.nfv_clapirn)
        self.log("total_train_loss", train_loss, on_epoch=True, sync_dist=True, on_step=False) 
        self.log("train_dice_loss", dice_loss, on_epoch=True, sync_dist=True, on_step=False)
        self.log("train_jac_loss", jac_loss, on_epoch=True, sync_dist=True, on_step=False)
        return train_loss
    

    def validation_step(self, batch, batch_idx):
        print("in validation step: batch_index", batch_idx)

        predicted_dice, target_dice, predicted_jac, target_jac  = self.forward(batch)
        val_loss, dice_loss, jac_loss = self.compute_loss(predicted_dice, target_dice, predicted_jac, target_jac, self.registration_model)
        
        print("validation_loss ", val_loss, dice_loss, jac_loss)
        print("validation values ", predicted_dice[0], target_dice[0], predicted_jac, target_jac / self.nfv_niftyreg if self.registration_model == "niftyreg" else target_jac / self.nfv_clapirn)
        self.log("total_val_loss", val_loss, on_epoch=True, sync_dist=True, on_step = False)
        self.log("val_dice_loss", dice_loss, on_epoch=True, sync_dist=True, on_step = False)
        self.log("val_jac_loss", jac_loss, on_epoch=True, sync_dist=True, on_step = False)

        return val_loss
    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = 0.0001) 
        return optimizer
 
    def test_clapirn(self, pair_idx, batch, lamda, nfv_percent):
            moving_img, fixed_img, moving_label, fixed_lbl, moving_idx, fixed_idx = batch
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


            labels = {"Thalamus": lr_thalamus, "Hippocampus": lr_hippocampus, "Amygdala": lr_amygdala, "Accumbens": lr_accumbens, "Putamen": lr_putamen, "Pallidum": lr_pallidum, "Caudate": lr_caudate, "Cerebellum WM": lr_cerebellum_WM, "Cerebellum Cortex": lr_cerebellum_cortex, "Cerebral Cortex": lr_cerebral_cortex, "Cerebral WM": lr_cerebral_WM, "Vessel": lr_vessel, "Lateral Ventricle": lr_lateral_ventricle}

            #dataframe for dice per lamda
            columns = ["pair_idx", "moving_index", "fixed_index", "predicted_dice", "lamda", "predicted_jac"]
            dice_average_per_image_per_lamda = pd.DataFrame(columns = columns)

            #dataframe for dice per label 
            columns_label = ["pair_idx","moving_index", "fixed_index", "predicted_dice", "lamda", "label", "predicted_jac"]
            dice_average_per_label_per_lamda = pd.DataFrame(columns = columns_label)
            with torch.no_grad():

                if self.encoder_model == "clapirn":   
                    enc_rep = self.enc(moving_img, fixed_img, self.batch_size)
                elif self.encoder_model == "symnet": 
                    enc_rep = self.enc(moving_img, fixed_img)   

                if self.encoding_type == "mean_encoding":
                    global_mean = self.global_statistics(enc_rep, self.encoding_type)
                elif self.encoding_type == "mean_min_max_encoding":
                    global_mean, global_max, global_min =  self.global_statistics(enc_rep, self.encoding_type)  
                for i in lamda:
                    lamda = torch.tensor(i, device=self.device).float().unsqueeze(0).unsqueeze(0)

                    mapped_log_lamda = self.mapping(torch.where(lamda > 0, lamda.log(), (lamda + 1e-5).log()))
                    if self.encoding_type == "mean_encoding":
                        enc_rep_cat = torch.cat([global_mean, mapped_log_lamda], dim= 1)
                    elif self.encoding_type == "mean_min_max_encoding":
                        enc_rep_cat = torch.cat([global_mean, global_max, global_min, mapped_log_lamda], dim= 1)

                    linear_map = self.linear_mapping(enc_rep_cat)
                    predicted_dice = self.dice_mapping(linear_map)
                    predicted_jac = self.jacobian_mapping(linear_map)
                    predicted_jac = (predicted_jac * self.nfv_clapirn)if predicted_jac > 0 else torch.tensor([0.0]).float()
                    if predicted_jac < (nfv_percent/100) * self.maximum_nfv:
                        dice_average_per_image_per_lamda = pd.concat([dice_average_per_image_per_lamda, pd.DataFrame({"pair_idx": pair_idx, "moving_index": batch[4][0], "fixed_index": batch[5][0],
                                                                                                                         "predicted_dice": predicted_dice.mean().item() ,"lamda": lamda.item(), "predicted_jac":predicted_jac.item()}, index=[0])], ignore_index=True)
                        for label in labels.keys():
                            pred_avg = (predicted_dice[0][labels[label][0]-1] + predicted_dice[0][labels[label][1]-1])/2

                            dice_average_per_label_per_lamda = pd.concat([dice_average_per_label_per_lamda, pd.DataFrame({"pair_idx": pair_idx, "moving_index": batch[4], "fixed_index": batch[5], 
                                                                                                                          "predicted_dice": pred_avg.item(), "lamda": lamda.item(), "label": label, "predicted_jac":predicted_jac.item()}, index=[0])], ignore_index=True)      
                        
                return dice_average_per_image_per_lamda, dice_average_per_label_per_lamda

    def test_niftyreg(self, pair_idx, batch, bending_energy, linear_elasticity, spacing, nfv_percent):
        moving_img, fixed_img, moving_label, fixed_lbl, moving_idx, fixed_idx = batch
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


        labels = {"Thalamus": lr_thalamus, "Hippocampus": lr_hippocampus, "Amygdala": lr_amygdala, "Accumbens": lr_accumbens, "Putamen": lr_putamen, "Pallidum": lr_pallidum, "Caudate": lr_caudate, "Cerebellum WM": lr_cerebellum_WM,
                   "Cerebellum Cortex": lr_cerebellum_cortex, "Cerebral Cortex": lr_cerebral_cortex, "Cerebral WM": lr_cerebral_WM, "Vessel": lr_vessel, "Lateral Ventricle": lr_lateral_ventricle}

        columns = ["pair_idx", "moving_index", "fixed_index", "predicted_dice", "be", "le","sx", "predicted_jac"]
        dice_average_per_image_per_be = pd.DataFrame(columns = columns)

        columns_label = ["pair_idx","moving_index", "fixed_index", "predicted_dice", "be", "le","sx", "label", "predicted_jac"]
        dice_average_per_label_per_be = pd.DataFrame(columns = columns_label)
        with torch.no_grad():

            if self.encoder_model == "clapirn":
                enc_rep = self.enc(moving_img, fixed_img, self.batch_size)    
            elif self.encoder_model == "symnet":
                enc_rep = self.enc(moving_img, fixed_img)                
                
            if self.encoding_type == "mean_encoding":
                global_mean = self.global_statistics(enc_rep, self.encoding_type)
            elif self.encoding_type == "mean_min_max_encoding":
                global_mean, global_max, global_min =  self.global_statistics(enc_rep, self.encoding_type)
                

            for b in bending_energy:
                for l in linear_elasticity:
                    be = torch.tensor(b, device=self.device).float().unsqueeze(0).unsqueeze(0)
                    le = torch.tensor(l, device=self.device).float().unsqueeze(0).unsqueeze(0)
                    sx = torch.tensor(spacing, device=self.device).float().unsqueeze(0).unsqueeze(0)

                    mapped_log_be = self.mapping(torch.where(be > 0, be.log(), (be + 1e-6).log()))
                    mapped_log_le = self.mapping(torch.where(le > 0, be.log(), (le + 1e-6).log()))
                    mapped_log_sx = self.mapping(sx.log())

                    if self.encoding_type == "mean_encoding":
                        enc_rep_cat = torch.cat([global_mean, mapped_log_sx, mapped_log_be, mapped_log_le], dim= 1)
                    elif self.encoding_type == "mean_min_max_encoding":
                        enc_rep_cat = torch.cat([global_mean, global_max, global_min, mapped_log_sx, mapped_log_be, mapped_log_le], dim= 1)

                    linear_map = self.linear_mapping(enc_rep_cat)

                    #dice and jacobian mapping 
                    predicted_dice = self.dice_mapping(linear_map)
                    predicted_jac = self.jacobian_mapping(linear_map)                
                    predicted_jac = (predicted_jac * self.nfv_niftyreg)if predicted_jac > 0 else torch.tensor([0.0]).float()
                    if predicted_jac < (nfv_percent/100) * self.maximum_nfv:
                        dice_average_per_image_per_be = pd.concat([dice_average_per_image_per_be, pd.DataFrame({"pair_idx": pair_idx, "moving_index": batch[4][0], "fixed_index": batch[5][0], "predicted_dice": predicted_dice.mean().item() ,"be": be.item(), 
                                                                                                                "le": le.item(), "sx": sx.item(), "predicted_jac":predicted_jac.item()}, index=[0])], ignore_index=True)
                        for label in labels.keys():
                            pred_avg = (predicted_dice[0][labels[label][0]-1] + predicted_dice[0][labels[label][1]-1])/2

                            dice_average_per_label_per_be = pd.concat([dice_average_per_label_per_be, pd.DataFrame({"pair_idx": pair_idx, "moving_index": batch[4][0], "fixed_index": batch[5][0], "predicted_dice": pred_avg.item(), "be": be.item(),
                                                                                                                    "le": le.item(),"sx": sx.item(), "label": label, "predicted_jac":predicted_jac.item()}, index=[0])], ignore_index=True)      
                        
            return dice_average_per_image_per_be, dice_average_per_label_per_be
        

    def global_statistics(self, encoded_representation, encoding_type):
        flattened_enc_rep = encoded_representation.view(encoded_representation.shape[0], encoded_representation.shape[1], -1)
        global_mean = torch.mean(flattened_enc_rep, dim=2) 
        global_max = torch.max(flattened_enc_rep, dim=2)[0] 
        global_min = torch.min(flattened_enc_rep, dim=2)[0] 

        if encoding_type == "mean_encoding":
            return global_mean
        elif encoding_type == "mean_min_max_encoding":
            return global_mean, global_max, global_min
      

class ComputeLoss(nn.Module):
    def __init__(self, task_num):
        super(ComputeLoss, self).__init__()
        self.weight_losses = 1
        self.minimum = 0
        self.maximum = 0

    def forward(self, predicted_dice, target_dice, predicted_jac, target_jac, registration_model):
        if registration_model == "clapirn":
            self.maximum = 211473
        elif registration_model == "niftyreg":
            # self.maximum = 23189
            self.maximum = 194404
        
        print("in compute loss: ", predicted_dice.shape, target_dice.shape, predicted_jac.shape, target_jac.shape)
        target_jac = (target_jac - self.minimum) / (self.maximum - self.minimum)
        dice_loss = nn.functional.mse_loss(predicted_dice, target_dice)
        jac_loss = nn.functional.mse_loss(predicted_jac, target_jac)
        total_loss = dice_loss + self.weight_losses * jac_loss
   
        return total_loss, dice_loss, jac_loss 


def hyper_predict(in_features, mapping_features, out_features):
    #mainly claprin
    lm_out_one = 64
    lm_out_two = 64
    lm_out_four = 32


    linear_mapping = nn.Sequential(
        nn.Linear(in_features, lm_out_one),
        nn.LeakyReLU(),
    )

    dice_mapping = nn.Sequential(
        nn.Linear(lm_out_one, lm_out_two),
        nn.LeakyReLU(),
        nn.Linear(lm_out_two, out_features -1),
    )

    jacobian_mapping = nn.Sequential(
        nn.Linear(lm_out_one,lm_out_four),
        nn.LeakyReLU(),
        nn.Linear(lm_out_four, 1),

    )

    mapping = nn.Sequential(
    nn.Linear(1, mapping_features),
    nn.LeakyReLU(),
    nn.Linear(mapping_features, mapping_features),
    )

    return mapping, linear_mapping, dice_mapping, jacobian_mapping



# def hyper_predict(in_features, mapping_features, out_features):

#     #niftyreg ; total_val_loss=0.00691-epoch=17-logger-mean_encoding_2HLnfv_nfv_194404_no_loss_weight.ckpt
#     lm_out_one = 64
#     lm_out_two = 64
#     lm_out_four = 32


#     linear_mapping = nn.Sequential(
#         nn.Linear(in_features, lm_out_one),
#         nn.LeakyReLU(),
#     )

#     dice_mapping = nn.Sequential(
#         nn.Linear(lm_out_one, lm_out_two),
#         nn.LeakyReLU(),
#         nn.Linear(lm_out_two, out_features -1),
#     )

#     jacobian_mapping = nn.Sequential(
#         nn.Linear(lm_out_one,lm_out_four),
#         nn.LeakyReLU(),
#         nn.Linear(lm_out_four,8),
#         nn.LeakyReLU(),
#         nn.Linear(8,1),
        
#     )

#     mapping = nn.Sequential(
#     nn.Linear(1, mapping_features),
#     nn.LeakyReLU(),
#     nn.Linear(mapping_features, mapping_features),
#     )

#     return mapping, linear_mapping, dice_mapping, jacobian_mapping

# def hyper_predict(in_features, mapping_features, out_features):

#     #niftyreg; total_val_loss=0.06348-epoch=12-logger-mean_encoding_3HLnfv_nfv_23189_no_loss_weight_datasize0.25.ckpt
#     lm_out_one = 64
#     lm_out_two = 64
#     lm_out_four = 64
#     lm_out_five = 32



#     linear_mapping = nn.Sequential(
#         nn.Linear(in_features, lm_out_one),
#         nn.LeakyReLU(),
#     )

#     dice_mapping = nn.Sequential(
#         nn.Linear(lm_out_one, lm_out_two),
#         nn.LeakyReLU(),
#         nn.Linear(lm_out_two, out_features -1),
#     )

#     jacobian_mapping = nn.Sequential(
#         nn.Linear(lm_out_one,lm_out_four),
#         nn.LeakyReLU(),
#         nn.Linear(lm_out_four,lm_out_five),
#         nn.LeakyReLU(),
#         nn.Linear(lm_out_five, 8),
#         nn.LeakyReLU(),
#         nn.Linear(8,1),
        
#     )

#     mapping = nn.Sequential(
#     nn.Linear(1, mapping_features),
#     nn.LeakyReLU(),
#     nn.Linear(mapping_features, mapping_features),
#     )

#     return mapping, linear_mapping, dice_mapping, jacobian_mapping

