from hyperpredict import hyper_predict, HyperPredictLightningModule
from utils import SetParams
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from setting import arg
from datetime import datetime

registration_model = "niftyreg"
encoder_model = "symnet"
datapath = "data/oasis/"
pretrained_path = "models/pretrained_models/"
encoder_path = "symnet.pth"
# registration_path = "clapirn.pth"
start_channel = 4
range_flow = 0.4
imgshape = (160, 192, 224)
imgshape_4 = (160 / 4, 192 / 4, 224 / 4)
imgshape_2 = (160 / 2, 192 / 2, 224 / 2)
train_batch_size =1
validation_batch_size = 1
test_batch_size = 1
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
batch_size = 1
maximum_nfv = 160 * 192 * 224
args = arg()

encoder = SetParams.set_encoder(encoder_model,  pretrained_path + encoder_path, start_channel, imgshape, imgshape_2, imgshape_4, range_flow)
training_generator, validation_generator, test_generator, testing_sub = SetParams.set_oasis_data(datapath, train_batch_size, validation_batch_size, test_batch_size, encoder_model)

print("data loaded", len(training_generator), len(validation_generator), len(test_generator), len(testing_sub))

if args.encoding_type == "mean_encoding":
    # select_in_features = {"clapirn_clapirn": 32, "clapirn_niftyreg": 48, "symnet_clapirn": 88, "symnet_niftyreg": 120}
    select_in_features = {"clapirn_clapirn": 32, "clapirn_niftyreg": 48, "symnet_clapirn": 88, "symnet_niftyreg": 152}

elif args.encoding_type == "mean_min_max_encoding":
    select_in_features = {"clapirn_clapirn": 64, "clapirn_niftyreg": 80, "symnet_clapirn": 200, "symnet_niftyreg": 232}
    
in_features = select_in_features[encoder_model + "_" + registration_model]
out_features = 36
mapping_features = 16 if encoder_model == "clapirn" else 32
model = HyperPredictLightningModule(hyper_predict(in_features, mapping_features, out_features),  registration_model, encoder_model, imgshape, encoder, batch_size, args.encoding_type)
# model.load_state_dict(torch.load("models/checkpoints/symnet_niftyreg/total_val_loss=0.00691-epoch=17-logger-mean_encoding_2HLnfv_nfv_194404_no_loss_weight.ckpt")["state_dict"])
# model.load_state_dict(torch.load("models/checkpoints/symnet_niftyreg/total_val_loss=0.01015-epoch=43-logger-mean_encoding_main_hyperpredict_network_nfv_194404_be_le_sx_datasize0.25.ckpt")["state_dict"])
# model.load_state_dict(torch.load("models/checkpoints/symnet_niftyreg/total_val_loss=0.01249-epoch=26-logger-training_mean_encoding_main_hyperpredict_network_datasize0.25_le_be_shufld_dataset_weight_2.0.ckpt")["state_dict"])
model.load_state_dict(torch.load("models/checkpoints/symnet_niftyreg/total_val_loss=0.01139-epoch=29-logger-mean_encoding_main_hyperpredict_network_datasize0.25_linear_elastisity.ckpt")["state_dict"])

# be = np.linspace(-11, 0, 100)
# be = np.exp(be)
# le = np.linspace(-11, 0, 100)
# le = np.exp(le)

# be = [0.0200, 0.0164, 0.0194, 0.0067, 0.0215]
# le = [0.0085, 0.0067, 0.0080, 0.0091, 0.0102, 0.0076]

# be = [0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.5, 1.0]

#important
# be = [0.0001, 0.006, 0.02, 0.01, 0.2]
# le = [0.008, 0.006, 0.05, 0.01, 0.1, 0.5]

be = [0.001]
le = [0.0001, 0.001, 0.0075,0.05, 0.1, 0.2, 0.5, 1]


sx = 5
for params in model.parameters():
    params.requires_grad = False


model.eval()
model.to(device)
# dice per image per be
columns = ["pair_idx", "moving_index", "fixed_index", "predicted_dice", "be", "le", "sx", "predicted_jac"]

dice_average_per_image_per_be = pd.DataFrame(columns = columns)

#dataframe for dice per label 
columns_label = ["pair_idx","moving_index", "fixed_index", "predicted_dice","be", "le", "sx", "label", "predicted_jac"]

dice_average_per_label_per_be = pd.DataFrame(columns = columns_label)
count  = 1
print("len test generator", len(testing_sub))
start = datetime.now()
with torch.no_grad():
    for pair_idx, data in enumerate(testing_sub):
        # if pair_idx > 0:
        #     break
        pred = []
        tar = []    
        data[0:4] = [d.to(device) for d in data[0:4]]
        # per_image, per_label = model.test_niftyreg(pair_idx, data, be,le, sx, args.nfv_percent)
        per_image, per_label = model.test_niftyreg(pair_idx, data, be, le, sx, args.nfv_percent)
  
        
        # per_image.to_csv("results/symnet_niftyreg/mean_encoding_2HLnfv_nfv_194404_no_loss_weight_image.csv", mode='a', header=False, index=False)
        # per_label.to_csv("results/symnet_niftyreg/mean_encoding_2HLnfv_nfv_194404_no_loss_weight_label.csv", mode='a', header=False, index=False)
        per_image.to_csv("results/symnet_niftyreg/sampled_lamda/sampled_lamda_values_images_le.csv", mode='a', header=True if count == 1 else False, index=False)
        per_label.to_csv("results/symnet_niftyreg/sampled_lamda/sampled_lamda_values_labels_le.csv", mode='a', header=True if count == 1 else False, index=False)
        
        
        print(count)
        count += 1

#do below to save optimal target valuestest
# image_data = pd.read_csv("results/symnet_niftyreg/mean_encoding_main_hyperpredict_network_nfv_194404_be_le_sx_datasize0.25_img.csv")
# image_data = image_data.groupby(["pair_idx"], as_index=False).apply(lambda x: x[x.predicted_dice == x.predicted_dice.max()])
# image_data.to_csv("results/symnet_niftyreg/mean_encoding_main_hyperpredict_network_nfv_194404_be_le_sx_datasize0.25_img.csv")

# label_data = pd.read_csv("results/symnet_niftyreg/mean_encoding_main_hyperpredict_network_nfv_194404_be_le_sx_datasize0.25_label.csv")
# label_data = label_data.groupby(["pair_idx", "label"], as_index=False).apply(lambda x: x[x.predicted_dice == x.predicted_dice.max()])
# label_data.to_csv("results/symnet_niftyreg/mean_encoding_main_hyperpredict_network_nfv_194404_be_le_sx_datasize0.25_label.csv")
print("time taken", datetime.now() - start)
        

