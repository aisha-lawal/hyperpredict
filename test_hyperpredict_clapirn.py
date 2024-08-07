from hyperpredict import hyper_predict, HyperPredictLightningModule
from utils import SetParams
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from setting import arg

args = arg()
registration_model = "clapirn"
encoder_model = "symnet"
datapath = "data/oasis/"
pretrained_path = "models/pretrained_models/"
encoder_path = "symnet.pth"
registration_path = "clapirn.pth"
start_channel = 4
range_flow = 0.4
imgshape = (160, 192, 224)
imgshape_4 = (160 / 4, 192 / 4, 224 / 4)
imgshape_2 = (160 / 2, 192 / 2, 224 / 2)
train_batch_size =1
validation_batch_size = 1
test_batch_size = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 1
maximum_nfv = 160 * 192 * 224

encoder = SetParams.set_encoder(encoder_model,  pretrained_path + encoder_path, start_channel, imgshape, imgshape_2, imgshape_4, range_flow)
registration = SetParams.set_registration(registration_model, start_channel, imgshape, imgshape_2, imgshape_4, range_flow, pretrained_path +registration_path) #needed for testing
training_generator, validation_generator, test_generator, testing_sub = SetParams.set_oasis_data(datapath, train_batch_size, validation_batch_size, test_batch_size, encoder_model)

if args.encoding_type == "mean_encoding":
    select_in_features = {"clapirn_clapirn": 32, "clapirn_niftyreg": 48, "symnet_clapirn": 88, "symnet_niftyreg": 152}
elif args.encoding_type == "mean_min_max_encoding":
    select_in_features = {"clapirn_clapirn": 64, "clapirn_niftyreg": 80, "symnet_clapirn": 200, "symnet_niftyreg": 232}
    


print("data loaded", len(training_generator), len(validation_generator), len(test_generator), len(testing_sub))
in_features = select_in_features[encoder_model + "_" + registration_model]
out_features = 36
mapping_features = 16 if encoder_model == "clapirn" else 32
model = HyperPredictLightningModule(hyper_predict(in_features, mapping_features, out_features),  registration_model, encoder_model, imgshape, encoder, batch_size,args.encoding_type)
model.load_state_dict(torch.load("models/checkpoints/symnet_clapirn/main_checkpoint.ckpt")["state_dict"])


lam = np.linspace(-10, 0, 200)
lam = np.exp(lam)
# lam = np.array([0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.5, 1.0])
for params in model.parameters():
    params.requires_grad = False

model.eval()
model.to(device)
columns = ["pair_idx", "moving_index", "fixed_index", "predicted_dice", "lamda", "predicted_jac"]
dice_average_per_image_per_lamda = pd.DataFrame(columns = columns)
columns_label = ["pair_idx","moving_index", "fixed_index", "predicted_dice","lamda", "label", "predicted_jac"]



dice_average_per_label_per_lamda = pd.DataFrame(columns = columns_label)
count  = 1
print("len of test generator", len(testing_sub))
with torch.no_grad():
    for pair_idx, data in enumerate(testing_sub):
        # if pair_idx > 0:
        #     break
        
        pred = []
        tar = []    
        data[0:4] = [d.to(device) for d in data[0:4]]
        per_image, per_label = model.test_clapirn(pair_idx, data, lam, args.nfv_percent)
        
        dice_average_per_image_per_lamda = pd.concat([dice_average_per_image_per_lamda, per_image])

        dice_average_per_label_per_lamda = pd.concat([dice_average_per_label_per_lamda, per_label])
        
        # per_image.to_csv("results/symnet_clapirn/mean_encoding_0.25%_data_values_images.csv", mode='a', header=True if count ==1 else False, index=False)
        # per_label.to_csv("results/symnet_clapirn/mean_encoding_0.25%_data_values_labels.csv", mode='a', header=True if count == 1 else False, index=False)
        per_label.to_csv("results/symnet_clapirn/final_result/final_label_no_optim.csv", mode='a', header=True if count ==1 else False, index=False)
        per_image.to_csv("results/symnet_clapirn/final_result/final_image_no_optim.csv", mode='a', header=True if count ==1 else False, index=False)
        # per_label.to_csv("results/symnet_clapirn/tester.csv", mode='a', header=True if count == 1 else False, index=False)
        
        print(count)
        count += 1


