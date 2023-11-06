import torch
from utils import generate_grid_unit, SetParams, niftyreg, compute_metric, MaskedAutoEncoder
import datetime
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.trainer import Trainer
from datetime import datetime
from hyperpredict import hyper_predict, HyperPredictLightningModule
from setting import arg




def main():


    if args.encoder_model == "clapirn":
        encoder_path = "clapirn.pth"
    
    elif args.encoder_model == "symnet":
        encoder_path = "symnet.pth"

    encoder = SetParams.set_encoder(args.encoder_model,  args.pretrained_path + encoder_path, args.start_channel, imgshape, imgshape_2, imgshape_4, range_flow)
    training_generator, validation_generator, test_generator = SetParams.set_oasis_data_registered(args.datapath, batch_size, batch_size, batch_size, args.encoder_model, args.registration_model)


    print("trainiable prameters in encoder", sum(p.numel() for p in encoder.parameters() if p.requires_grad == True))
    print("Train, validation, testing: ", len(training_generator), len(validation_generator), len(test_generator))
    print("{0} as encoder, {1} as registration, {2} as pretrained encoder".format(args.encoder_model, args.registration_model, encoder_path))


   
    model_checkpoint = ModelCheckpoint(
        monitor= "total_val_loss",
        filename= "{total_val_loss:.5f}-{epoch:02d}-logger-"+ args.logger_name,
        dirpath= "models/checkpoints/"+ args.encoder_model + "_" + args.registration_model,
        save_last= False,
        save_top_k = 1,
        mode= "min",
    )
    early_stopping = EarlyStopping(
        monitor= "total_val_loss",
        patience= 8,
        mode= "min",
    )


    logger = WandbLogger(
    project= "HyperPredict_"+ args.encoder_model + "_" + args.registration_model,
    name = args.run_type + "_" + args.logger_name,
    log_model= all,
    )

    if args.encoding_type == "mean_encoding":
        select_in_features = {"clapirn_clapirn": 32, "clapirn_niftyreg": 48, "symnet_clapirn": 88, "symnet_niftyreg": 120}
    elif args.encoding_type == "mean_min_max_encoding":
        select_in_features = {"clapirn_clapirn": 64, "clapirn_niftyreg": 80, "symnet_clapirn": 200, "symnet_niftyreg": 232}
    
    in_features = select_in_features[args.encoder_model + "_" + args.registration_model]
    out_features = 36
    mapping_features = 16 if args.encoder_model == "clapirn" else 32

    pl_model = HyperPredictLightningModule(hyper_predict(in_features, mapping_features, out_features),  args.registration_model, args.encoder_model, imgshape, encoder, batch_size, args.encoding_type)

    if args.run_type == "sanity_check":
        trainer = Trainer(fast_dev_run=True,  devices = 1)

    elif args.run_type == "overfitting":
        trainer = Trainer(fast_dev_run=False,overfit_batches = 7,limit_val_batches=1, max_epochs = 25, log_every_n_steps = 2, devices= 1, logger = logger)
        
    elif args.run_type == "training":
        trainer = Trainer(fast_dev_run=False, max_epochs= 40, log_every_n_steps=2, devices= 1,callbacks=[model_checkpoint, early_stopping], logger = logger, precision=32, limit_train_batches= args.data_size)
    
    trainer.fit(model = pl_model, train_dataloaders = training_generator, val_dataloaders= validation_generator)



if __name__ == "__main__":


    imgshape = (160, 192, 224)
    imgshape_4 = (160 / 4, 192 / 4, 224 / 4)
    imgshape_2 = (160 / 2, 192 / 2, 224 / 2) 
    model_name = "hyperpredict"

    batch_size = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = arg()
    range_flow = 0.4
    torch.set_float32_matmul_precision('high')
    start_t = datetime.now()
    main()
  
    end_t = datetime.now()
    total_t = end_t - start_t
    print("Time: ", total_t.total_seconds())

