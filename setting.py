from argparse import ArgumentParser

def arg():
    parser = ArgumentParser()

    parser.add_argument("--datapath", type=str,
                dest="datapath",
                default='data/oasis/',
                help="data path for images - trained on oasis or abdominal CT")

    parser.add_argument("--encoder_model", type=str, dest="encoder_model", 
                default="symnet", help="define the encoder to run")

    parser.add_argument("--registration_model", type=str, dest="registration_model", 
                default="clapirn", help="define the registration model to run")
    parser.add_argument("--encoding_type", type=str, dest="encoding_type", 
                default="mean_encoding", help="type of encoding, either mean_encoding or mean_min_max_encoding")

    parser.add_argument("--pretrained_path", type=str, dest="pretrained_path", 
                default="models/pretrained_models/", help="path to pretrained models")
    
    parser.add_argument("--run_type", type=str, dest="run_type", 
                default="training", help="training/sanity_check/overfitting")
    
    parser.add_argument("--train_data_size", type=float, dest="train_data_size", 
                default=1.0, help="limit training data size")
    
    parser.add_argument("--val_data_size", type=float, dest="val_data_size", 
                default=1.0, help="limit validation data size")
    
    parser.add_argument("--logger_name", type=str, dest="logger_name", 
                default="", help="name of the logger")

    parser.add_argument('--start_channel', type = int, dest = 'start_channel', default = 4, help = "start channel for model")
    parser.add_argument('--nfv_percent', type = float, dest = 'nfv_percent', default = 0.5, help = "percentage of nfv to consider at test time")
    args = parser.parse_args()


    return args