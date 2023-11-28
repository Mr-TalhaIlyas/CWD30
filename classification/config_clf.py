from configs import *
import yaml

with open('configs/data_specs.yaml') as fh:
    data_specs = yaml.load(fh, Loader=yaml.FullLoader)
    
config = dict(
                gpus_to_use = '1',
                DPI = 300,
                LOG_WANDB= False,
                BENCHMARK= False,
                DEBUG = False,

                project_name= 'TIP_Revision',
                experiment_name= 'plantseedling.maxvit_base_tf_224.cwd',

                log_directory= "/home/user01/data/talha/CWD/logs/",
                checkpoint_path= "/home/user01/data/talha/CWD/chkpts/",
                pretrained_path= "/home/user01/data/talha/CWD/chkpts/clf_cwd.maxvit_base_tf_224.in1k.pth",
                # pretrained_path= None,
                # training settings
                batch_size= 24,#96
                WEIGHT_DECAY= 0.00005,
                # AUX_LOSS_Weights= 0.4,

                # Regularization SD 0.5 LS 1e-2
                stochastic_drop_path= 5e-1,
                SD_mode= 'batch',
                layer_scaling_val= 1e-5,

                # learning rate
                learning_rate= 6e-05,
                lr_schedule= 'cos',
                epochs= 30,
                start_epoch= 0,
                warmup_epochs= 1,
                # one of 'batch_norm' or 'sync_bn' or 'layer_norm'
                norm_typ= 'sync_bn',
                BN_MOM= 0.1,
                SyncBN_MOM= 0.1,

                model = config_model.hrda,

                data = dict(
                            data_dir = "/home/user01/data/talha/CWD/datasets/clf/ip102/", 
                            # training time settings
                            img_height= 224,
                            img_width= 224,
                            input_channels= 3,
                            label_smoothing= 0.15,
                            # only for training data
                            Augment_data= True,
                            # Augmentation Prbabilities should be same legth
                            step_epoch=    [0, 10],
                            geometric_aug = [0.3, 0.3],
                            noise_aug =     [0.3, 0.3],

                            Normalize_data = True,
                            Shuffle_data = True,

                            pin_memory=  True,
                            num_workers= 1,
                            prefetch_factor= 2,
                            persistent_workers= True,
                            data_specs = data_specs,
                            ),
                )
