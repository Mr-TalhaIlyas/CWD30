# CWD30 based Crop-Weed Semantic Segmentation Models

#### This code base is built on top of [SMP](https://github.com/qubvel/segmentation_models.pytorch) 


### Pretrained Weights


### Training

In `config.yaml` file set the following values as your requirement.

The models accepted are 

* [segnext]()
* [deeplabv3plus]()
* [unet]()
* [ocr]()

Other models and backbones can also be used as SMP has a lot of ther options too.

```
selected_model: 'unet'

encoder_chkpt:  '../chkpts/model.cwd.resnet101.in1k.
# OR "imagenet" 

# Data loader parameters

data_dir: "/home/user01/data/talha/CWD/datasets/beanweed/"
# add sub dirs inside of train/val and test
sub_directories: ['imgs/', 'lbls/']
```
### Train

After setting simply run `python main.py`

if you have `wandb` installed then also set `LOG_WANDB: True` in yaml file.

### Troubleshoot SMP

#### Chanel log from original SMP repo 
Just install the SMP following the origianl documnetaiton provided by repo and make following changes if encounter any problem.

[Pull Request](https://github.com/qubvel/segmentation_models.pytorch/pull/824)
