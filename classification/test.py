#%%
import os, pprint
# os.chdir(os.path.dirname(__file__))
os.chdir("/home/user01/data/talha/CWD/scripts/")

from configs.config_clf import config

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = config['gpus_to_use'];

if config['LOG_WANDB']:
    import wandb
    # from datetime import datetime
    # my_id = datetime.now().strftime("%Y%m%d%H%M")
    wandb.init(dir=config['log_directory'],
               project=config['project_name'], name=config['experiment_name'],
            #    resume='allow', id=my_id, # this one introduces werid behaviour in the app
               config_include_keys=config.keys(), config=config)
    # print(f'WANDB config ID : {my_id}')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from fmutils import fmutils as fmu
from tabulate import tabulate

from pathlib import Path
import imgviz, cv2, random, glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from termcolor import cprint
from tqdm import tqdm
from itertools import cycle, chain
mpl.rcParams['figure.dpi'] = 300


from data.dataloader import iNatLoader, inat_loader
from data.utils import collate, images_transform

from core.backbones.modelhub import ResNet101, EfficientNet_v2_m, MobileNet_v3_large, ResNext101_32x8d
from core.backbones.modelhub import Swin_b, ViT_l_32, MaxViT_t
from core.backbones.mit import MixVisionTransformer
from core.utils.metrics import ConfusionMatrix
from core.utils.lr_scheduler import LR_Scheduler
from core.utils.chkpt_manager import load_checkpoint, save_chkpt

import torch.nn.functional as F
from torchmetrics import Accuracy
from torchinfo import summary
from sklearn.metrics import confusion_matrix, classification_report
from tools.utils import values_fromreport
import timm
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, top_k_accuracy_score

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



model = ResNet101(pretrained=True, num_classes=num_classes)
# print(f'loading chkpt {chkpt}')
# matched, unmatched = load_checkpoint(model, pretrained_path=chkpt)

matched, unmatched = load_checkpoint(model,
    pretrained_path="/home/user01/data/talha/CWD/chkpts/clf_inat.resnet101.in1k.pth")
print(unmatched)

model = model.to(DEVICE)

cwd_test_files = fmu.get_all_files('/home/user01/data/talha/CWD/cwd/')
print(len(cwd_test_files))
test_data = iNatLoader(cwd_test_files, dir2lbl,
                        config['data']['img_height'], config['data']['img_width'],
                        config['data']['label_smoothing'], 
                        False, config['data']['Normalize_data'])

test_loader = DataLoader(test_data, batch_size=128, shuffle=True, collate_fn=collate)




all_true_labels = []
all_predictions = []
all_scores = []
model.eval()
with torch.no_grad():
    per_class_correct = np.zeros(num_classes, dtype=int)
    per_class_total = np.zeros(num_classes, dtype=int)

    for data_batch in tqdm(test_loader, desc='Testing'):
        images = images_transform(data_batch['img'])
        labels = torch.from_numpy(np.asarray(data_batch['lbl'])).to(DEVICE)
        try:
            outputs = model(images)
            predictions = outputs.argmax(1).cpu()
            scores = outputs.cpu().numpy()

            all_true_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.numpy())
            all_scores.extend(scores)
        except AttributeError:
            print('AttributeError Occured')
            continue
        # break
avg_acc = accuracy_score(all_true_labels, all_predictions)

wf1_score = f1_score(all_true_labels, all_predictions,
                     labels=np.arange(num_classes),average='weighted')

matrix = confusion_matrix(all_true_labels, all_predictions,
                          labels=np.arange(num_classes))
per_class_acc = matrix.diagonal()/matrix.sum(axis=1)

top_k_acc = top_k_accuracy_score(all_true_labels, all_scores, k=5, labels=np.arange(num_classes))

print(f'Top-5 Accuracy: {top_k_acc}')
print(f"Per-class accuracy: {per_class_acc}")
# print(f"Per-Class Mean: {np.mean(matrix.diagonal()/matrix.sum(axis=1)):.4f}")
print(f'Average accuracy: {avg_acc:.4f}')
print(f"Weighted F1 score: {wf1_score:.4f}")

# %%
