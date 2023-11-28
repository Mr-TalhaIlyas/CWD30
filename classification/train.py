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

import imgviz, cv2, random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from termcolor import cprint
from tqdm import tqdm
from itertools import cycle, chain
mpl.rcParams['figure.dpi'] = 300


from data.dataloader import ClassifierLoader
from data.utils import collate, images_transform

from core.backbones.modelhub import ResNet101, MobileNet_v3_small, MobileNet_v3_large, ResNext101_32x8d
from core.backbones.mscan import MSCANet
from core.utils.lr_scheduler import LR_Scheduler
from core.utils.chkpt_manager import load_checkpoint, save_chkpt

import torch.nn.functional as F
from torchmetrics import Accuracy
from torchinfo import summary
from sklearn.metrics import confusion_matrix, classification_report
from tools.utils import values_fromreport
import timm
from torch.utils.data import ConcatDataset, DataLoader
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x = []
dirs = fmu.get_all_dirs(config['data']['data_dir']+'/train/images/')
for d in dirs:
    x.append(d.split('/')[-1])
x.sort()
classes = x

config['model']['encoder']['as_classifier'] = True
config['model']['encoder']['num_classes'] = len(classes)
config['model']['encoder']['img_size'] = config['data']['img_height']

print(f'Printing Configuration File:\n{30*"="}\n')
pprint.pprint(config)
print(f'{30*"*"}\n')
print(tabulate({'Classes':classes}, headers='keys', showindex="always"))

'''
For iNaturalist dataset

'''
# trian_files, val_files, dir2lbl = inat_loader(config['data']['data_dir'])
# num_classes = len(dir2lbl.keys())
# classes = num_classes

# train_data = iNatLoader(trian_files, dir2lbl,
#                         config['data']['img_height'], config['data']['img_width'],
#                         config['data']['label_smoothing'], 
#                         config['data']['Augment_data'], config['data']['Normalize_data'])
# val_data = iNatLoader(val_files, dir2lbl,
#                         config['data']['img_height'], config['data']['img_width'],
#                         config['data']['label_smoothing'], 
#                         False, config['data']['Normalize_data'])
'''
For CWD30 and other datasets

'''
train_data = ClassifierLoader(fmu.get_all_files(config['data']['data_dir']+'/train/images/'), 
                              classes, config['data']['img_height'], config['data']['img_width'],
                              config['data']['label_smoothing'], 
                              config['data']['Augment_data'], config['data']['Normalize_data'])

train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True,
                          num_workers=config['data']['num_workers'], drop_last=True, # important for adaptive augmentation to work properly.
                          collate_fn=collate, pin_memory=config['data']['pin_memory'],
                          prefetch_factor=config['data']['prefetch_factor'],
                          persistent_workers=config['data']['persistent_workers']
                          )

val_data = ClassifierLoader(fmu.get_all_files(config['data']['data_dir']+'/train/images/'), 
                              classes, config['data']['img_height'], config['data']['img_width'],
                              config['data']['label_smoothing'], 
                              False, config['data']['Normalize_data'])

val_loader = DataLoader(val_data, batch_size=config['batch_size'], shuffle=True,
                          num_workers=config['data']['num_workers'], drop_last=True, # important for adaptive augmentation to work properly.
                          collate_fn=collate, pin_memory=config['data']['pin_memory'],
                          prefetch_factor=config['data']['prefetch_factor'], 
                          persistent_workers=config['data']['persistent_workers']
                          )

# Concatenate datasets
# combined_dataset = ConcatDataset([train_data, val_data, test_data])

# # Create a single DataLoader from the combined dataset
# train_loader = DataLoader(combined_dataset, batch_size=config['batch_size'], shuffle=True,
#                              num_workers=config['data']['num_workers'], drop_last=True,
#                              collate_fn=collate, pin_memory=config['data']['pin_memory'],
#                              prefetch_factor=config['data']['prefetch_factor'],
#                              persistent_workers=config['data']['persistent_workers'])

# batch = next(iter(train_loader))
# s=255
# img_ls = []
# [img_ls.append((batch['img'][i]*s).astype(np.uint8)) for i in range(config['batch_size'])]
# plt.imshow(imgviz.tile(img_ls, shape=(2,config['batch_size']//2), border=(255,0,0)))
# print(f"class: {classes[batch['lbl'][3]]}")
#%%

# model = MobileNet_v3_small(pretrained=True, num_classes=len(classes))
model = timm.create_model('maxvit_base_tf_224.in1k',
                          pretrained=True,
                          num_classes=len(classes),  # remove classifier nn.Linear
                        )

if config['pretrained_path'] is not None:
    print("Loading pretrained model...")
    matched, unmatched = load_checkpoint(model, pretrained_path=config['pretrained_path'])
    print(unmatched)


model.to(DEVICE)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    # encoder = torch.compile(model)#, mode="max-autotune")

criterion = nn.CrossEntropyLoss(label_smoothing=config['data']['label_smoothing'])
accuracy = Accuracy(task="multiclass", num_classes=len(classes))

optim = torch.optim.AdamW([{'params': model.parameters(),
                            'lr': config['learning_rate']}],
                          weight_decay=config['WEIGHT_DECAY'])
lr_scheduler = LR_Scheduler(config['lr_schedule'], config['learning_rate'], config['epochs'],
                            iters_per_epoch=len(train_loader), warmup_epochs=config['warmup_epochs'])

# Initializing plots
if config['LOG_WANDB']:
    wandb.watch(model, log='parameters', log_freq=100)
    wandb.log({"val_acc": 0, "acc": 0,  
               "loss": 10, "learning_rate": 0}, step=0)

#%%
start_epoch = 0
epoch, best_acc, curr_vacc = 0, 0, 0
total_avg_vacc = []

for epoch in range(start_epoch, config['epochs']):
    pbar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}/{config['epochs']}")
    model.train()
    correct, total = 0, 0
    tl, ta = [], []
    for step, data_batch in enumerate(pbar):
        # prepare data
        images = images_transform(data_batch['img'])
        labels = torch.from_numpy(np.asarray(data_batch['lbl'])).to(DEVICE)
        # forward pass
        lr_scheduler(optim, step, epoch)
        optim.zero_grad()
        outputs = model(images)
        # backward pass
        loss = criterion(outputs, labels)
        if torch.cuda.device_count() > 1: # average loss across CUDA devices.
            loss = loss.mean()
        loss.backward()
        optim.step()
        # accuracy 
        acc = accuracy(outputs.argmax(1).cpu(), labels.cpu())

        tl.append(loss.item())
        ta.append(acc)
        pbar.set_description(f'Epoch {epoch+1}/{config["epochs"]} - t_loss {loss.item():.4f} - Acc {acc:.4f}')
        
    print(f'=> Average loss: {np.nanmean(tl):.4f}, Average Acc: {np.nanmean(ta):.4f}')
    
    all_preds, all_lbls = [], []
    if (epoch + 1) % 2 == 0:
        model.eval()
        correct, total = 0, 0
        va = []
        
        with torch.no_grad():
            for data_batch in tqdm(val_loader, desc='Validating'):
                images = images_transform(data_batch['img'])
                labels = torch.from_numpy(np.asarray(data_batch['lbl'])).to(DEVICE)
                outputs = model(images)
                vacc = accuracy(outputs.argmax(1).cpu(), labels.cpu()).item()
                all_preds.append(outputs.argmax(1).cpu().numpy())
                all_lbls.append(labels.cpu().numpy())
                va.append(vacc)
                break
            if config['LOG_WANDB']:
                wandb.log({"val_acc": np.nanmean(va)}, step=epoch+1)

        total_avg_vacc.append(np.nanmean(va))
        curr_vacc = np.nanmax(total_avg_vacc)
        print(f'Average Val accuracy: {np.nanmean(va):.4f}%, Best Val accuracy: {curr_vacc:.4f}%')

        all_preds = np.asarray(all_preds).reshape(-1,)
        all_lbls = np.asarray(all_lbls).reshape(-1,)
        
        matrix = confusion_matrix(all_lbls, all_preds, normalize='true')
        report = classification_report(all_lbls, all_preds,
                                       output_dict=True,
                                       zero_division=0)
        p, r, f1 = values_fromreport(report)
        cprint(f'Class Accuracies:: {matrix.diagonal()/matrix.sum(axis=1)}', 'blue')
        cprint(f'Avg Acc: {np.mean(matrix.diagonal()/matrix.sum(axis=1))}', 'yellow')
        cprint(f'Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}', 'light_magenta')
        
    if config['LOG_WANDB']:
        wandb.log({"loss": np.nanmean(tl), "acc": np.nanmean(ta),
                   "learning_rate": optim.param_groups[0]['lr'],
                   }, step=epoch+1)

    if curr_vacc > best_acc:
        best_acc = curr_vacc
        chkpt =save_chkpt(config, model, optim, epoch, loss.item(), best_acc,
                            module='clf', return_chkpt=True)

#%%
'''
Test the model
'''
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, top_k_accuracy_score

print(f'loading chkpt {chkpt}')
matched, unmatched = load_checkpoint(model, pretrained_path=chkpt)

# model = timm.create_model('resnet18.a1_in1k',
#                           pretrained=True,
#                           num_classes=len(classes),  # remove classifier nn.Linear
#                         )
# matched, unmatched = load_checkpoint(model,
#     pretrained_path="/home/user01/data/talha/CWD/chkpts/clf_deepweeds.maxvit_base_tf_224.in1k.pth")
# print(unmatched)

# model = model.to(DEVICE)
#%%
test_data = ClassifierLoader(
                            fmu.get_all_files(config['data']['data_dir']+'/train/images/'), 
                            # fmu.get_all_files('/home/user01/data/talha/CWD/datasets/clf/inat2021/filtered/'),
                            classes, config['data']['img_height'], config['data']['img_width'],
                            config['data']['label_smoothing'], 
                            False, config['data']['Normalize_data'])

test_loader = DataLoader(test_data, batch_size=32, shuffle=True, collate_fn=collate)

all_true_labels = []
all_predictions = []
all_scores = []
model.eval()
with torch.no_grad():
    per_class_correct = np.zeros(len(classes), dtype=int)
    per_class_total = np.zeros(len(classes), dtype=int)

    for data_batch in tqdm(test_loader, desc='Testing'):
        images = images_transform(data_batch['img'])
        labels = torch.from_numpy(np.asarray(data_batch['lbl'])).to(DEVICE)
        try:
            outputs = model(images)
            predictions = outputs.argmax(1).cpu()
            scores = outputs.cpu().numpy()  # Convert to NumPy array
            
            all_true_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.numpy())
            all_scores.extend(scores)
        except AttributeError:
            print('AttributeError Occured')
            continue

avg_acc = accuracy_score(all_true_labels, all_predictions)

wf1_score = f1_score(all_true_labels, all_predictions, average='weighted')

matrix = confusion_matrix(all_true_labels, all_predictions)
per_class_acc = matrix.diagonal()/(matrix.sum(axis=1) + 1e-7)
top_k_acc = top_k_accuracy_score(all_true_labels, all_scores, k=5, labels=np.arange(len(classes)))

print(f'Top-5 Accuracy: {top_k_acc}')
print(f"Per-class accuracy: {per_class_acc}")
# print(f"Per-Class Mean: {np.mean(matrix.diagonal()/matrix.sum(axis=1)):.4f}")
print(f'Average accuracy: {avg_acc:.4f}')
print(f"Weighted F1 score: {wf1_score:.4f}")

# %%
if config['LOG_WANDB']:
    wandb.run.finish()
#%%