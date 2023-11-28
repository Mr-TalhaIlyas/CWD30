
#%%
import os

# os.chdir(os.path.dirname(__file__))
os.chdir("/home/user01/data/talha/CWD/scripts/")

from configs.config_clf import config

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "0";

import numpy as np
import torch
import torch.nn as nn

from core.backbones.modelhub import ResNet101, MaxViT_t
from core.backbones.mit import MixVisionTransformer
from core.utils.chkpt_manager import load_checkpoint, save_chkpt
from data.dataloader import ClassifierLoader
from data.utils import collate, images_transform
from fmutils import fmutils as fmu
from tabulate import tabulate
from tools.utils import get_model_embeddings, plot_tsne, plot_tsne_3d
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import random

x = []
dirs = fmu.get_all_dirs(config['data']['data_dir']+'/test/images/')
for d in dirs:
    x.append(d.split('/')[-1])
x.sort()
classes = x

config['model']['encoder']['as_classifier'] = True
config['model']['encoder']['num_classes'] = len(classes)
config['model']['encoder']['img_size'] = config['data']['img_height']

data_files = fmu.get_all_files(config['data']['data_dir']+'/test/images/')
random.shuffle(data_files)

test_data = ClassifierLoader(data_files[0:20000], 
                              classes, config['data']['img_height'], config['data']['img_width'],
                              config['data']['label_smoothing'], 
                              False, config['data']['Normalize_data'])

test_loader = DataLoader(test_data, batch_size=32, shuffle=True, collate_fn=collate)

batch = next(iter(test_loader))


config['model']['encoder']
# model = MixVisionTransformer(**config['model']['encoder'])
model = ResNet101(pretrained=False, num_classes=len(classes))

matched, unmatched = load_checkpoint(model,
    pretrained_path="/home/user01/data/talha/CWD/clf_chkpt/clf_ip102_resnet_cwd30.pth")
print(unmatched)


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(DEVICE)




# %%
embeddings, labels = get_model_embeddings(model, test_loader, 'avgpool')

print(embeddings.shape, labels.shape)

cls_names = [classes[i] for i in labels]

#%%
ranges = [14, 13, 9, 8, 13, 16, 19, 10]  # Number of labels in each super-class

# Initialize an empty array for super-class labels
sup_lbls = np.zeros_like(labels)

# Current index in the ranges array
current_index = 0
for i, size in enumerate(ranges):
    sup_lbls[labels >= current_index] = i
    current_index += size

# Verify the remapping
print("Original Labels:", labels[:20])
print("Super-class Labels:", sup_lbls[:20])
# %%
# Reduce the dimensionality of the embeddings using t-SNE
tsne_embeddings = TSNE(n_components=2, random_state=43,
                       perplexity=45, early_exaggeration=100.0).fit_transform(embeddings.squeeze())
plot_tsne(tsne_embeddings, sup_lbls)




