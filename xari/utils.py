# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 12:58:18 2023

@author: talha
"""

import numpy as np
import PIL.Image
from matplotlib import pylab as P
import saliency.core as saliency
import torch 
import torch.nn as nn
from torchvision import models

def  MaxViT_t(pretrained=True, num_classes=1000):

    if pretrained:
        maxvit_t = models.maxvit_t(weights='DEFAULT')
    else:
        maxvit_t = models.maxvit_t(weights=None)
    maxvit_t.classifier[5] = nn.Linear(maxvit_t.classifier[5].in_features, num_classes)

    return maxvit_t

def  EfficientNet_v2_m(pretrained=True, num_classes=1000):
    if pretrained:
        efficientnet_v2_m = models.efficientnet_v2_m(weights='DEFAULT')
    else:
        efficientnet_v2_m = models.efficientnet_v2_m(weights=None)
    efficientnet_v2_m.classifier[1] = nn.Linear(efficientnet_v2_m.classifier[1].in_features, num_classes)

    return efficientnet_v2_m

# Boilerplate methods.
def ShowImage(im, title='', ax=None):
    if ax is None:
        P.figure()
    P.axis('off')
    P.imshow(im)
    P.title(title)

def ShowGrayscaleImage(im, title='', ax=None):
    if ax is None:
        P.figure()
    P.axis('off')
    P.imshow(im, cmap=P.cm.gray, vmin=0, vmax=1)
    P.title(title)

def ShowHeatMap(im, title, ax=None):
    if ax is None:
        P.figure()
    P.axis('off')
    P.imshow(im, cmap='inferno')
    P.title(title)

def LoadImage(file_path):
    im = PIL.Image.open(file_path)
    im = im.resize((224, 224))
    im = np.asarray(im)
    return im
from torchvision import transforms
transformer = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
def PreprocessImages(images):
    # assumes input is 4-D, with range [0,255]
    #
    # torchvision have color channel as first dimension
    # with normalization relative to mean/std of ImageNet:
    #    https://pytorch.org/vision/stable/models.html
    images = np.array(images)
    images = images/255
    images = np.transpose(images, (0,3,1,2))
    images = torch.tensor(images, dtype=torch.float32)
    images = transformer.forward(images)
    return images.requires_grad_(True)

def load_checkpoint(model, pretrained_path=None):
    if pretrained_path is not None:
        chkpt = torch.load(pretrained_path,
                            map_location='cpu')
        try:
            # load pretrained
            try:
                pretrained_dict = chkpt['model_state_dict']
                print("[INFO] Loaded Model checkpoint:")
                for key, value in chkpt.items():
                    if key != 'model_state_dict':
                        if key !='optimizer_state_dict':
                            print(f"{key}={value}", end='  ')
                print()
            except KeyError:
                pretrained_dict = chkpt
            # load model state dict
            state = model.state_dict()
            # loop over both dicts and make a new dict where name and the shape of new state match
            # with the pretrained state dict.
            matched, unmatched = [], []
            new_dict = {}
            for i, j in zip(pretrained_dict.items(), state.items()):
                pk, pv = i # pretrained state dictionary
                nk, nv = j # new state dictionary
                # if name and weight shape are same
                if pk.strip('module.') == nk.strip('module.') and pv.shape == nv.shape:
                    new_dict[nk] = pv
                    matched.append(pk)
                else:
                    unmatched.append(pk)

            state.update(new_dict)
            model.load_state_dict(state)
            print('[INFO] Pre-trained state loaded successfully...', 'blue')
            print(f'Mathed kyes: {len(matched)}, Unmatched Keys: {len(unmatched)}')
        except:
            print(f'ERROR in pretrained_dict @ {pretrained_path}', 'red')
    else:
        print('Enter pretrained_dict path.')
    return matched, unmatched

#%%

