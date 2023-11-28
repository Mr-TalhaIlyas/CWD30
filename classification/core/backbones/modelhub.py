#%%
import torch 
import torch.nn as nn
from torchvision import models
from core.utils.chkpt_manager import load_checkpoint
import timm

timm_record = [
                #'''
                # Conv.
                #'''
                'resnet50.a1_in1k', 'resnet18.a1_in1k', 
                'mobilenetv3_small_100', 'mobilenetv3_large_100',
               'efficientnet_b0', 'efficientnet_b3', 'efficientnet_b0'
               'convnext_large.fb_in1k', 'convnext_base.fb_in1k', 'convnext_tiny.fb_in1k',
               #'''
                # Tran.
                #
               'tiny_vit_21m_224.in1k', 'vit_base_patch16_224.augreg2_in21k_ft_in1k', 
               'cait_xxs36_224.fb_dist_in1k', 'cait_s24_224.fb_dist_in1k',
               'swin_s3_tiny_224.ms_in1k', 'swin_s3_base_224.ms_in1k', 'swin_large_patch4_window7_224.ms_in22k_ft_in1k',
               #'''
                # Conv. + Tran.
                #
               'maxvit_small_tf_224.in1k', 'maxvit_base_tf_224.in1k', 
               'coatnet_1_rw_224.sw_in1k', 'coatnet_3_rw_224.sw_in12k',
               'efficientformer_l1.snap_dist_in1k', 'efficientformer_l3.snap_dist_in1k', 'efficientformer_l7.snap_dist_in1k'
                #'''
                # Notes
                #
               '<===modile net functions not in timm===>', 
               '<===for resnet 101 use pytorch or timm can be used if wights can be loaded===>',
                '{lets not use it as input is 384} cait_m36_384.fb_dist_in1k',
               ]

def  ResNet101(pretrained=True, num_classes=1000):
    if pretrained:
        resnet101 = models.resnet101(weights='DEFAULT')
    else:
        resnet101 = models.resnet101(weights=None)
    resnet101.fc = nn.Linear(resnet101.fc.in_features, num_classes)
    
    return resnet101

def  MobileNet_v3_large(pretrained=True, num_classes=1000):
    if pretrained:
        mobilenet_v3_large = models.mobilenet_v3_large(weights='DEFAULT')
    else:
        mobilenet_v3_large = models.mobilenet_v3_large(weights=None)
    mobilenet_v3_large.classifier[3] = nn.Linear(mobilenet_v3_large.classifier[3].in_features, num_classes)

    return mobilenet_v3_large

def  MobileNet_v3_small(pretrained=True, num_classes=1000):
    if pretrained:
        mobilenet_v3_small = models.mobilenet_v3_small(weights='DEFAULT')
    else:
        mobilenet_v3_small = models.mobilenet_v3_small(weights=None)
    mobilenet_v3_small.classifier[3] = nn.Linear(mobilenet_v3_small.classifier[3].in_features, num_classes)

    return mobilenet_v3_small





def  EfficientNet_v2_m(pretrained=True, num_classes=1000):
    if pretrained:
        efficientnet_v2_m = models.efficientnet_v2_m(weights='DEFAULT')
    else:
        efficientnet_v2_m = models.efficientnet_v2_m(weights=None)
    efficientnet_v2_m.classifier[1] = nn.Linear(efficientnet_v2_m.classifier[1].in_features, num_classes)

    return efficientnet_v2_m

def  MobileNet_v3_large(pretrained=True, num_classes=1000):
    if pretrained:
        mobilenet_v3_large = models.mobilenet_v3_large(weights='DEFAULT')
    else:
        mobilenet_v3_large = models.mobilenet_v3_large(weights=None)
    mobilenet_v3_large.classifier[3] = nn.Linear(mobilenet_v3_large.classifier[3].in_features, num_classes)

    return mobilenet_v3_large

def  ResNext101_32x8d(pretrained=True, num_classes=1000):
    if pretrained:
        resnext101_32x8d = models.resnext101_32x8d(weights='DEFAULT')
    else:
        resnext101_32x8d = models.resnext101_32x8d(weights=None)

    resnext101_32x8d.fc = nn.Linear(resnext101_32x8d.fc.in_features, num_classes)

    return resnext101_32x8d


#%%

def  Swin_b(pretrained=True, num_classes=1000):
    if pretrained:
        swin_b = models.swin_b(weights='DEFAULT')
    else:
        swin_b = models.swin_b(weights=None)
    swin_b.head = nn.Linear(swin_b.head.in_features, num_classes)

    return swin_b

def  ViT_l_32(pretrained=True, num_classes=1000):
    if pretrained:
        vit_l_32 = models.vit_l_32(weights='DEFAULT')
    else:
        vit_l_32 = models.vit_l_32(weights=None)
    vit_l_32.heads[0] = nn.Linear(vit_l_32.heads[0].in_features, num_classes)

    return vit_l_32

def  MaxViT_t(pretrained=True, num_classes=1000):

    if pretrained:
        maxvit_t = models.maxvit_t(weights='DEFAULT')
    else:
        maxvit_t = models.maxvit_t(weights=None)
    maxvit_t.classifier[5] = nn.Linear(maxvit_t.classifier[5].in_features, num_classes)

    return maxvit_t

#%%


# model = MaxViT_t(pretrained=True, num_classes=790)

# from torchinfo import summary
# summary(model, input_size=(1,3,224,224), depth=2)
# %%
class MaxVitFeatures(nn.Module):
    def __init__(self, original_model):
        super(MaxVitFeatures, self).__init__()

        model_children = list(original_model.children())

        # assuming preprocessing is the first child and rest are the blocks
        self.preprocess = model_children[0]
        self.blocks = model_children[1]  # this should be a ModuleList of blocks according to your description

    def forward(self, x):
        outputs = []
        x = self.preprocess(x)
        
        for block in self.blocks:
            x = block(x)
            outputs.append(x)

        return outputs

maxvit_t = models.maxvit_t(weights='DEFAULT')
maxvit_features = MaxVitFeatures(maxvit_t)

class ResNetFeatures(nn.Module):
    def __init__(self, original_model):
        super(ResNetFeatures, self).__init__()
        # Each 'block' corresponds to one of the four main blocks in ResNet
        self.block1 = nn.Sequential(*list(original_model.children())[:5]) # First block
        self.block2 = list(original_model.children())[5] # Second block
        self.block3 = list(original_model.children())[6] # Third block
        self.block4 = list(original_model.children())[7] # Fourth block

    def forward(self, x):
        outputs = []
        x = self.block1(x)
        outputs.append(x)
        x = self.block2(x)
        outputs.append(x)
        x = self.block3(x)
        outputs.append(x)
        x = self.block4(x)
        outputs.append(x)
        return outputs

resnext101_32x8d = models.resnext101_32x8d(pretrained=True)
resnext101_32x8d_wrapper = ResNetFeatures(resnext101_32x8d)