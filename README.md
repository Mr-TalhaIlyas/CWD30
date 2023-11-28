[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FMr-TalhaIlyas%2FCWD30&count_bg=%2300E7FD&title_bg=%23555555&icon=microsoftonedrive.svg&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

# CWD30

#### Full Paper [arXiv](https://arxiv.org/abs/2305.10084) 
CWD30 comprises over 219,770 high-resolution images of 20 weed species and 10 crop species, encompassing various growth stages, multiple viewing angles, and environmental conditions. The images were collected from diverse agricultural fields across different geographic locations and seasons, ensuring a representative dataset. 
#### Data Download [Link](https://o365jbnu-my.sharepoint.com/:f:/g/personal/talha_student_jbnu_ac_kr/EsdFSAmsct5KulaAkd7YRYUBJIXhvUcYQ2SzDhp2nB7OWg?e=oprZlS) 

##### [*If you use our data/paper in your projects kindly **cite** the paper and **star** the repo*]().

### Global Distribution of Crops in the CWD30 dataset.

![alt text](https://github.com/Mr-TalhaIlyas/CWD30/blob/main/screens/map.png)

## MODEL ZOO

<details>
<summary>Classification Models</summary>

|Model|Weights|Acc|
|---|---|---|
|ResNet-18|[chkpt]()|79.5|
|ResNet-50|[chkpt]()|84.6|
|ResNet-101|[chkpt]()|81.36|
|MobileNetv3-S|[chkpt]()|80.5|
|MobileNetv3-L|[chkpt]()|74.67|
|EffNet-B0|[chkpt]()|83.2|
|EffNet-B3|[chkpt]()|83.64|
|EffNet-B5|[chkpt]()|84.5|
|ConvNeXt-T|[chkpt]()|85.6|
|ConvNeXt-M|[chkpt]()|85.9|
|ConvNeXt-L|[chkpt]()|84.7|
|ViT-T|[chkpt]()|83.43|
|ViT-B|[chkpt]()|86.4|
|CaiT-T|[chkpt]()|85.2|
|CaiT-S|[chkpt]()|86.9|
|Swin-T|[chkpt]()|85.59|
|Swin-B|[chkpt]()|85.3|
|Swin-L|[chkpt]()|87.0|
|MaxViT-S|[chkpt]()|86.5|
|MaxViT-B|[chkpt]()|87.08|
|CoAtNet-1|[chkpt]()|86.1|
|CoAtNet-3|[chkpt]()|84.3|
|EffFormer-L1|[chkpt]()|80.5|
|EffFormer-L3|[chkpt]()|82.7|
|EffFormer-L7|[chkpt]()|81.2|
</details>

<details>
<summary>Semantic Segmentation Models</summary>

|Model       |BeanWeed                   |SugarBeet                  |CarrotWeed                 |
|---         |---                        |---                        |---                        |
|UNet        |[44.05 mIOU, chkpt]()      |[44.05 mIOU, chkpt]()      |[44.05 mIOU, chkpt]()      |
|DeepLab v3+ |[56.33 mIOU, chkpt]()      |[56.33 mIOU, chkpt]()      |[56.33 mIOU, chkpt]()      |
|OCR         |[56.33 mIOU, chkpt]()      |[56.33 mIOU, chkpt]()      |[56.33 mIOU, chkpt]()      |
|SegNext     |[56.33 mIOU, chkpt]()      |[56.33 mIOU, chkpt]()      |[56.33 mIOU, chkpt]()      |
</details>

<details>
<summary>Instances Segmentation Models</summary>

|Model|Data|Weights|PQ|
|---|---|---|---|
|MaskRCNN (ResNet-101 FPN backbone)|PhenoBench|[chkpt]()|44.05|
|MaskRCNN (ResNet-101 FPN backbone)|GrowliFlower|[chkpt]()|56.33|
</details>


### Citation
```
@article{ilyas2023cwd30,
  title={CWD30: A Comprehensive and Holistic Dataset for Crop Weed Recognition in Precision Agriculture},
  author={Ilyas, Talha and Arsa, Dewa Made Sri and Ahmad, Khubaib and Jeong, Yong Chae and Won, Okjae and Lee, Jong Hoon and Kim, Hyongsuk},
  journal={arXiv preprint arXiv:2305.10084},
  year={2023}
}
```
```
Paper is currently under review. ;)
```
