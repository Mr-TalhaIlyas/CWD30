[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FMr-TalhaIlyas%2FCWD30&count_bg=%2300E7FD&title_bg=%23555555&icon=microsoftonedrive.svg&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

# CWD30 | [Project Page](https://cwd-30.github.io/cwd-30/)

#### Full Paper [arXiv](https://arxiv.org/abs/2305.10084) 
CWD30 comprises over 219,770 high-resolution images of 20 weed species and 10 crop species, encompassing various growth stages, multiple viewing angles, and environmental conditions. The images were collected from diverse agricultural fields across different geographic locations and seasons, ensuring a representative dataset. 
#### Data Download [Link](https://o365jbnu-my.sharepoint.com/:f:/g/personal/talha_student_jbnu_ac_kr/EsdFSAmsct5KulaAkd7YRYUBJIXhvUcYQ2SzDhp2nB7OWg?e=oprZlS) 

##### [*If you use our data/paper in your projects kindly **cite** the paper and **star** the repo*]().

### Global Distribution of Crops in the CWD30 dataset.

![alt text](https://github.com/Mr-TalhaIlyas/CWD30/blob/main/screens/map.png)

## MODEL ZOO

<details>
<summary>Classification Models</summary>

⚠️NOTE⚠️ We are currently in middle of uploading the weights. All might not be available.

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

### Pretrained Weights on iNaturalist
  
|ReNet-101|Weights|Acc.|
|---|---|---|
|[iNat21](https://github.com/visipedia/inat_comp/tree/master/2021)|✅[chkpt](https://o365jbnu-my.sharepoint.com/:u:/g/personal/talha_student_jbnu_ac_kr/Eej_bdo_W4VMjG6GHfr3YS8B3sIJKeN32xXGI5rr4O_ajg?e=0N96wn)|<80%|
|[iNat17](https://github.com/visipedia/inat_comp/tree/master/2017)|✅[chkpt](https://o365jbnu-my.sharepoint.com/:u:/g/personal/talha_student_jbnu_ac_kr/EeeOI8gv3mxAg7dx6HXVanQBp_5dq4BFDpyyJ5CxQ-KpGQ?e=X4uNub)|60.41%|

</details>

<details>
<summary>Semantic Segmentation Models</summary>

  Access dataset via:
  * [Sugar Beet](https://www.ipb.uni-bonn.de/data/sugarbeets2016/)
  * [Carrot Weed](https://github.com/cwfid/dataset)
  * [Bean Weed](https://o365jbnu-my.sharepoint.com/personal/talha_student_jbnu_ac_kr/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Ftalha%5Fstudent%5Fjbnu%5Fac%5Fkr%2FDocuments%2FDatasets%2FBean%20UDA)
  
⚠️NOTE⚠️ We are currently in middle of uploading the weights. All might not be available.

|Model       |BeanWeed                   |SugarBeet                  |CarrotWeed                 |
|---         |---                        |---                        |---                        |
|UNet        |✅[72.49 mIOU, chkpt](https://o365jbnu-my.sharepoint.com/:u:/g/personal/talha_student_jbnu_ac_kr/EaHpmYLSs6dJmfVMlp4wjMwBqnnAJQz4QoskdSeKyN_mWw?e=iQ1dCA)     |✅[85.47 mIOU, chkpt](https://o365jbnu-my.sharepoint.com/:u:/g/personal/talha_student_jbnu_ac_kr/EZv7lWyh8sJJngFz3mhEfegBpxhEgBENA1UYYOmLw7OboA?e=02UuxA)      |✅[78.32 mIOU, chkpt](https://o365jbnu-my.sharepoint.com/:u:/g/personal/talha_student_jbnu_ac_kr/ESV2mP0mfqBEqXf4U0JYnVQBPgWujDMlU4ybhSdDtrHW9g?e=eDhpcW)      |
|DeepLab v3+ |[78.03 mIOU, chkpt]()      |[86.02 mIOU, chkpt]()      |✅[83.16 mIOU, chkpt](https://o365jbnu-my.sharepoint.com/:u:/g/personal/talha_student_jbnu_ac_kr/EVkfmjyMmapNii0jRxncs5UB4Ipi3qYiMNPEF4lQc6g_-w?e=9LtjUe)      |
|OCR         |[79.51 mIOU, chkpt]()      |[87.34 mIOU, chkpt]()      |✅[86.53 mIOU, chkpt](https://o365jbnu-my.sharepoint.com/:u:/g/personal/talha_student_jbnu_ac_kr/EQvrRiTSwBlBniZGkYicFl8Bf4pUPhLyJeBWCUD6LOCW6Q?e=hWyD5E)      |
|SegNext     |[83.90 mIOU, chkpt]()      |[87.65 mIOU, chkpt]()      |✅[88.54 mIOU, chkpt](https://o365jbnu-my.sharepoint.com/:u:/g/personal/talha_student_jbnu_ac_kr/EdI8iQLqhX9LvgTKAFGyHMEBhqc4wcBW7yOVNKb9q78j3A?e=AvJhty)      |

✅[MSCAN backbone SegNext](https://o365jbnu-my.sharepoint.com/:u:/g/personal/talha_student_jbnu_ac_kr/EQdev4A2alhOjhFLw5hxoOwBkIfW6tTD_RD9ElF1AqpvEA?e=s8D7oi)

</details>

<details>
<summary>Instances Segmentation Models</summary>

Access dataset via:
* [PhenoBench](https://www.phenobench.org/)
* [GrowliFlower](https://rs.ipb.uni-bonn.de/data/growliflower/)

|Model|Data|Weights|PQ|
|---|---|---|---|
|MaskRCNN (ResNet-101 FPN backbone)|PhenoBench|✅[chkpt](https://o365jbnu-my.sharepoint.com/:u:/g/personal/talha_student_jbnu_ac_kr/EZsslt1DqAlGnNLFPX67AAABcyLAQOayNRM_K_Me-yyCeA?e=J2eWBy)|44.05|
|MaskRCNN (ResNet-101 FPN backbone)|GrowliFlower|✅[chkpt](https://o365jbnu-my.sharepoint.com/:u:/g/personal/talha_student_jbnu_ac_kr/EUVXn6Az9fxEjgsCHJA4BMUB5O-S0x0U_C22NP__-AT6aQ?e=uFPxbr)|56.33|
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
