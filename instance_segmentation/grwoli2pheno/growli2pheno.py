# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 09:09:45 2023

@author: talha
"""

import cv2
from fmutils import directorytree as fdt
from fmutils import fmutils as fmu
from pathlib import Path
import numpy as np
import os

growli_data = 'D:/RV/agri/GrowliFlowerL/'
out_dir = 'D:/RV/agri/GrowliFlowerL/data/'
pheno_bench_dir = 'D:/RV/agri/phenobench-baselines/data/' # just for cloning dir structure
dim = 512 # resize dimension
fdt.clone_dir_tree(pheno_bench_dir, out_dir)

gimgs = os.path.join(growli_data, 'images')
glbls = os.path.join(growli_data, 'labels')

splits = ['train', 'val', 'test']

for split in splits:
    # split = splits[0]
    
    imgs = fmu.get_all_files(Path(gimgs, split))
    lbls = fmu.get_all_dirs(Path(glbls, split))
    
    leaf_insts = fmu.get_all_files(lbls[0]) # 'maskLeaves'
    plant_insts = fmu.get_all_files(lbls[1]) # 'maskPlants'
    voids = fmu.get_all_files(lbls[3]) # 'maskVoid'
    
    for i in range(len(imgs)):
        # i = 100
    
        filename = Path(leaf_insts[i]).stem
        
        img = cv2.imread(imgs[i], -1) # rewritng as png
        img = cv2.resize(img, (dim, dim), interpolation = cv2.INTER_LINEAR)
        
        leaf = cv2.imread(leaf_insts[i], -1)
        leaf = cv2.cvtColor(leaf, cv2.COLOR_BGR2GRAY)
        leaf = cv2.resize(leaf, (dim, dim), interpolation = cv2.INTER_NEAREST)
        assert np.min(np.unique(leaf)), f'BG label 60 not in image {leaf_insts[i]}'
        leaf_inst = np.where(leaf == 60, 0, leaf).astype(np.uint16)
        
        plant = cv2.imread(plant_insts[i], -1)
        plant = cv2.cvtColor(plant, cv2.COLOR_BGR2GRAY)
        plant = cv2.resize(plant, (dim, dim), interpolation = cv2.INTER_NEAREST)
        assert np.min(np.unique(plant)), f'BG label 60 not in image {plant_insts[i]}'
        plant_inst = np.where(plant == 60, 0, plant).astype(np.uint16)
        
        void = cv2.imread(voids[i], -1)
        void = cv2.cvtColor(void, cv2.COLOR_BGR2GRAY)
        void = cv2.resize(void, (dim, dim), interpolation = cv2.INTER_NEAREST)
        assert np.min(np.unique(void)) == 60, f'BG label 60 not in image {voids[i]}'
        void_inst = np.where(void == 60, 0, void).astype(np.uint16)
        
        _, sem1 = cv2.threshold(void_inst, 0, 1, cv2.THRESH_BINARY)
        sem1 = sem1 * 3 # 3 is ignored in dataloader for parital plants
        
        _, sem2 = cv2.threshold(plant_inst, 0, 1, cv2.THRESH_BINARY)
        sem2 = cv2.bitwise_and(sem2.astype(np.uint8), (1-sem1).astype(np.uint8))
        semantic = sem1 + sem2
        
        cv2.imwrite(f"{Path(out_dir, split, 'images', filename+'.png')}", img)
        cv2.imwrite(f"{Path(out_dir, split, 'leaf_instances', filename+'.png')}", leaf_inst)
        cv2.imwrite(f"{Path(out_dir, split, 'plant_instances', filename+'.png')}", plant_inst)
        cv2.imwrite(f"{Path(out_dir, split, 'semantics', filename+'.png')}", semantic)
        
        leaf_vis = np.where(semantic == 1, 255, semantic)
        leaf_vis = np.where(leaf_vis == 3, 45, leaf_vis)
        plant_vis = leaf_vis
        cv2.imwrite(f"{Path(out_dir, split, 'leaf_visibility', filename+'.png')}", leaf_vis.astype(np.uint8))
        cv2.imwrite(f"{Path(out_dir, split, 'plant_visibility', filename+'.png')}", plant_vis.astype(np.uint8))
    #     break
    # break
    print(f'Split {split} done.')
    
