#%%
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "2";
import click
import os
import torch
from os.path import join, dirname, abspath
from torch.utils.data import DataLoader
from dataloaders.datasets import Leaves, collate_pdc
import models
import yaml
import time
from tqdm import tqdm
from matplotlib import cm
import matplotlib
import cv2
from tqdm import tqdm
import numpy as np
from eval import calculate_pq_over_dataset

def save_model(model, epoch, optim, name):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        }, name)


percentage = 0.1
config = "/home/user01/data/talha/phenobench/leaf_instance_segmentation/rcnn/configs/maskrcnn_leaves.yaml"
cfg = yaml.safe_load(open(config))

out = cfg['data']['out']
os.makedirs(out, exist_ok=True)
os.makedirs(os.path.join(out,'predictions/leaf_instances/'), exist_ok=True)
os.makedirs(os.path.join(out,'predictions/semantics/'), exist_ok=True)
os.makedirs(os.path.join(out,'leaf_bboxes/'), exist_ok=True)

cfg['data']['percentage'] = percentage

train_dataset = Leaves(
    datapath=cfg['data']['train'], overfit=cfg['train']['overfit'])
train_loader = DataLoader(train_dataset, batch_size=cfg['train']['batch_size'], collate_fn=collate_pdc,
                            shuffle=True, drop_last=False, persistent_workers=True, pin_memory=True, num_workers=cfg['train']['workers'])
if cfg['train']['overfit']:
    val_loader = DataLoader(train_dataset, batch_size=cfg['train']['batch_size'], collate_fn=collate_pdc, shuffle=False,
                            drop_last=False, persistent_workers=True, pin_memory=True, num_workers=cfg['train']['workers'])
else:
    val_dataset = Leaves(
        datapath=cfg['data']['val'], overfit=cfg['train']['overfit'])
    val_loader = DataLoader(val_dataset, batch_size=cfg['train']['batch_size'], collate_fn=collate_pdc, shuffle=False,
                            drop_last=False, persistent_workers=True, pin_memory=True, num_workers=cfg['train']['workers'])


model = models.get_model(cfg)
optim = torch.optim.AdamW(model.network.parameters(), lr=cfg['train']['lr'])
scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.99)

best_map_det = 0
best_map_ins = 0
best_iou = 0
best_pq = 0
#%%
start = time.time()
with torch.autograd.set_detect_anomaly(False):
    n_iter = 0  # used for tensorboard
    for e in range(cfg['train']['max_epoch']):
        # print('start')
        model.network.train()
        start = time.time()

        for idx, item in enumerate(train_loader):

            optim.zero_grad()
            loss = model.training_step(item)
            loss.backward()
            optim.step()

            it_time = time.time() - start 
            print((
                    f'Epoch: {e}/{cfg["train"]["max_epoch"]} -- '
                    f'Step: {idx * cfg["train"]["batch_size"]}/{len(train_dataset)} -- '
                    f'Loss: {loss.item():.4f} -- '
                    f'Lr: {scheduler.get_lr()[0]:.6f} -- '
                    f'Time: {it_time:.2f} sec'
                ))
            model.writer.add_scalar('Loss/Train/', loss.detach().cpu().item(), n_iter)
            n_iter += 1
            start = time.time()
        
        scheduler.step()
        name = os.path.join(model.ckpt_dir,'last.pt')
        save_model(model, e, optim, name)
        # Validation
        if (e + 1) % 2 == 0: # eval every 2 epoch
            model.network.eval()
            for idx, item in enumerate(iter(val_loader)):
                with torch.no_grad():
                    size = item['image'][0].shape[1]
                    _, instance, _ = model.test_step(item)

                    res_names = item['name']
                    for i in range(len(res_names)):
                        fname_ins = os.path.join(out,'predictions/leaf_instances/',res_names[i])
                        # fname_sem = os.path.join(out,'predictions/semantics/',res_names[i])
                        # fname_box = os.path.join(out,'leaf_bboxes',res_names[i].replace('png','txt'))

                        # cv2.imwrite(fname_sem, semantic[i].cpu().long().numpy())
                        cv2.imwrite(fname_ins,instance[i].cpu().long().numpy())

                        size = item['image'][i].shape[1]
                        # scores = predictions[i]['scores'].cpu().numpy()
                        # labels = predictions[i]['labels'].cpu().numpy()
                        # converting boxes to center, width, height format
                        # boxes_ = predictions[i]['boxes'].cpu().numpy()
                        # num_pred = len(boxes_)
                        # cx = (boxes_[:,2] + boxes_[:,0])/2
                        # cy = (boxes_[:,3] + boxes_[:,1])/2
                        # bw = boxes_[:,2] - boxes_[:,0]
                        # bh = boxes_[:,3] - boxes_[:,1]

                        # # ready to be saved
                        # pred_cls_box_score = np.hstack((labels.reshape(num_pred,1), 
                        #                         cx.reshape(num_pred,1)/size,
                        #                         cy.reshape(num_pred,1)/size,
                        #                         bw.reshape(num_pred,1)/size,
                        #                         bh.reshape(num_pred,1)/size,
                        #                         scores.reshape(num_pred,1)
                        #                     ))
                        # np.savetxt(fname_box, pred_cls_box_score, fmt='%f')
                    print(f'Validation: {idx}/{len(val_loader)}')

            pred_dir = os.path.join(out,'predictions/leaf_instances/')
            gt_dir = os.path.join(cfg['data']['val'], 'leaf_instances/')
            pq_results = calculate_pq_over_dataset(gt_dir, pred_dir)

            
            print(30 * '==')
            print(f"Panoptic Quality (PQ): {pq_results['pq']}")
            print(f"Segmentation Quality (SQ): {pq_results['sq']}")
            print(f"Recognition Quality (RQ): {pq_results['rq']}")
            print(30 * '==')
        
            # checking improvements on validation set
            if pq_results['pq'] >= best_pq:
                print('Saving Checkpoint')
                best_pq = pq_results['pq']
                name = os.path.join(model.ckpt_dir,'best_pq.pt')
                save_model(model, e, optim, name)
                print('Done Saving Checkpoint')
#%%
        
        
