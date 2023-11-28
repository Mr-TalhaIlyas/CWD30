#%%
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "2";
from termcolor import cprint
from tkinter.tix import Tree
import torchvision
import torch
import os
import torch.nn as nn
from torchmetrics import JaccardIndex
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.utils.tensorboard import SummaryWriter
import torchvision.ops as tops
import yaml
import torchvision.models.detection as detection_models
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor



class CustomMaskRCNN(detection_models.MaskRCNN):
    def __init__(self, num_classes, pretrain_weights_path='None',
                 in1k_pretrained=False, freeze_backbone=False, debug=False):

        # Initialize the ResNet-101 backbone with FPN
        backbone = resnet_fpn_backbone('resnet101', pretrained=in1k_pretrained)

        # Initialize Mask R-CNN with the custom backbone
        super().__init__(backbone, num_classes)

        # Load COCO pre-trained weights into the model (for common layers)
        model_coco = detection_models.maskrcnn_resnet50_fpn_v2(pretrained=True)
        coco_state_dict = model_coco.state_dict()

        # Update the state_dict of your model with COCO pre-trained weights
        # Skipping weights for the backbone and heads if they are incompatible
        self_state_dict = self.state_dict()
        for name, param in coco_state_dict.items():
            if name in self_state_dict and param.shape == self_state_dict[name].shape:
                self_state_dict[name].copy_(param)
        print('COCO pre-trained weights loaded successfully...')
        # Replace the classifier and mask predictor for your number of classes
        in_features = self.roi_heads.box_predictor.cls_score.in_features
        self.roi_heads.box_predictor = detection_models.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

        in_features_mask = self.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256  # Can be tuned
        self.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

         # Load pre-trained weights into the backbone if provided
        if pretrain_weights_path != 'None':
            if os.path.exists(pretrain_weights_path):
                chkpt = torch.load(pretrain_weights_path)
                if 'model_state_dict' in chkpt:
                    pretrained_dict = chkpt['model_state_dict']
                    # Remove fully connected layer parameters from dict
                    for k in ['fc.weight', 'fc.bias']:
                        if k in pretrained_dict:
                            del pretrained_dict[k]
                    # Add 'body.' prefix to match the keys in backbone's state_dict
                    pretrained_dict = {'backbone.body.' + k: v for k, v in pretrained_dict.items()}
                    missing_keys, unexpected_keys = self.load_state_dict(pretrained_dict, strict=False)
                    print("Loading new backbone Pre-trained weights...")
                    print(f"Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
                    print(f"Missing keys: {missing_keys}")
                    print(f"Weight file: {pretrain_weights_path}")
                    print("Custom pre-trained weights loaded successfully.")
                else:
                    print("No 'model_state_dict' key found in checkpoint.")
            else:
                print("No 'model_state_dict' key found in checkpoint.")

        # Freeze the ResNet-101 backbone layers
        if freeze_backbone:
            for name, parameter in backbone.body.named_parameters():
                parameter.requires_grad = False
            print("Backbone layers are frozen.")

        if debug:
            # (Optional) Check which layers are frozen
            for name, parameter in self.named_parameters():
                print(f"{name:50} | Requires Grad: {parameter.requires_grad}")


class MaskRCNN(nn.Module):

    def __init__(self, cfg: dict):
        super().__init__()
        self.n_classes = cfg['train']['n_classes']
        self.epochs = cfg['train']['max_epoch'] 
        self.batch_size = cfg['train']['batch_size']
        self.pretrained = cfg['pretrained_chkpt']
        self.freeze_backbone = cfg['freeze_backbone']
        self.in1k_pretrained = cfg['in1k_pretrained']

        self.iou = JaccardIndex(num_classes=self.n_classes, ignore_index=0, task='multiclass')
        self.ap = MeanAveragePrecision(box_format='xyxy')# reduction='none', num_classes=self.n_classes
        self.ap_ins = MeanAveragePrecision(box_format='xyxy', iou_type='segm')#, num_classes=self.n_classes

        # self.weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.COCO_V1
        # self.network = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=None, progress=True, num_classes=self.n_classes)
        self.network = CustomMaskRCNN(self.n_classes,
                                      self.pretrained,
                                      self.in1k_pretrained,
                                      self.freeze_backbone)
        self.network = self.network.float().cuda()
        # for name, param in self.network.named_parameters():
        #     param.requires_grad = True
        self.prob_th = cfg['val']['prob_th']
        self.overlapping_th = cfg['val']['nms_th']

        self.ckpt_dir, self.tboard_dir = self.set_up_logging_directories(cfg)
        self.writer = SummaryWriter(log_dir=self.tboard_dir)
        self.log_val_predictions = True
            
    def forward(self, batch):
        # moving everything to cuda here to avoid stalling when workers != 0
        for b in range(len(batch['targets'])):
            batch['image'][b] = batch['image'][b].cuda()
            for k in batch['targets'][b]:
                batch['targets'][b][k] = batch['targets'][b][k].cuda()
        # import ipdb;ipdb.set_trace()
        out = self.network(batch['image'], batch['targets'])
        return out 

    def getLoss(self, out):
        loss = out['loss_classifier'] + out['loss_box_reg'] + out['loss_mask'] + out['loss_objectness'] + out['loss_rpn_box_reg']
        return loss

    def training_step(self, batch):
        out = self.forward(batch)
        loss = self.getLoss(out)
        return loss

    def on_validation_start(self):
        if self.log_val_predictions:
            self.img_with_box = []
            self.img_with_masks = []
            self.img_with_sem = []

    def validation_step(self, batch):

        # moving everything to cuda here to avoid stalling when workers != 0
        for b in range(len(batch['targets'])):
            batch['image'][b] = batch['image'][b].cuda()
            for k in batch['targets'][b]:
                batch['targets'][b][k] = batch['targets'][b][k].cuda()
        out = self.network(batch['image'])

        # here start the postprocessing 
        img = batch['image'][0]
        _, h, w = img.shape
        b = len(batch['targets'])
        semantic_labels = torch.zeros((b,h,w))
        semantic_predictions = torch.zeros((b,h,w))
        instance_predictions = torch.zeros((b,h,w))

        predictions_dictionaries = []

        for b_idx in range(b):

            masks = out[b_idx]['masks'].squeeze()
            scores = out[b_idx]['scores']
            boxes = out[b_idx]['boxes']
            labels = out[b_idx]['labels']

            # non maximum suppression
            refined = tops.nms(boxes, scores, self.overlapping_th)
            refined_boxes = boxes[refined]
            refined_scores = scores[refined]
            refined_masks = masks[refined]
            refined_labels = labels[refined]

            # keeping only high scores
            high_scores = refined_scores > self.prob_th

            # if any scores are above self.prob_th we can compute metrics
            if high_scores.sum():
                surviving_boxes = refined_boxes[high_scores]
                surviving_scores = refined_scores[high_scores]
                surviving_masks = refined_masks[high_scores]
                surviving_labels = refined_labels[high_scores]
                
                surviving_dict = {}
                surviving_dict['boxes'] = surviving_boxes.cuda()
                surviving_dict['labels'] = surviving_labels.cuda()
                surviving_dict['scores'] = surviving_scores.cuda()
                surviving_dict['masks'] = (surviving_masks>0.5).type(torch.uint8).cuda()
                sem_out = surviving_labels.unsqueeze(dim=1).unsqueeze(dim=1)*(surviving_masks>0.5).type(torch.uint8)
                sem_out, _ = sem_out.max(dim=0)
                sem_out = sem_out.cuda()

                ins_out = torch.arange(surviving_masks.shape[0]).unsqueeze(dim=1).unsqueeze(dim=1).cuda()*surviving_masks
                ins_out, _ = ins_out.max(dim=0)
                ins_out = ins_out.cuda()
            
            # if not populate prediction dict with empty tensor to get 0 for ap and ap_ins
            # define zero sem and ins masks for iou metric
            else:
                surviving_dict = {}
                surviving_dict['boxes'] = torch.empty((0, 4)).cuda()
                surviving_dict['labels'] = torch.empty(0).cuda()
                surviving_dict['scores'] = torch.empty(0).cuda()
                surviving_dict['masks'] = torch.empty((0, h, w)).cuda()
                sem_out = torch.zeros((h, w)).cuda()
                ins_out = torch.zeros((h, w)).cuda()

            predictions_dictionaries.append(surviving_dict)

            if self.log_val_predictions:
                import matplotlib.pyplot as plt
                import cv2
                masks = surviving_dict['masks']
                labels = surviving_dict['labels']
                bbox = surviving_dict['boxes'].long()
                img = batch['image'][b_idx].cpu().permute(1, 2, 0).numpy()
                # import ipdb;ipdb.set_trace()
                mimg = torch.zeros(batch['image'][0].shape[1:]).cuda()
                for i in range(masks.shape[0]):
                    # import ipdb;ipdb.set_trace()
                    color = (1,0,0) if labels[i] == 2 else (0,1,0)
                    img = cv2.rectangle(
                        img, (bbox[i][0].item(), bbox[i][1].item()), (bbox[i][2].item(), bbox[i][3].item()), color, 3)
                    # import ipdb;ipdb.set_trace()
                    mimg += i*masks[i]
                self.img_with_box.append(img)
                self.img_with_masks.append(mimg)
                self.img_with_sem.append(sem_out)
            
            try:
                self.ap.update([surviving_dict], [batch['targets'][b_idx]])
                self.ap_ins.update([surviving_dict], [batch['targets'][b_idx]])
            except TypeError as e:
                print(f"Warning: Skipping a batch due to TypeError - {e}")
                

            sem_gt = batch['targets'][b_idx]['labels'].unsqueeze(
                dim=1).unsqueeze(dim=1)*batch['targets'][b_idx]['masks']
            sem_gt,_ = sem_gt.max(dim=0)


            semantic_labels[b_idx,:,:] = sem_gt
            semantic_predictions[b_idx,:,:] = sem_out
            instance_predictions[b_idx,:,:] = ins_out
        
        del surviving_dict, sem_out, ins_out
        torch.cuda.empty_cache()
        self.iou.update(semantic_predictions.long(), semantic_labels.long())
        return semantic_predictions, instance_predictions, predictions_dictionaries

    def test_step(self, batch):
        # moving everything to cuda here to avoid stalling when workers != 0
        for b in range(len(batch['targets'])):
            batch['image'][b] = batch['image'][b].cuda()
            for k in batch['targets'][b]:
                batch['targets'][b][k] = batch['targets'][b][k].cuda()
        out = self.network(batch['image'])

        # here start the postprocessing 
        img = batch['image'][0]
        _, h, w = img.shape
        b = len(batch['targets'])
        semantic_labels = torch.zeros((b,h,w))
        semantic_predictions = torch.zeros((b,h,w))
        instance_predictions = torch.zeros((b,h,w))

        predictions_dictionaries = []

        for b_idx in range(b):

            masks = out[b_idx]['masks'].squeeze()
            scores = out[b_idx]['scores']
            boxes = out[b_idx]['boxes']
            labels = out[b_idx]['labels']

            # non maximum suppression
            refined = tops.nms(boxes, scores, self.overlapping_th)
            refined_boxes = boxes[refined]
            refined_scores = scores[refined]
            refined_masks = masks[refined]
            refined_labels = labels[refined]

            # keeping only high scores
            high_scores = refined_scores > self.prob_th

            # if any scores are above self.prob_th we can compute metrics
            if high_scores.sum():
                surviving_boxes = refined_boxes[high_scores]
                surviving_scores = refined_scores[high_scores]
                surviving_masks = refined_masks[high_scores]
                surviving_labels = refined_labels[high_scores]
                
                surviving_dict = {}
                surviving_dict['boxes'] = surviving_boxes.cuda()
                surviving_dict['labels'] = surviving_labels.cuda()
                surviving_dict['scores'] = surviving_scores.cuda()
                surviving_dict['masks'] = surviving_masks.type(torch.uint8).cuda()

                surviving_masks[surviving_masks>=0.5] = 1
                surviving_masks[surviving_masks<0.5] = 0

                sem_out = surviving_labels.unsqueeze(dim=1).unsqueeze(dim=1)*surviving_masks
                sem_out, _ = sem_out.max(dim=0)
                sem_out = sem_out.cuda()

                ins_out = (torch.arange(surviving_masks.shape[0]).unsqueeze(dim=1).unsqueeze(dim=1).cuda()+1)*surviving_masks
                ins_out, _ = ins_out.max(dim=0)
                ins_out = ins_out.cuda()
            
            # if not populate prediction dict with empty tensor to get 0 for ap and ap_ins
            # define zero sem and ins masks for iou metric
            else:
                surviving_dict = {}
                surviving_dict['boxes'] = torch.empty((0, 4)).cuda()
                surviving_dict['labels'] = torch.empty(0).cuda()
                surviving_dict['scores'] = torch.empty(0).cuda()
                surviving_dict['masks'] = torch.empty((0, h, w)).cuda()
                sem_out = torch.zeros((h, w)).cuda()
                ins_out = torch.zeros((h, w)).cuda()

            predictions_dictionaries.append(surviving_dict)

            sem_gt = batch['targets'][b_idx]['labels'].unsqueeze(
                dim=1).unsqueeze(dim=1)*batch['targets'][b_idx]['masks']
            sem_gt,_ = sem_gt.max(dim=0)


            semantic_labels[b_idx,:,:] = sem_gt
            semantic_predictions[b_idx,:,:] = sem_out
            instance_predictions[b_idx,:,:] = ins_out
            
        del surviving_dict, sem_out, ins_out
        torch.cuda.empty_cache()
        return semantic_predictions, instance_predictions, predictions_dictionaries

    def compute_metrics(self):
        ap, ins, iou = self.ap.compute(), self.ap_ins.compute(), self.iou.compute()
        # import ipdb;ipdb.set_trace()
        self.ap.reset()
        self.ap_ins.reset()
        self.iou.reset()       
        return ap, ins, iou

    @staticmethod
    def to_cpu(input, output):
        for x in output:
            x['boxes'] = x['boxes'].cpu()
            x['scores'] = x['scores'].cpu()
            x['labels'] = x['labels'].cpu()
            x['masks'] = x['masks'].to(torch.uint8).squeeze().cpu()

        for x in input['image']:
            x = x.cpu()

        for x in input['targets']:
            x['boxes'] = x['boxes'].cpu()
            x['labels'] = x['labels'].cpu()
            x['masks'] = x['masks'].cpu() 

        return input, output

    @staticmethod
    def set_up_logging_directories(cfg):
        os.makedirs(cfg['checkpoint'], exist_ok = True) 
        os.makedirs(cfg['tensorboard'], exist_ok = True) 

        versions = os.listdir(cfg['checkpoint'])
        versions.sort()

        if len(versions) == 0:
            current_version = 0

        else:
            max_v = 0
            for fname in versions:
                if os.path.isdir(os.path.join(cfg['checkpoint'],fname)):
                    tmp_v = int(fname.split('_')[1])
                    if tmp_v > max_v:
                        max_v = tmp_v

            current_version = max_v  + 1

        new_dir = 'version_{}'.format(current_version)
        ckpt = os.path.join(cfg['checkpoint'],new_dir)
        tboard = os.path.join(cfg['tensorboard'],new_dir)
        os.makedirs(ckpt, exist_ok = True) 
        os.makedirs(tboard, exist_ok = True) 
        # save cfg here
        cfg_path = os.path.join(ckpt, 'cfg.yaml')
        with open(cfg_path, 'w') as f: 
            yaml.dump(cfg, f, default_flow_style=False)

        return ckpt, tboard
