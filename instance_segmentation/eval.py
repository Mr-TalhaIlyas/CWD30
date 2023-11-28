import os
import cv2
import numpy as np
from sklearn.metrics import jaccard_score
from tqdm import tqdm
from collections import defaultdict

# Function to calculate IoU between two binary masks
def calculate_iou(gt, pred):
    intersection = np.logical_and(gt, pred)
    union = np.logical_or(gt, pred)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

# Function to calculate PQ for a single pair of images
def calculate_pq_for_image(gt_image, pred_image):
    # Find unique instances in ground truth and predictions
    gt_instances = np.unique(gt_image)[1:]  # Exclude background
    pred_instances = np.unique(pred_image)[1:]  # Exclude background

    # Match predictions to ground truth instances
    matched_instances = {}  # {gt_instance: (pred_instance, IoU)}
    for gt_id in gt_instances:
        gt_mask = gt_image == gt_id
        for pred_id in pred_instances:
            pred_mask = pred_image == pred_id
            iou = calculate_iou(gt_mask, pred_mask)
            
            if iou > 0.5:  # Threshold for match
                if gt_id not in matched_instances or matched_instances[gt_id][1] < iou:
                    matched_instances[gt_id] = (pred_id, iou)

    # Calculate PQ components
    tp = len(matched_instances)  # True positives
    fp = len(pred_instances) - len(matched_instances)  # False positives
    fn = len(gt_instances) - len(matched_instances)  # False negatives

    # Calculate PQ
    iou_sum = sum([iou for _, iou in matched_instances.values()])
    sq = iou_sum / (tp + 0.5 * fp + 0.5 * fn) if tp > 0 else 0
    rq = tp / (tp + 0.5 * fp + 0.5 * fn) if tp > 0 else 0
    pq = sq * rq

    return pq, sq, rq, tp, fp, fn

# Function to calculate PQ over a dataset
def calculate_pq_over_dataset(gt_dir, pred_dir, resize=256):
    pq_scores = defaultdict(int)
    print('Calculating PQ...')
    # for gt_file in tqdm(sorted(os.listdir(gt_dir)), desc='Calculating PQ'):
    for gt_file in sorted(os.listdir(gt_dir)):
        pred_file = gt_file  # Assumes prediction file has the same name
        
        gt_path = os.path.join(gt_dir, gt_file)
        pred_path = os.path.join(pred_dir, pred_file)
        
        gt_image = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
        pred_image = cv2.imread(pred_path, cv2.IMREAD_UNCHANGED)
        
        gt_image = cv2.resize(gt_image, (resize, resize), interpolation=cv2.INTER_NEAREST)
        pred_image = cv2.resize(pred_image, (resize, resize), interpolation=cv2.INTER_NEAREST)
        
        pq, sq, rq, tp, fp, fn = calculate_pq_for_image(gt_image, pred_image)
        pq_scores['pq'] += pq
        pq_scores['sq'] += sq
        pq_scores['rq'] += rq
        pq_scores['tp'] += tp
        pq_scores['fp'] += fp
        pq_scores['fn'] += fn

    # Calculate average scores
    num_images = len(os.listdir(gt_dir))
    for key in pq_scores:
        pq_scores[key] /= num_images

    return pq_scores