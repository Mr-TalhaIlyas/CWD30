import torch
import torch.nn.functional as F
import copy
from dataloaders.pdc_base import PlantsBase 

def collate_pdc(items):
    batch = {}
    images = []
    targets = []
    names=[]
    for i in range(len(items)):
        images.append(items[i]['image'])
        targets.append(items[i]['targets'])
        names.append(items[i]['name'])
    batch['image'] = list(images)
    batch['targets'] = list(targets)
    batch['name'] = list(names)
    return batch

class Leaves(PlantsBase):
    def __init__(self, datapath, overfit=False, area_threshold=50, cfg=None, is_train=False):
        super().__init__(datapath, overfit, cfg, is_train)
        self.area_threshold = area_threshold

    def masks_to_boxes(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Compute the bounding boxes around the provided masks.

        Returns a [N, 4] tensor containing bounding boxes. The boxes are in ``(x1, y1, x2, y2)`` format with
        ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

        Args:
            masks (Tensor[N, H, W]): masks to transform where N is the number of masks
                and (H, W) are the spatial dimensions.

        Returns:
            Tensor[N, 4]: bounding boxes
        """
        if masks.numel() == 0:
            return torch.zeros((0, 4), device=masks.device, dtype=torch.float)

        n = masks.shape[0]

        bounding_boxes = torch.zeros(
            (n, 4), device=masks.device, dtype=torch.float)

        for index, mask in enumerate(masks):
            if mask.sum() < self.area_threshold :
                continue
            y, x = torch.where(mask != 0)
            bounding_boxes[index, 0] = torch.min(x)
            bounding_boxes[index, 1] = torch.min(y)
            bounding_boxes[index, 2] = torch.max(x)
            bounding_boxes[index, 3] = torch.max(y)
        bounding_boxes_area = bounding_boxes.sum(dim=1)
        bounding_boxes = bounding_boxes[~(bounding_boxes_area==0)]
        return bounding_boxes, bounding_boxes_area


    def __getitem__(self, index):
        sample = self.get_sample(index)
        ignore_mask = sample['ignore_mask']
        instances = sample['leaf_instances'] * (~ignore_mask)
        instances = torch.unique(instances, return_inverse=True)[1]
        semantic = copy.deepcopy(instances)
        semantic[semantic > 0] = 1
        masks = F.one_hot(instances).permute(2, 0, 1) 
        cls_exploded = masks * semantic.unsqueeze(0)
        cls_exploded = torch.reshape(cls_exploded, (cls_exploded.shape[0], cls_exploded.shape[1] * cls_exploded.shape[2]))
        # cls_vec contains the class_id for each masks
        cls_vec, _ = torch.max(cls_exploded, dim=1)  
        # computing bounding boxes from masks
        boxes, area = self.masks_to_boxes(masks)
        # apply reduction for null boxes
        masks = masks[[~(area == 0)]]
        cls_vec = cls_vec[[~(area == 0)]]

        image = (sample['images']/255).float()

        maskrcnn_input = {}
        maskrcnn_input['image'] = image
        maskrcnn_input['name'] = sample['image_name']
        maskrcnn_input['targets'] = {}
        maskrcnn_input['targets']['masks'] = masks.to(torch.uint8)
        maskrcnn_input['targets']['labels'] = cls_vec
        maskrcnn_input['targets']['boxes'] = boxes

        return maskrcnn_input
