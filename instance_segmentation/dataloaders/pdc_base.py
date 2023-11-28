import torch
from torch.utils.data import Dataset
import cv2
import os
from tqdm import tqdm

class PlantsBase(Dataset):
    def __init__(self, data_path, overfit=False, cfg=None, is_train=False):
        super().__init__()

        self.data_path = data_path
        self.overfit = overfit

        self.image_list = [
                x for x in os.listdir(os.path.join(self.data_path, "images")) if ".png" in x
            ]

        self.len = len(self.image_list)

        self.field_list = os.listdir(self.data_path)

    @staticmethod
    def load_data(data_path, image_list, field_list):
        data_frame = {}
        for field in tqdm(field_list):
            data_frame[field] = []
            for image_name in tqdm(image_list):
                image = cv2.imread(
                    os.path.join(os.path.join(data_path, field), image_name),
                    cv2.IMREAD_UNCHANGED,
                )
                if len(image.shape) > 2:
                    sample = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    sample = torch.tensor(sample).permute(2, 0, 1)
                else:
                    sample = torch.tensor(image.astype("int16"))

                data_frame[field].append(sample)
        return data_frame

    @staticmethod
    def load_one_data(data_path, image_list, field_list, idx):
        data_frame = {}
        for field in field_list:
            image = image_list[idx]
            image = cv2.imread(
                os.path.join(os.path.join(data_path, field), image),
                cv2.IMREAD_UNCHANGED,
            )
            if len(image.shape) > 2:
                sample = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                sample = torch.tensor(sample).permute(2, 0, 1)
            else:
                sample = torch.tensor(image.astype("int16"))

            data_frame[field] = sample
        return data_frame


    def get_sample(self, index):
        sample = {}

        sample = self.load_one_data(self.data_path, self.image_list, self.field_list, index)
        # import ipdb;ipdb.set_trace()
        partial_crops = sample["semantics"] == 3
        partial_weeds = sample["semantics"] == 4

        # 1 where there's stuff to be ignored by instance segmentation, 0 elsewhere
        sample["ignore_mask"] = torch.logical_or(partial_crops, partial_weeds).bool()

        # remove partial plants
        sample["semantics"][partial_crops] = 1
        sample["semantics"][partial_weeds] = 2

        # remove instances that aren't crops or weeds
        sample["plant_instances"][sample["semantics"] == 0] = 0
        sample["leaf_instances"][sample["semantics"] == 0] = 0

        # make ids successive
        sample["plant_instances"] = torch.unique(
            sample["plant_instances"] + sample["semantics"] * 1e6, return_inverse=True
        )[1]
        sample["leaf_instances"] = torch.unique(
            sample["leaf_instances"], return_inverse=True
        )[1]
        sample["leaf_instances"][sample["semantics"] == 2] = 0
        
        sample["image_name"] = self.image_list[index]
        return sample

    def __len__(self):
        if self.overfit:
            return 12
        return self.len
