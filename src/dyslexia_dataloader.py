import os
import json
from os.path import join
from typing import Callable, List, Dict

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


SIMPLE_PREPROCESS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ]
)


def load_image(path: str, width: int = 224, height: int = 224):
    img = cv2.imread(path)
    img = cv2.resize(img, (width, height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype('float')
    img /= 255
    return np.array(img)


def get_train_val_test_paths(
        group_ids: List[str],
        train_frac: float,
        val_frac: float
):
    np.random.seed(25)

    n = len(group_ids)
    indices = np.arange(n)
    np.random.shuffle(indices)

    group_ids = [group_ids[i] for i in indices]

    train_end = int(train_frac * n) + 1
    val_end   = int((train_frac + val_frac) * n) + 1

    train_group_ids = group_ids[:train_end]
    val_group_ids   = group_ids[train_end:val_end]
    test_group_ids  = group_ids[val_end:]
    return train_group_ids, val_group_ids, test_group_ids


def collate_fn(batch):
    return torch.stack([x[0] for x in batch]).float(), torch.tensor([x[1] for x in batch]).long()


class DyslexiaDataset(Dataset):
    def __init__(
        self,
        group_ids: List[str],
        group_id2grade: Dict[str, int],
        width: int,
        height: int,
        images_path: str,
        preprocess: Callable = SIMPLE_PREPROCESS
    ):
        assert all([gi in group_id2grade for gi in group_ids])

        self.group_ids      = group_ids
        self.group_id2grade = group_id2grade
        self.width          = width
        self.height         = height
        self.images_path    = images_path
        self.preprocess     = preprocess

    def __len__(self):
        return len(self.group_ids)

    def __getitem__(self, idx):
        group_id = self.group_ids[idx]

        image_name = f"{group_id}.png"
        image_path = join(self.images_path, image_name)
        image      = load_image(image_path, width=self.width, height=self.height)
        grade      = self.group_id2grade[group_id]

        image = self.preprocess(image)
        return [image, grade]


class DyslexiaDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        data_path: str,
        train_frac: float,
        val_frac: float,
        width: int = 224,
        height: int = 224,
        preprocess: Callable = SIMPLE_PREPROCESS,
        collate_fn: Callable = collate_fn
    ):
        super().__init__()

        images_path = join(data_path, "dyslexia_images")
        image_names = [image_name for image_name in sorted(os.listdir(images_path)) if ".png" in image_name]
        with open(join(data_path, "group_id2grade.json"), "r") as iofile:
            group_id2grade = json.load(iofile)

        group_ids = [image_name[:-4] for image_name in image_names]

        # shuffling and split
        train_group_ids, val_group_ids, test_group_ids = get_train_val_test_paths(group_ids, train_frac, val_frac)

        self.batch_size = batch_size
        self.data_path  = data_path
        self.train_frac = train_frac
        self.val_frac   = val_frac
        self.test_frac  = 1 - train_frac - val_frac
        self.width      = width
        self.height     = height
        self.preprocess = preprocess
        self.collate_fn = collate_fn

        args = (group_id2grade, width, height, images_path, preprocess)
        self.ds_train = DyslexiaDataset(train_group_ids, *args)
        self.ds_val   = DyslexiaDataset(val_group_ids, *args)
        self.ds_test  = DyslexiaDataset(test_group_ids, *args)

    def setup(self, stage: str):
        print(f"Stage: {stage}")
        print(
            f"Train: {len(self.ds_train)} images\n"
            f"Validation: {len(self.ds_val)} images\n"
            f"Test: {len(self.ds_test)} images\n"
        )

    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=16,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=16,
            collate_fn=self.collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.ds_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=16,
            collate_fn=self.collate_fn
        )
