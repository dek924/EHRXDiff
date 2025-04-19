import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytorch_lightning as pl

from torch.utils.data import random_split, DataLoader
from torchvision.transforms import Compose, ToTensor, Resize, Normalize

from cheff.machex import MaCheXDataset, MimicT2IDataset
from EHRXDiff.cheff.custom_dataset import CXREHRDataset


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, machex_path, test_size, num_workers, mimic=False, *args, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transforms = Compose([Resize(256), ToTensor(), Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

        if not mimic:
            self.machex = MaCheXDataset(machex_path, self.transforms)
        else:
            self.machex = MimicT2IDataset(machex_path, self.transforms)

        train_size = len(self.machex) - test_size
        self.train_dataset, self.test_dataset = random_split(
            self.machex, (train_size, test_size), generator=torch.Generator().manual_seed(1337)
        )

    def train_dataloader(self):
        loader = DataLoader(
            dataset=self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=False, shuffle=True
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            dataset=self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=False, shuffle=False
        )
        return loader


class CustomDataModuleFromConfig(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        img_root_dir,
        img_meta_dir,
        tab_root_dir,
        tab_data_type,
        max_event_len,
        num_workers,
        debug=False,
        data_aug=False,
        null_cond=None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_root_dir = img_root_dir
        self.img_meta_dir = img_meta_dir
        self.tab_root_dir = tab_root_dir
        self.tab_data_type = tab_data_type
        self.max_event_len = max_event_len
        self.debug = debug
        self.data_aug = data_aug
        self.null_cond = null_cond

        if self.data_aug:
            self.train_transforms = None
            self.test_transforms = Compose([Resize(256), ToTensor(), Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
        else:
            self.train_transforms = Compose([Resize(256), ToTensor(), Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
            self.test_transforms = Compose([Resize(256), ToTensor(), Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

        self.train_dataset = CXREHRDataset(
            phase="train",
            img_root_dir=self.img_root_dir,
            img_meta_dir=self.img_meta_dir,
            tab_root_dir=self.tab_root_dir,
            tab_data_type=self.tab_data_type,
            max_event_len=self.max_event_len,
            transforms=self.train_transforms,
            data_aug=self.data_aug,
            debug=self.debug,
            null_cond=self.null_cond,
        )
        self.val_dataset = CXREHRDataset(
            phase="valid",
            img_root_dir=self.img_root_dir,
            img_meta_dir=self.img_meta_dir,
            tab_root_dir=self.tab_root_dir,
            tab_data_type=self.tab_data_type,
            max_event_len=self.max_event_len,
            transforms=self.test_transforms,
            debug=self.debug,
            null_cond=self.null_cond,
        )
        self.test_dataset = CXREHRDataset(
            phase="test",
            img_root_dir=self.img_root_dir,
            img_meta_dir=self.img_meta_dir,
            tab_root_dir=self.tab_root_dir,
            tab_data_type=self.tab_data_type,
            max_event_len=self.max_event_len,
            transforms=self.test_transforms,
            debug=self.debug,
            null_cond=self.null_cond,
        )

    def train_dataloader(self):
        loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            shuffle=True,
            collate_fn=self.train_dataset.collate_fn,
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            shuffle=False,
            collate_fn=self.val_dataset.collate_fn,
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            shuffle=False,
            collate_fn=self.test_dataset.collate_fn,
        )
        return loader