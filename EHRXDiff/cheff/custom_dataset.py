import os
import gc
import h5py
import json
import random
import pickle
import numpy as np
import pandas as pd

import torch
from PIL import Image
from torch.nn.functional import pad
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize, RandomAffine, Resize
from cheff.utils import load_mimic_cxr_meta, load_tab_h5py_file


class CXREHRDataset(Dataset):
    def __init__(
        self,
        phase,
        img_root_dir,
        img_meta_dir,
        tab_root_dir,
        tab_data_type,
        max_event_len,
        transforms,  # for image
        data_aug=False,
        debug=False,
        null_cond=None,
    ):
        super().__init__()
        assert null_cond in ["table", None]

        self.phase = phase
        self.img_root_dir = img_root_dir
        self.tab_root_dir = tab_root_dir
        self.data_aug = data_aug
        self.null_cond = null_cond

        # load img_meta: dicom_id, jpg_fpath
        self.img_meta = load_mimic_cxr_meta(img_meta_dir, self.img_root_dir)
        self.img_meta = self.img_meta[["dicom_id", "jpg_fpath"]]
        self.img_meta = self.img_meta.set_index("dicom_id")
        print("Load image meta info : ", self.img_meta.shape)

        # Load table data
        self.max_event_len = max_event_len
        self.tab_data_type = tab_data_type

        if debug:
            self.phase = "test"
        print(f"Loading tab_inputs for {self.phase} from {tab_root_dir}...")

        self.tab_inputs, self.tab_input_key = load_tab_h5py_file(self.tab_data_type, self.phase, self.tab_root_dir)

        table_info = pd.DataFrame({"data_key": self.tab_input_key})
        table_info["null_cond"] = None
        print("null cond setting: ", str(self.null_cond))

        if self.null_cond == "table":
            cond_tab_info = table_info.copy()
            print("Init table info: ", cond_tab_info.shape)

            cond_tab_info["dicom_id"] = cond_tab_info["data_key"].str.split("_").str[0]
            cond_tab_info = cond_tab_info.drop_duplicates(subset=["dicom_id"]).drop(columns=["dicom_id"])
            cond_tab_info["null_cond"] = "table"
            table_info = pd.concat([table_info, cond_tab_info]).reset_index(drop=True)
            self.trg_transforms = Compose(
                [
                    Resize(256),
                    RandomAffine(degrees=10, scale=(0.9, 1.1)),
                    ToTensor(),
                    Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ]
            )
            print("Add null cond: table, ", table_info.shape)

        self.table_info = table_info
        print(f"Load {self.phase} tab_inputs: {len(self.table_info)}")

        if transforms is None:
            if self.data_aug:
                self.transforms = Compose(
                    [
                        Resize(256),
                        RandomAffine(degrees=10, scale=(0.9, 1.1)),
                        ToTensor(),
                        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                    ]
                )
            else:
                self.transforms = ToTensor()
        else:
            self.transforms = transforms

    def __len__(self):
        return len(self.table_info)

    def img_processing(self, target_img, prev_img):
        if self.data_aug or self.null_cond:
            transform_seed = np.random.randint(2147483647)
            random.seed(transform_seed)
            torch.manual_seed(transform_seed)

        if self.null_cond:
            target_img = self.trg_transforms(target_img)
            prev_img = self.transforms(prev_img)
        else:
            target_img = self.transforms(target_img)
            prev_img = self.transforms(prev_img)
        gc.collect()
        return target_img, prev_img

    def __getitem__(self, idx):
        sample = self.table_info.iloc[idx]
        data_key = sample["data_key"]
        null_cond = sample["null_cond"]
        target_dicom_id, prev_dicom_id = data_key.split("_")
        if null_cond == "table":
            target_dicom_id = prev_dicom_id

        # imgs
        prev_img = Image.open(self.img_meta.loc[prev_dicom_id, "jpg_fpath"]).convert("RGB")
        target_img = Image.open(self.img_meta.loc[target_dicom_id, "jpg_fpath"]).convert("RGB")
        target_img, prev_img = self.img_processing(target_img, prev_img)

        # table
        if null_cond == "table":
            table_input = np.zeros([self.max_event_len, 1536])
        else:
            table_input = np.array(self.tab_inputs["ehr"][data_key])
        attn_mask = np.zeros(len(table_input))

        outputs = {
            "table": torch.FloatTensor(table_input),
            "attn_mask": torch.BoolTensor(attn_mask),
            "prev_img": torch.FloatTensor(prev_img),
            "target_img": torch.FloatTensor(target_img),
            "data_key": data_key,
        }
        gc.collect()
        return outputs

    def collate_fn(self, batch):
        table_emb = [pad(b["table"], (0, 0, 0, self.max_event_len - len(b["table"]))) for b in batch]
        attn_mask = [pad(b["attn_mask"], (0, self.max_event_len - len(b["attn_mask"])), value=1) for b in batch]
        gc.collect()
        return {
            "prev_img": torch.stack([b["prev_img"] for b in batch]),
            "target_img": torch.stack([b["target_img"] for b in batch]),
            "table": torch.stack(table_emb),
            "attn_mask": torch.stack(attn_mask).bool(),
            "data_key": [b["data_key"] for b in batch],
        }
