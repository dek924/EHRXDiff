import os
import gc
import h5py
import pickle
import numpy as np
import pandas as pd

import torch
from PIL import Image
from torch.nn.functional import pad
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
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
    ):
        super().__init__()

        self.phase = phase
        self.img_root_dir = img_root_dir
        self.tab_root_dir = tab_root_dir
        self.data_aug = data_aug

        # load img_meta: dicom_id, jpg_fpath
        self.img_meta = load_mimic_cxr_meta(img_meta_dir, self.img_root_dir)
        self.img_meta = self.img_meta[["subject_id", "dicom_id", "jpg_fpath"]]
        self.img_meta = self.img_meta.set_index("dicom_id")
        print("Load image meta info : ", self.img_meta.shape)

        # Load table data
        self.max_event_len = max_event_len
        self.tab_data_type = tab_data_type

        # load tab_inputs: data_key (target_dicom_id + prev_dicom_id), tab_inputs
        if debug:
            self.phase = "test"
        print(f"Loading tab_inputs for {self.phase} from {tab_root_dir}...")

        self.tab_inputs, self.data_keys = load_tab_h5py_file(self.tab_data_type, self.phase, self.tab_root_dir)
        print(f"Load {self.phase} tab_inputs: {len(self.data_keys)}")

        if transforms is None:
            self.transforms = ToTensor()
        else:
            self.transforms = transforms

        self.modes = ["table", "prev_img", "img"]

    def __len__(self):
        return len(self.data_keys)

    def __getitem__(self, idx):
        data_key = self.data_keys[idx]
        _data_key = self.data_keys[idx]
        target_dicom_id, prev_dicom_id = data_key.split("_")

        target_img = Image.open(self.img_meta.loc[target_dicom_id, "jpg_fpath"]).convert("RGB")
        prev_img = Image.open(self.img_meta.loc[prev_dicom_id, "jpg_fpath"]).convert("RGB")

        target_img = self.transforms(target_img)
        prev_img = self.transforms(prev_img)

        # table
        table_input = np.array(self.tab_inputs["ehr"][data_key])
        attn_mask = np.zeros(len(table_input))

        outputs = {
            "table": torch.FloatTensor(table_input),
            "attn_mask": torch.BoolTensor(attn_mask),
            "prev_img": torch.FloatTensor(prev_img),
            "target_img": torch.FloatTensor(target_img),
            "data_key": _data_key,
        }

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


