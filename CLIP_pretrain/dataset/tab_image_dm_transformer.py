# Originally found in https://github.com/lucidrains/DALLE-pytorch
import os
import gc
import sys
import PIL
import h5py
import torch
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "EHRXDiff"))

from torchvision import transforms as T
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from cheff.utils import load_mimic_cxr_meta, load_tab_h5py_file


class TabImageDataset(Dataset):
    def __init__(
        self,
        phase,
        img_root_dir,
        img_meta_dir,
        tab_root_dir,
        tab_data_type="openai_filtered",
        max_event_len=1024,
        image_size=256,
        resize_ratio=0.75,
        shuffle=False,
    ):
        super().__init__()
        self.phase = phase
        self.shuffle = shuffle
        self.img_root_dir = img_root_dir
        self.img_meta_dir = img_meta_dir
        self.tab_root_dir = tab_root_dir
        self.tab_data_type = tab_data_type
        self.max_event_len = max_event_len

        # Load Image
        self.img_meta = load_mimic_cxr_meta(self.img_meta_dir, self.img_root_dir).astype("str")
        print("Load image meta info : ", self.img_meta.shape)

        def make_jpath(data):
            _subject_id = data["subject_id"]
            _study_id = data["study_id"]
            _dicom_id = data["dicom_id"]
            return f"/p{_subject_id[:2]}/p{_subject_id}/s{_study_id}/{_dicom_id}.jpg"

        self.img_meta["jpg_fpath"] = self.img_meta.apply(lambda x: make_jpath(x), axis=1)
        self.img_meta["jpg_fpath"] = self.img_root_dir + self.img_meta["jpg_fpath"]
        self.img_meta = self.img_meta[["dicom_id", "jpg_fpath"]]
        self.img_meta = self.img_meta.set_index("dicom_id")
        self.resize_ratio = resize_ratio
        self.image_transform = T.Compose(
            [
                T.Lambda(self.fix_img),
                T.RandomResizedCrop(image_size, scale=(self.resize_ratio, 1.0), ratio=(1.0, 1.0)),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        # Load Table
        self.tab_files, self.keys = load_tab_h5py_file(self.tab_data_type, self.phase, self.tab_root_dir)
        self.keys = list(self.keys)

        print("# of samples: ", len(self.keys))

    def __len__(self):
        return len(self.keys)

    def fix_img(self, img):
        return img.convert("RGB") if img.mode != "RGB" else img

    def __getitem__(self, ind):
        key = self.keys[ind]
        _, prev_dicom_id = key.split("_")

        tab_file = np.array(self.tab_files["ehr"][key])
        attn_mask = np.zeros(len(tab_file))
        image_file = self.img_meta.loc[prev_dicom_id, "jpg_fpath"]
        image_tensor = self.image_transform(PIL.Image.open(image_file))

        # Success
        return {
            "img": torch.FloatTensor(image_tensor),
            "tab": torch.FloatTensor(tab_file),
            "attn_mask": torch.BoolTensor(attn_mask),
        }

    def collate_fn(self, batch):
        embeddings = [torch.nn.functional.pad(b["tab"], (0, 0, 0, self.max_event_len - len(b["tab"]))) for b in batch]
        attn_mask = [
            torch.nn.functional.pad(b["attn_mask"], (0, self.max_event_len - len(b["attn_mask"])), value=1) for b in batch
        ]
        return {
            "img": torch.stack([b["img"] for b in batch]),
            "tab": torch.stack(embeddings),
            "attn_mask": torch.stack(attn_mask).bool(),
        }


class TabImageDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size,
        img_root_dir,
        img_meta_dir,
        tab_root_dir,
        tab_data_type="openai_filtered",
        max_event_len=1024,
        num_workers=0,
        image_size=256,
        resize_ratio=0.75,
        shuffle=False,
        phase="train",
    ):
        super().__init__()
        self.phase = phase
        self.img_root_dir = img_root_dir
        self.img_meta_dir = img_meta_dir
        self.tab_root_dir = tab_root_dir
        self.tab_data_type = tab_data_type
        self.max_event_len = max_event_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.resize_ratio = resize_ratio
        self.shuffle = shuffle

    def setup(self, stage=None):
        self.dataset = TabImageDataset(
            self.phase,
            image_size=self.image_size,
            resize_ratio=self.resize_ratio,
            shuffle=self.shuffle,
            img_root_dir=self.img_root_dir,
            img_meta_dir=self.img_meta_dir,
            tab_root_dir=self.tab_root_dir,
            tab_data_type=self.tab_data_type,
            max_event_len=self.max_event_len,
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            collate_fn=self.dataset.collate_fn,
        )
