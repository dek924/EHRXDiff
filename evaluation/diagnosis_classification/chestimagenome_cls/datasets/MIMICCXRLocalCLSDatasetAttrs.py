import os
import h5py
import json
import torch
import random
import numpy as np
import pandas as pd
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode


def load_tab_h5py_file(tab_data_type, phase, tab_root_dir):
    if "openai" in tab_data_type:
        if tab_data_type == "openai_filtered":
            file_name = f"mimiciv-cxr-filtered_{phase}_openai.h5"
        elif tab_data_type == "openai":
            file_name = f"mimiciv-cxr_{phase}_openai.h5"
        else:
            raise ValueError
    else:
        file_name = f"mimiciv-cxr-{tab_data_type}_{phase}.h5"

    tab_inputs = h5py.File(os.path.join(tab_root_dir, file_name), "r", driver='sec2')
    print("Load ", os.path.join(tab_root_dir, file_name))

    tab_input_key = tab_inputs['ehr'].keys()

    return tab_inputs, tab_input_key


class MIMICCXRLocalCLSDatasetAttrs(Dataset):
    def __init__(self, args, phase="train", data_aug=None, debug=False, infer_root=None):
        super(MIMICCXRLocalCLSDatasetAttrs, self).__init__()
        if phase == "val":
            phase = "valid"
        if debug:
            phase = "valid"

        self.args = args
        self.phase = phase

        print(f"Image root: {args.imgroot}")

        data_path = os.path.join(args.dataroot, "reference_exp_attr_level", f"{phase}_ref.json")
        data_df = pd.DataFrame(json.load(open(data_path)))

        print(f"Load dataset {data_path}", data_df.shape)
        mimic_meta = pd.read_csv(os.path.join(args.imgmetaroot, "mimic-cxr-2.0.0-metadata.csv"))
        mimic_meta = mimic_meta[mimic_meta.ViewPosition.isin(["PA", "AP"])]
        mimic_meta = mimic_meta.rename(columns={"dicom_id": "image_id"})

        def _build_image_fpath(row):
            pid = str(int(row["subject_id"]))
            sid = str(int(row["study_id"]))
            iid = str(row["image_id"].split(".")[0])
            return f"{args.imgroot}/p{pid[:2]}/p{pid}/s{sid}/{iid}.jpg"
        data_df["image_path"]  = data_df.apply(_build_image_fpath, axis=1) 

        if phase == "test":
            cohort = pd.read_csv(os.path.join(args.dataroot, "mimiciv_cohort_meta.csv"), usecols=['subject_id', 'hadm_id', 'dicom_id', 'study_id', 'prev_dicom_id', 'StudyDateTime', 'prev_StudyDateTime']).drop_duplicates()
            cohort["data_key"] = cohort["dicom_id"]
            cohort[["image_id", "prev_image_id"]] = cohort["data_key"].str.split("_", expand=True)

            print(data_df.shape)
            data_df = data_df.merge(cohort, on=['subject_id', 'study_id', 'image_id'], how="inner")
            print(data_df.shape)
            data_df = data_df.drop_duplicates()
            print(data_df.shape)

            # Load test data
            _, tab_input_key = load_tab_h5py_file(args.tab_data_type, "test", args.dataroot)
            data_df = data_df[data_df.data_key.isin(tab_input_key)]

            if args.prev_img_as_trg:
                def _build_prev_image_fpath(row):
                    pid = str(int(row["subject_id"]))
                    sid = str(int(row["prev_study_id"]))
                    iid = str(row["prev_image_id"].split(".")[0])
                    return f"{args.imgroot}/p{pid[:2]}/p{pid}/s{sid}/{iid}.jpg"

                data_df = pd.merge(data_df, mimic_meta[["study_id", "image_id"]].rename(columns={"image_id": "prev_image_id", "study_id": "prev_study_id"}), on="prev_image_id").drop_duplicates()
                data_df["prev_image_path"]  = data_df.apply(_build_prev_image_fpath, axis=1) 

            # if args.eval_prev_data:
            print("Add prev label")
            print(data_df.shape)
            _data_df = pd.DataFrame(json.load(open(data_path)))
            data_df = data_df.merge(_data_df[["image_id", "attribute", "relation"]].rename(columns={"image_id": "prev_image_id", "relation": "prev_relation"}), on=["prev_image_id", "attribute"], how="left").drop_duplicates()
            data_df["prev_relation"] = data_df["prev_relation"].fillna(0)
            print(data_df.shape)
            
        self.data_df = data_df.drop_duplicates().reset_index(drop=True)
        self.transform = torch.nn.Sequential(
            transforms.Resize([224, 224]),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        )
        self.infer_root = infer_root
        self.data_aug = data_aug
        self.attr_pool = self.data_df.attribute.unique()
        print(f"Build upperbound dataset {len(data_df)} samples ({data_df.image_id.nunique()} images, {data_df.attribute.nunique()} attributes)")

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        sample = self.data_df.iloc[index]
        if (self.infer_root is not None) & (self.phase == "test"):
            image = read_image(os.path.join(self.infer_root, sample["data_key"] + ".png"), mode=ImageReadMode.RGB)
            sample = self.data_df.iloc[index]
        else:
            sample = self.data_df.iloc[index]
            if self.args.prev_img_as_trg:
                image = read_image(sample["prev_image_path"], mode=ImageReadMode.RGB)
            else:
                image = read_image(sample["image_path"], mode=ImageReadMode.RGB)
        image = self.transform(image)

        if self.data_aug is not None:
            transform_seed = np.random.randint(2147483647)
            random.seed(transform_seed)
            image = self.data_aug(image)

        if self.args.eval_prev_data:
            return {"img": image, "label": sample["relation"], "attribute": sample["attribute"], "prev_label": sample["prev_relation"]}
        return {"img": image, "label": sample["relation"], "attribute": sample["attribute"]}

    def __len__(self):
        return len(self.data_df)

