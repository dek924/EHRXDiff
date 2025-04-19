import os
import sys
import h5py
import random
import datetime

import skimage
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torchxrayvision as xrv
import torchvision.transforms as TF

from glob import glob
from tqdm import tqdm
from sklearn.metrics import *
from collections import Counter
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datautils import load_mimic_cxr_meta, get_gt_path


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, datatype, img_path, mimic_cxr_dir, img_root_dir, tab_root_dir, use_prev_img_as_trg=False, transforms=None):
        self.datatype = datatype
        self.img_path = img_path
        self.mimic_cxr_dir = mimic_cxr_dir
        self.img_root_dir = img_root_dir
        self.tab_root_dir = tab_root_dir
        self.use_prev_img_as_trg = use_prev_img_as_trg
        self.transforms = transforms
        assert self.datatype in ["gt", "pred"]


        self.img_meta = load_mimic_cxr_meta(self.mimic_cxr_dir)
        self.img_meta["jpg_fpath"] = self.img_meta.apply(lambda x: f"/p{str(x.subject_id)[:2]}/p{str(x.subject_id)}/s{str(x.study_id)}/{x.dicom_id}.jpg", axis=1)
        self.img_meta["jpg_fpath"] = img_root_dir + self.img_meta["jpg_fpath"]

        tab_inputs = h5py.File(os.path.join(self.tab_root_dir, "mimiciv-cxr-filtered_test_openai.h5"), "r")["ehr"]
        meta_df = pd.read_csv(os.path.join(self.tab_root_dir, "mimiciv_cohort_meta.csv"), usecols=["subject_id", "hadm_id", "dicom_id", "study_id"])
        meta_df["data_key"] = meta_df["dicom_id"]
        meta_df[["dicom_id", "prev_dicom_id"]] = meta_df["data_key"].str.split("_", expand=True)
        meta_df = pd.merge(meta_df, self.img_meta[["study_id", "dicom_id"]].add_prefix("prev_"), on="prev_dicom_id", how="inner")
        meta_df = meta_df[meta_df.data_key.isin(tab_inputs.keys())]

        label_path = os.path.join(os.path.dirname(self.mimic_cxr_dir), "mimic-cxr-2.0.0-chexpert.csv")
        self.mimic_cols = [
            "Atelectasis",
            "Consolidation",
            "Pneumothorax",
            "Edema",
            "Pleural Effusion",
            "Pneumonia",
            "Cardiomegaly",
            "Lung Lesion",
            "Fracture",
            "Lung Opacity",
            "Enlarged Cardiomediastinum",
        ]
        self.mimic_cols = sorted(self.mimic_cols)
        label_data = pd.read_csv(label_path)
 
        labels = []
        for pathology in self.mimic_cols:
            if pathology in label_data:
                mask = label_data[pathology]
            labels.append(mask.values)
            
        labels = np.asarray(labels).T
        labels = labels.astype(np.float32)

        # Make all the -1 values into nans to keep things simple
        labels[labels == -1] = np.nan
        print(label_data.shape)
        print(labels.shape)
        label_data["label"] = [list(row) for row in labels]

        meta_df = meta_df.merge(self.img_meta[["dicom_id", "jpg_fpath"]], on="dicom_id")
        meta_df = meta_df.merge(self.img_meta[["dicom_id", "jpg_fpath"]].add_prefix("prev_"), on="prev_dicom_id")
        meta_df = meta_df.merge(label_data[["study_id", "label"]], on="study_id")
        meta_df = meta_df.merge(label_data[["study_id", "label"]].add_prefix("prev_"), on="prev_study_id", how="left")
        self.meta_df = meta_df
        self.img_path = [_path for _path in self.img_path if _path.split("/")[-1].split(".")[0].split("_")[0] in meta_df.dicom_id.unique()]

        print(f"Evaluate {len(self.meta_df)} sample, {len(self.img_path)} images, {self.meta_df.dicom_id.nunique()} unique dicoms")

    def __len__(self):
        if self.datatype == "gt":
            return len(self.meta_df)
        else:
            return len(self.img_path)

    def __getitem__(self, idx):
        if self.datatype == "gt":
            sample = self.meta_df.iloc[idx]
            if self.use_prev_img_as_trg:
                path = sample["prev_jpg_fpath"]
            else:
                path = sample["jpg_fpath"]
            data_key = sample["data_key"]
        else:
            path = self.img_path[idx]
            data_key = path.split("/")[-1].split(".")[0]
            sample = self.meta_df[self.meta_df.data_key == data_key].iloc[0]

        img = skimage.io.imread(path)
        img = xrv.datasets.normalize(img, 255)  # convert 8-bit image to [-1024, 1024] range
        if len(img.shape) == 3:
            img = img.mean(2)[None, ...]  # Make single color channel
        else:
            img = img[None, ...]

        transform = TF.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)])

        img = transform(img)
        img = torch.from_numpy(img)
        
        label = np.array(sample["label"])
        prev_label = np.array(sample["prev_label"])

        return {"img": img, "label": label, "prev_label": prev_label}


def test(args, model, dataset):
    dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=False, drop_last=False, batch_size=args.batch_size, num_workers=args.num_workers
    )

    total_output = []
    total_label = []
    total_prev_label = []

    with torch.no_grad():
        model.eval()
        for batch in tqdm(dataloader):
            img = batch["img"].to(device)
            label = batch["label"]
            prev_label = batch["prev_label"]
            pred = model(img)
            total_output.append(pred.detach().cpu())
            total_label.append(label.detach())
            total_prev_label.append(prev_label.detach())

    total_output = torch.cat(total_output, dim=0)
    total_label = torch.cat(total_label, dim=0)
    total_prev_label = torch.cat(total_prev_label, dim=0)
    total_prev_label = torch.nan_to_num(total_prev_label, nan=0.0)
    total_output = nn.Softmax()(total_output)
    
    if args.use_prev_label_for_eval:
        total_output = total_prev_label
    else:
        if args.backbone == "densenet121-res224-all":
            all_targets = model.targets
            map_dict = {"Effusion": "Pleural Effusion"}
            all_targets = [map_dict[_label] if _label in map_dict else _label for _label in all_targets]
            eval_targets = dataset.mimic_cols
            target_indices = [all_targets.index(label) for label in eval_targets if label in all_targets]
            total_output = total_output[:, target_indices]

    if dataset.datatype == "gt":
        save_dir = "./log"
        if args.use_prev_img_as_trg:
            save_name = f"{args.backbone}_test_auroc_{dataset.datatype}_previmg"
        else:
            save_name = f"{args.backbone}_test_auroc_{dataset.datatype}"
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_name = args.infer_root.split("/")[-1]
        if not os.path.exists(os.path.join(os.path.dirname(args.infer_root), "logs")):
            os.makedirs(os.path.join(os.path.dirname(args.infer_root), "logs"))
        save_dir = os.path.join(os.path.dirname(args.infer_root), "logs")
        save_name = f"{save_name}_{args.backbone}_test_auroc_{dataset.datatype}"

    if args.use_prev_label_for_eval:
        save_name += "_prev_label"

    save_path = os.path.join(save_dir, f"{save_name}.txt")
    print(f"log write on {save_path}")
    mask = ~total_label.isnan()
    total_test_acc = accuracy_score(total_label[mask], total_output[mask].round())
    total_test_auroc_micro = roc_auc_score(total_label[mask], total_output[mask], average="micro")
    total_test_auprc_micro = average_precision_score(total_label[mask], total_output[mask], average="micro")
    with open(save_path, "w") as f:
        f.write(f"Logging time: {datetime.datetime.now()}\n")
        f.write(f"total_test_auroc_micro : " + "{:.3f}\n".format(round(total_test_auroc_micro, 3)))
        f.write(f"total_test_auprc_micro : " + "{:.3f}\n".format(round(total_test_auprc_micro, 3)))
        f.write(f"total_test_acc_micro   : " + "{:.3f}\n".format(round(total_test_acc * 100, 3)))


    test_output_attr = {}
    for diff_label in ["all", "diff", "same"]:
        for i, cxr_class in enumerate(dataset.mimic_cols):
            if diff_label == "all":
                mask = ~total_label[:, i].isnan()
                total_probs_attr = total_output[mask, i]
                total_labels_attr = total_label[mask, i]
            elif diff_label == "diff":
                mask = (total_label[:, i] != total_prev_label[:, i]) & (~total_label[:, i].isnan())
                total_probs_attr = total_output[mask, i]
                total_labels_attr = total_label[mask, i]
            elif diff_label == "same":
                mask = (total_label[:, i] == total_prev_label[:, i]) & (~total_label[:, i].isnan())
                total_probs_attr = total_output[mask, i]
                total_labels_attr = total_label[mask, i]
            else:
                raise NotImplementedError

            try:
                test_output_attr[cxr_class] = {
                    "acc": accuracy_score(total_labels_attr, total_probs_attr.round()),
                    "f1": f1_score(total_labels_attr, total_probs_attr >= 0.5, average='micro'),
                    "auc": roc_auc_score(total_labels_attr, total_probs_attr),
                    "prc": average_precision_score(total_labels_attr, total_probs_attr, average="micro")
                }
                test_output_attr[cxr_class]["support"] = [v for (k, v) in sorted(dict(Counter(total_labels_attr.numpy())).items(), key=lambda item: item[0])]

            except ValueError:
                test_output_attr[cxr_class] = {
                    "acc": accuracy_score(total_labels_attr, total_probs_attr.round()),
                    "f1": 0, "auc": 0, "prc": 0
                }
                test_output_attr[cxr_class]["support"] = [v for (k, v) in sorted(dict(Counter(total_labels_attr.numpy())).items(), key=lambda item: item[0])]

        if diff_label == "all":
            mask = ~total_label.isnan()
        elif diff_label == "diff":
            mask = (~total_label.isnan()) & (total_label != total_prev_label)
        elif diff_label == "same":
            mask = (~total_label.isnan()) & (total_label == total_prev_label)
        else:
            raise NotImplementedError

        total_test_acc = accuracy_score(total_label[mask], total_output[mask].round())
        total_test_auroc_micro = roc_auc_score(total_label[mask], total_output[mask], average="micro")
        total_test_auprc_micro = average_precision_score(total_label[mask], total_output[mask], average="micro")

        macro_acc_attr = np.mean([v["acc"] for v in test_output_attr.values()])
        macro_f1_attr = np.mean([v["f1"] for v in test_output_attr.values()])
        macro_auc_attr = np.mean([v["auc"] for v in test_output_attr.values()])
        macro_prc_attr = np.mean([v["prc"] for v in test_output_attr.values()])
        
        with open(save_path, "a") as f:
            f.write("-" * 150 + "\n")
            f.write(f"diff_label = {diff_label}" + "\n")
            f.write("-" * 150 + "\n")
            f.write(f"{len(test_output_attr)} attribute pairs" + "\n")
            for attr in sorted(dataset.mimic_cols):
                auc = test_output_attr[attr]["auc"] if attr in test_output_attr else -1 
                prc = test_output_attr[attr]["prc"] if attr in test_output_attr else -1 
                f1 = test_output_attr[attr]["f1"] if attr in test_output_attr else -1 
                acc = test_output_attr[attr]["acc"] if attr in test_output_attr else -1 
                support = test_output_attr[attr]["support"] if attr in test_output_attr else -1
                f.write(f"{attr:80s} | auc & prc & f1 & acc score: {auc:.3f} / {prc:.3f} / {f1:.3f} / {acc:.3f} | supp: {support}" + "\n")
            f.write("-" * 150 + "\n")
            f.write(f"""macro AUC & PRC & f1 & acc (all attr)  : {macro_auc_attr:.3f} / {macro_prc_attr:.3f} / {macro_f1_attr:.3f} / {macro_acc_attr * 100:.3f}""" + "\n")
            f.write(f"""micro AUC & PRC & acc (all attr)    : {total_test_auroc_micro:.3f} / {total_test_auprc_micro:.3f} / {total_test_acc * 100:.3f}""" + "\n")


def main(args):
    args.gt_path = get_gt_path(
        phase="test",
        img_root_dir=args.img_root_dir,
        tab_root_dir=args.tab_root_dir,
        mimic_cxr_dir=args.img_meta_dir,
        tab_data_type="filtered"
    )

    model = xrv.models.DenseNet(weights="densenet121-res224-all").to(device)
    if args.infer_root is not None:
        print("dataset_type: pred")
        print("prediction dataset load from : ", args.infer_root)
        infer_root = args.infer_root
        args.gen_path = glob(os.path.join(infer_root, f"*.{args.img_ext}"))
        print("len of gt, gen:", len(args.gt_path), len(args.gen_path))

        if len(args.gen_path) == 0:
            raise NotImplementedError
        test_dataset = ImagePathDataset("pred", args.gen_path, mimic_cxr_dir=args.img_meta_dir, img_root_dir=args.img_root_dir, tab_root_dir=args.tab_root_dir, transforms=TF.ToTensor())
    else:
        print("dataset_type: gt")
        test_dataset = ImagePathDataset("gt", args.gt_path, mimic_cxr_dir=args.img_meta_dir, img_root_dir=args.img_root_dir, tab_root_dir=args.tab_root_dir, transforms=TF.ToTensor(), datatype="gt", use_prev_img_as_trg=args.use_prev_img_as_trg)
            
    print(args.backbone)
    test(args, model, test_dataset)

if __name__ == "__main__":

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--infer_root",
        type=str,
        default=None,
    )
    parser.add_argument("--img_root_dir", type=str, required=True, help="Directory containing preprocessed images")
    parser.add_argument("--img_meta_dir", type=str, required=True, help="Directory containing MIMIC-CXR-JPG metadata")
    parser.add_argument("--tab_root_dir", type=str, required=True, help="Directory containing tabular EHR data")
    parser.add_argument("--backbone", type=str, default="densenet121-res224-all")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size to use")
    parser.add_argument("--img_ext", type=str, default="png", help="Image extension to search for in folder")
    parser.add_argument("--num_workers", default=4, type=int, help=("Number of processes to use for data loading"))
    parser.add_argument("--use_prev_label_for_eval", action="store_true")
    parser.add_argument("--use_prev_img_as_trg", action="store_true")
    
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    
    set_seed(42)
    args = parser.parse_args()

    main(args)
