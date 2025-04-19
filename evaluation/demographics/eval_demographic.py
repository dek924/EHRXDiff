import os
import sys
import random
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
    def __init__(self, data_type, img_path, img_root_dir, img_meta_dir, tab_root_dir, tab_db_dir, transforms, use_prev_img_as_trg=False):
        self.data_type = data_type
        self.img_path = img_path
        self.transforms = transforms
        self.use_prev_img_as_trg = use_prev_img_as_trg
        img_meta = load_mimic_cxr_meta(img_meta_dir)
        img_meta["jpg_fpath"] = img_meta.apply(lambda x: img_root_dir + f"/p{str(x.subject_id)[:2]}/p{str(x.subject_id)}/s{str(x.study_id)}/{x.dicom_id}.jpg", axis=1)
        self.img_meta = img_meta[["dicom_id", "jpg_fpath"]]

        cohort = pd.read_csv(os.path.join(tab_root_dir, "mimiciv_cohort_meta.csv"))
        admission = pd.read_csv(os.path.join(tab_db_dir, "hosp", "admissions.csv.gz"), usecols=["subject_id", "hadm_id", "race"])

        cohort["data_key"] = cohort["dicom_id"]
        cohort[["dicom_id", "prev_dicom_id"]] = cohort["data_key"].str.split("_", expand=True)
        cohort = cohort.merge(admission, how="left", on=["subject_id", "hadm_id"])
        cohort = cohort[["subject_id", "hadm_id", "gender", "AGE", "dicom_id", "race"]]
        cohort = cohort.rename(columns={"AGE": "age"})
        race_dict = {"ASIAN": 0, "BLACK": 1, "WHITE": 2}

        def apply_race(race):
            for k, v in race_dict.items():
                if k in race:
                    return v
            return -1

        cohort["race"] = cohort["race"].apply(lambda x: apply_race(x))
        self.cohort = cohort

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, i):
        path = self.img_path[i]
        dicom_id = path.split("/")[-1].split(".")[0].split("_")[0]
        if self.use_prev_img_as_trg and (self.data_type == "pred"):
            prev_dicom_id = path.split("/")[-1].split(".")[0].split("_")[1]
            path = self.img_meta[self.img_meta.dicom_id == prev_dicom_id].jpg_fpath.values[0]

        img = skimage.io.imread(path)
        img = xrv.datasets.normalize(img, 255)  # convert 8-bit image to [-1024, 1024] range
        if len(img.shape) == 3:
            img = img.mean(2)[None, ...]  # Make single color channel
        else:
            img = img[None, ...]

        transform = TF.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)])

        img = transform(img)
        img = torch.from_numpy(img)
        race = self.cohort[self.cohort.dicom_id == dicom_id]["race"].unique()[0]
        age = self.cohort[self.cohort.dicom_id == dicom_id]["age"].unique()[0]
        assert len(self.cohort[self.cohort.dicom_id == dicom_id]["race"].unique() == 1)
        assert len(self.cohort[self.cohort.dicom_id == dicom_id]["age"].unique() == 1)

        return {"dicom_id": dicom_id, "img": img, "race": race, "age": age}

def main(args):
    set_seed(123)

    args.gt_path = get_gt_path(
        phase="test",
        img_root_dir=args.img_root_dir,
        tab_root_dir=args.tab_root_dir,
        mimic_cxr_dir=args.img_meta_dir,
        tab_data_type=args.tab_data_type
    )

    demographic_type = args.demographic_type
    if demographic_type == "age":
        model = xrv.baseline_models.riken.AgeModel()
    elif demographic_type == "race":
        model = xrv.baseline_models.emory_hiti.RaceModel()
    model = model.to(args.device)

    infer_root = args.infer_root
    args.gen_path = glob(os.path.join(infer_root, f"*.{args.img_ext}"))
    print("len of gt, gen:", len(args.gt_path), len(args.gen_path))
    assert len(args.gen_path) > 0

    gen_dataset = ImagePathDataset(data_type="pred", img_path=args.gen_path, img_root_dir=args.img_root_dir, img_meta_dir=args.img_meta_dir, tab_root_dir=args.tab_root_dir, tab_db_dir=args.tab_db_dir, transforms=TF.ToTensor(), use_prev_img_as_trg=args.use_prev_img_as_trg)
    if args.eval_only_inference:
        dataset_types = ["pred"]
        datasets = [gen_dataset]
    else:
        gt_dataset = ImagePathDataset(data_type="gt", img_path=args.gt_path, img_root_dir=args.img_root_dir, img_meta_dir=args.img_meta_dir, tab_root_dir=args.tab_root_dir, tab_db_dir=args.tab_db_dir, transforms=TF.ToTensor(), use_prev_img_as_trg=False)
        dataset_types = ["gt", "pred"]
        datasets = [gt_dataset, gen_dataset]

    for dataset_type, dataset in zip(dataset_types, datasets):
        print("dataset_type:", dataset_type)
        dataloader = torch.utils.data.DataLoader(
            dataset, shuffle=False, drop_last=False, batch_size=args.batch_size, num_workers=args.num_workers
        )

        total_output = []
        total_label = []

        with torch.no_grad():
            model.eval()
            for batch in tqdm(dataloader):
                img = batch["img"].to(args.device)
                label = batch[demographic_type]
                pred = model(img)
                total_output.append(pred.detach().cpu())
                total_label.append(label.detach())

        total_output = torch.cat(total_output, dim=0)
        total_label = torch.cat(total_label, dim=0)

        save_name = args.infer_root.split("/")[-1]
        save_name = f"{save_name}_{args.tab_data_type}"

        if not os.path.exists(os.path.join(os.path.dirname(args.infer_root), "logs", "demographic")):
            os.makedirs(os.path.join(os.path.dirname(args.infer_root), "logs", "demographic"))

        if demographic_type == "race":
            total_output = total_output[total_label != -1]
            total_label = total_label[total_label != -1]
            
            total_output = nn.Softmax(dim=1)(total_output).detach().cpu()
            total_test_acc = accuracy_score(total_label, torch.argmax(total_output, dim=1))
            total_test_auroc_macro = roc_auc_score(total_label, total_output, average="macro", multi_class="ovr", labels=[0,1,2])
            print("total_test_acc_micro:", round(total_test_acc * 100, 4))
            print("total_test_auroc_macro (ovr):", total_test_auroc_macro.round(4))

            save_name = f"{save_name}_{args.demographic_type}_test_auroc_{dataset_type}"
            if args.use_prev_img_as_trg:
                save_name += "_prev_img"
            save_path = os.path.join(os.path.dirname(args.infer_root), "logs", "demographic", f"{save_name}.txt")
            print("log write on {save_path}")
            with open(save_path, "w") as f:
                f.write(f"total_test_auroc_macro (ovr): " + "{:.3f}\n".format(round(total_test_auroc_macro, 3)))
                f.write(f"total_test_acc_micro        : " + "{:.3f}\n".format(round(total_test_acc * 100, 3)))

        elif demographic_type == "age":
            import scipy.stats as stats

            pp_val = stats.pearsonr(total_output.squeeze(), total_label)
            save_name = f"{save_name}_{args.demographic_type}_test_pearsonr_{dataset_type}"
            if args.use_prev_img_as_trg:
                save_name += "_prev_img"
            save_path = os.path.join(os.path.dirname(args.infer_root), "logs", "demographic", f"{save_name}.txt")
            print(f"log write on {save_path}")
            try:
                with open(save_path, "w") as f:
                    f.write(f"pearsonr:        " + "{:.3f}\n".format(round(pp_val.statistic, 3)))
                    f.write(f"pearsonr pvalue: " + "{:.3f}\n".format(round(pp_val.pvalue, 3)))
            except:
                with open(save_path, "w") as f:
                    f.write(f"pearsonr:        " + "{:.3f}\n".format(round(pp_val[0], 3)))
                    f.write(f"pearsonr pvalue: " + "{:.3f}\n".format(round(pp_val[1], 3)))

        else:
            raise NotImplementedError


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--img_root_dir", type=str, required=True, help="Directory containing preprocessed images")
    parser.add_argument("--img_meta_dir", type=str, required=True, help="Directory containing MIMIC-CXR-JPG metadata")
    parser.add_argument("--tab_root_dir", type=str, required=True, help="Directory containing tabular EHR data")
    parser.add_argument("--tab_db_dir", type=str, required=True, help="Directory containing MIMIC-IV data")
    parser.add_argument("--infer_root", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size to use")
    parser.add_argument("--img_ext", type=str, default="png", help="Image extension to search for in folder")
    parser.add_argument(
        "--num_workers", default=4, type=int, help=("Number of processes to use for data loading. Defaults to `min(8, num_cpus)`")
    )
    parser.add_argument("--eval_only_inference", action="store_true")
    parser.add_argument("--use_prev_img_as_trg", action="store_true")
    parser.add_argument("--demographic_type", type=str, default="race")
    parser.add_argument("--tab_data_type", type=str, default="filtered")

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main(args)
