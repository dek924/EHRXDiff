import os
import h5py
import json
import math
import wandb
import random
import skimage
import argparse
import numpy as np
import pandas as pd

from glob import glob
from sklearn.metrics import *
from tqdm import tqdm as tqdm_base

import torch
import torchvision
import torch.nn as nn
import torchxrayvision as xrv
import torch.nn.functional as f
import torchvision.transforms as TF
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam, AdamW


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ImagePathDataset(Dataset):
    def __init__(self, tab_root_dir, img_paths, transforms):
        self.tab_root_dir = tab_root_dir
        self.img_paths = img_paths
        self.transforms = transforms
        cohort = pd.read_csv(os.path.join(tab_root_dir, "mimiciv_cohort_meta.csv"))
        cohort["data_key"] = cohort["dicom_id"]
        cohort[["dicom_id", "prev_dicom_id"]] = cohort["data_key"].str.split("_", expand=True)

        gender_dict = {"M": 0, "F": 1}
        cohort["gender_label"] = cohort.gender.map(gender_dict)
        self.cohort = cohort

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, i):
        path = self.img_paths[i]
        dicom_id = path.split("/")[-1].split(".")[0].split("_")[0]

        img = skimage.io.imread(path)
        img = xrv.datasets.normalize(img, 255)  # convert 8-bit image to [-1024, 1024] range
        if len(img.shape) == 3:
            img = img.mean(2)[None, ...]  # Make single color channel
        else:
            img = img[None, ...]

        transform = TF.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)])

        img = transform(img)
        img = torch.from_numpy(img)
        gender = self.cohort[self.cohort.dicom_id == dicom_id]["gender_label"].unique()[0]
        assert len(self.cohort[self.cohort.dicom_id == dicom_id]["gender_label"].unique() == 1)

        return {"img": img, "label": gender, "dicom_id": dicom_id, "img": img}


class MIMICCXRMetaDataset(Dataset):
    def __init__(self, img_root_dir, img_meta_dir, tab_root_dir, phase, transform, data_aug=None, use_prev_img_as_trg=False):
        self.phase = phase
        self.img_root_dir = img_root_dir
        self.img_meta_dir = img_meta_dir
        self.tab_root_dir = tab_root_dir
        self.transform = transform
        self.data_aug = data_aug
        self.transform = transform
        self.use_prev_img_as_trg = use_prev_img_as_trg

        with open(os.path.join(self.tab_root_dir, f"dataset_split_subject.json"), "r") as f:
            dataset_split = json.load(f)

        cxrmeta = pd.read_csv(os.path.join(img_meta_dir, "mimic-cxr-2.0.0-metadata.csv"), usecols=['dicom_id', 'subject_id', 'study_id', 'ViewPosition'])
        cxrmeta = cxrmeta[cxrmeta.ViewPosition.isin(['PA', 'AP'])]
        cxrmeta = cxrmeta.astype("str")
        if phase == "train":
            cxrmeta = cxrmeta[~cxrmeta.subject_id.isin(dataset_split["valid"] + dataset_split["test"])]
        else:
            cxrmeta = cxrmeta[cxrmeta.subject_id.isin(dataset_split[phase])]
            cohort = pd.read_csv(os.path.join(self.tab_root_dir, "mimiciv_cohort_meta.csv"))
            cohort["data_key"] = cohort["dicom_id"]
            cohort[["dicom_id", "prev_dicom_id"]] = cohort["data_key"].str.split("_", expand=True)
            cxrmeta = cxrmeta.merge(cohort[["subject_id", "gender"]].drop_duplicates(), on="subject_id")
            tab_inputs = h5py.File(os.path.join(self.tab_root_dir, f"mimiciv-cxr-filtered_{phase}_openai.h5"), "r")["ehr"]
            self.tab_keys = list(set(tab_inputs.keys()) & set(cohort.data_key.unique()))

        gender_dict = {"M": 0, "F": 1}
        cxrmeta["gender_label"] = cxrmeta.gender.map(gender_dict)
        cxrmeta["jpg_fpath"] = cxrmeta.apply(lambda x: f"{self.img_root_dir}/p{str(x.subject_id)[:2]}/p{str(x.subject_id)}/s{str(x.study_id)}/{x.dicom_id}.jpg", axis=1)
    
        self.df = cxrmeta
        print(f"Build dataset {len(cxrmeta)} samples ({cxrmeta.dicom_id.nunique()} images, {cxrmeta.subject_id.nunique()} patients)")
    
    def __len__(self):
        if self.phase == "train":
            return len(self.df)
        else:
            return len(self.tab_keys)

    def __getitem__(self, idx):
        if self.phase == "train":
            sample = self.df.iloc[idx]
        else:
            dicom_id_value = self.tab_keys[idx].split("_")[0]
            condition_indices = self.df.index[self.df['dicom_id'] == dicom_id_value].tolist()[0]
            sample = self.df.iloc[condition_indices]
            if self.use_prev_img_as_trg:
                prev_dicom_id_value = self.tab_keys[idx].split("_")[1]
                prev_condition_indices = self.df.index[self.df['dicom_id'] == prev_dicom_id_value].tolist()[0]
                prev_sample = self.df.iloc[prev_condition_indices]
                sample["jpg_fpath"] = prev_sample["jpg_fpath"]

        img = skimage.io.imread(sample["jpg_fpath"])
        img = xrv.datasets.normalize(img, 255)  # convert 8-bit image to [-1024, 1024] range
        if len(img.shape) == 3:
            img = img.mean(2)[None, ...]  # Make single color channel
        else:
            img = img[None, ...]
        
        img = self.transform(img)
        
        if self.data_aug:
            MAX_RAND_VAL = 2147483647
            seed = np.random.randint(MAX_RAND_VAL)
            random.seed(seed)
            torch.random.manual_seed(seed)
            img = self.data_aug(img)

        label = sample["gender_label"]

        return {"img": img, "label": label}
        

def tqdm(*args, **kwargs):
    if hasattr(tqdm_base, "_instances"):
        for instance in list(tqdm_base._instances):
            tqdm_base._decr_instances(instance)
    return tqdm_base(*args, **kwargs)


def get_optimizer(args, model):
    optim_name = args.optimizer
    if optim_name == "SGD":
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif optim_name == "Adam":
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(args.beta1, args.beta2))
    elif optim_name == "AdamW":
        optimizer = AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2),
            eps=1e-8,  # args.adam_epsilon,
        )
    else:
        raise NotImplementedError
    return optimizer


def get_model(model_name, num_class):
    if model_name == "densenet121-res224-all":
        model = xrv.models.DenseNet(weights="densenet121-res224-all")
        model.apply_sigmoid = True
        model.op_threshs = None
        model.classifier = nn.Linear(in_features=1024, out_features=num_class)
    else:
        raise ValueError()
    return model


def train(args):
    start_epoch = 0
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    ### Set wandb logging 
    if args.wandb:
        if args.wandb_id is None:
            args.wandb_id = wandb.util.generate_id()
            
        wandb.init(
            config=args,
            entity=args.wandb_entity_name,
            project=args.wandb_project_name,
            name="gender_cls",
            id=args.wandb_id,
            resume=args.wandb_resume,
            reinit=not args.wandb_resume,
        )

    # Data augmentation & Transform (img)
    data_aug = None
    if args.data_aug:
        data_aug = torchvision.transforms.Compose([
            xrv.datasets.ToPILImage(),
            torchvision.transforms.RandomAffine(10, 
                                                translate=(0.1, 0.1), 
                                                scale=(0.9, 1.1)),
            torchvision.transforms.ToTensor()
        ])
        print(data_aug)

    transform = TF.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(args.img_size)])

    # Dataset
    train_dataset = MIMICCXRMetaDataset(img_root_dir=args.img_root_dir, img_meta_dir=args.img_meta_dir, tab_root_dir=args.tab_root_dir, phase="train", transform=transform, data_aug=data_aug)
    valid_dataset = MIMICCXRMetaDataset(img_root_dir=args.img_root_dir, img_meta_dir=args.img_meta_dir, tab_root_dir=args.tab_root_dir, phase="valid", transform=transform, data_aug=data_aug)

    # Dataloader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=(args.device.type == "cuda"))
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(args.device.type == "cuda"))

    # Model
    model = get_model(args.model, num_class=2)
    model = nn.DataParallel(model).to(args.device)
    wandb.watch(model)

    # Optimizer
    args.num_training_steps = math.ceil(len(train_loader.dataset) / args.batch_size) * args.num_epochs
    optimizer = get_optimizer(args, model)
    criterion = torch.nn.CrossEntropyLoss()

    best_validauc = 0.0
    best_validacc = 0.0
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(start_epoch + 1, args.num_epochs + 1):
        train_avg_loss, train_acc = train_epoch(
            args=args,
            model=model,
            optimizer=optimizer,
            dataloader=train_loader,
            criterion=criterion,
            epoch=epoch,
            scaler=scaler,
        )
        print(f"Train accuracy & loss at epoch {epoch}: {train_acc:.2f}% / {train_avg_loss:.2f}")

        val_output = test_model(
            model=model,
            dataloader=valid_loader,
            criterion=criterion,
            device=args.device,
        )
        val_loss, val_auc, val_acc, val_ap = val_output["loss"], val_output["mean_auc"], val_output["acc"], val_output["mean_ap"]
        print(f"Valid Loss/AUC/acc/ap at epoch {epoch}: {val_loss:.2f} / {val_auc:.2f} / {val_acc:.2f} / {val_ap:.2f}")
        log_dict = {
            "Train/loss": train_avg_loss,
            "Train/acc": train_acc,
            "Val/loss": val_output["loss"],
            "Val/mean_auc": val_output["mean_auc"],
            "Val/acc": val_output["acc"],
            "Val/mean_ap": val_output["mean_ap"],
            "epoch": epoch,
        }

        save_dict = {
            "epoch": epoch,
            "model": model.module.state_dict(),
            "optimizer": optimizer,
            "best_validauc": best_validauc,
        }
        torch.save(save_dict, os.path.join(args.output_dir, "latest.pth.tar"))
        if val_output["mean_auc"] > best_validauc:
            best_validauc = val_output["mean_auc"]
            torch.save(save_dict, os.path.join(args.output_dir, "best_val_auc.pth.tar"))
            wandb.log({"best_validauc": best_validauc, "epoch": epoch}, commit=True)

        if val_output["acc"] > best_validacc:
            best_validacc = val_output["acc"]
            torch.save(save_dict, os.path.join(args.output_dir, "best_val_acc.pth.tar"))
            wandb.log({"best_validacc": best_validacc, "epoch": epoch}, commit=True)

        wandb.log(log_dict, commit=True)

    print("Done")


def test(args):
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Data augmentation & Transform (img)
    data_aug = None
    transform = TF.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(args.img_size)])

    # Dataset
    if args.infer_root:
        infer_root = args.infer_root
        save_name = args.infer_root.split("/")[-1]
        save_path = os.path.join(
                        os.path.dirname(args.infer_root), "logs", "demographic", f"{save_name}_gender_test_auroc.txt"
                    )
        print(f"log write on {save_path}")
        gen_paths = glob(os.path.join(infer_root, f"*.{args.img_ext}"))
        valid_dataset = None
        test_dataset = ImagePathDataset(tab_root_dir=args.tab_root_dir, img_paths=gen_paths, transforms=transform)
    else:        
        save_path = os.path.join(
                        args.output_dir, "logs", f"GT_gender_test_auroc.txt"
                    )
        if args.use_prev_img_as_trg: 
            save_path = save_path.replace("GT_gender_test_auroc", "GT_gender_test_auroc_previmg")
        print(f"log write on {save_path}")
        
        valid_dataset = MIMICCXRMetaDataset(img_root_dir=args.img_root_dir, img_meta_dir=args.img_meta_dir, tab_root_dir=args.tab_root_dir, phase="valid", transform=transform, data_aug=data_aug, use_prev_img_as_trg=args.use_prev_img_as_trg)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(args.device.type == "cuda"))
    
        test_dataset = MIMICCXRMetaDataset(img_root_dir=args.img_root_dir, img_meta_dir=args.img_meta_dir, tab_root_dir=args.tab_root_dir, phase="test", transform=transform, data_aug=data_aug, use_prev_img_as_trg=args.use_prev_img_as_trg)
        
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    # Dataloader
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(args.device.type == "cuda"))
    
    # Model
    model = get_model(args.model, num_class=2)
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint["model"])
    model = nn.DataParallel(model).to(args.device)

    # criterion
    criterion = torch.nn.CrossEntropyLoss()

    if not args.infer_root:
        val_output = test_model(
            model=model,
            dataloader=valid_loader,
            criterion=criterion,
            device=args.device,
        )
        val_loss, val_auc, val_acc, val_ap = val_output["loss"], val_output["mean_auc"], val_output["acc"], val_output["mean_ap"]

    test_output = test_model(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=args.device,
    )
    test_loss, test_auc, test_acc, test_ap = test_output["loss"], test_output["mean_auc"], test_output["acc"], test_output["mean_ap"]
    with open(save_path, "w") as f:
        if not args.infer_root:
            f.write(f"Validset")
            f.write(f"loss / macro auroc / sample acc / macro ap : {val_loss:.3f}% | {val_auc:.3f} | {val_acc:.3f} | {val_ap:.3f}\n")

        f.write(f"Testset")
        f.write(f"loss / macro auroc / sample acc / macro ap : {test_loss:.3f}% | {test_auc:.3f} | {test_acc:.3f} | {test_ap:.3f}")

    print("Done")


def test_model(model, dataloader, criterion, device, phase="trainval"):
    model.eval()

    loss_sum = 0.0
    correct_sum = 0
    num_samples = 0
    total_preds = torch.FloatTensor([]).to(device)
    total_labels = torch.LongTensor([]).to(device)
    with torch.no_grad():
        for batch_idx, samples in enumerate(dataloader):
            with torch.cuda.amp.autocast():
                images = samples["img"].to(device)
                targets = samples["label"].to(device)
                
                outputs = model(images)

                loss = criterion(outputs, targets)
                correct = (targets == outputs.max(dim=1)[1]).sum()

            num_samples += targets.shape[0]
            loss_sum += loss.item() * targets.shape[0]
            correct_sum += correct.item()
            total_preds = torch.cat([total_preds, outputs], dim=0)
            total_labels = torch.cat([total_labels, targets], dim=0)

        total_labels = total_labels.detach().cpu()
        total_preds = f.softmax(total_preds, dim=1)[:, 1].detach().cpu()
        mean_auc = roc_auc_score(total_labels, total_preds)
        mean_ap = average_precision_score(total_labels, total_preds)
    
    outputs = {
        "loss": loss_sum / num_samples, 
        "mean_auc": mean_auc,
        "mean_ap": mean_ap,
        "acc": correct_sum / num_samples,
    }
    return outputs


def train_epoch(args, model, dataloader, optimizer, criterion, epoch, scaler):
    model.train()
    device = args.device

    num_samples = 0
    train_loss_sum = 0.0
    train_acc_sum = 0.0
    t = tqdm(dataloader)

    for batch_idx, samples in enumerate(t):
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            images = samples["img"].float().to(device)
            targets = samples["label"].to(device)
            outputs = model(images)

            loss = criterion(outputs, targets)
            correct = (targets == outputs.max(dim=1)[1]).sum()

        num_samples += targets.shape[0]
        train_loss_sum += loss.item() * targets.shape[0]
        train_acc_sum += correct.item()

        t.set_description(f"Epoch {epoch} - Train - Loss = {train_loss_sum / num_samples:4.4f}")
        wandb.log({"Train/step_loss": train_loss_sum / num_samples, "lr": optimizer.param_groups[0]["lr"]}, step=(epoch - 1) * len(dataloader) + batch_idx, commit=True)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return train_loss_sum / num_samples, train_acc_sum / num_samples



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Finetune linear classification on MIMIC-CXR")
    
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--optimizer', default="AdamW", choices=["SGD", "Adam", "AdamW"])
    parser.add_argument("--num_epochs", default=100, type=int, help="Number of epochs of training.")
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.999, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--num_workers", default=8, type=int)

    parser.add_argument("--model", default="densenet121-res224-all", type=str)
    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument("--img_root_dir", type=str, required=True, help="Directory containing preprocessed images")
    parser.add_argument("--img_meta_dir", type=str, required=True, help="Directory containing MIMIC-CXR-JPG metadata")
    parser.add_argument("--tab_root_dir", type=str, required=True, help="Directory containing tabular EHR data")
    parser.add_argument("--output_dir", default="./results", type=str)
    parser.add_argument("--data_aug", dest="data_aug", action="store_true", help="use data augmentation or not")
    
    ## wandb logging
    parser.add_argument("--wandb", action="store_true", default=False, help="Whether to use wandb logging")
    parser.add_argument("--wandb_entity_name", type=str, default="")
    parser.add_argument("--wandb_project_name", type=str, default="")
    parser.add_argument("--wandb_id", type=str, default=None)
    parser.add_argument("--wandb_resume", action="store_true", default=False, help="Whether to allow wandb resume")

    # Evaluation
    parser.add_argument("--eval", action="store_true", help="ONly eval")
    parser.add_argument("--use_prev_img_as_trg", action="store_true", help="use prev img as target")
    parser.add_argument("--img_ext", type=str, default="png", help="Image extension to search for in folder")
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoint/best_val_auc.pth.tar")
    parser.add_argument("--infer_root", type=str, default=None)

    args, unknown = parser.parse_known_args()
    print(unknown)
    
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.eval:
        test(args)
    else:
        train(args)
        test(args)