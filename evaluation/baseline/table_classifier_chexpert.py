import os
import h5py
import wandb
import random
import datetime
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from tqdm import tqdm
from pathlib import Path
from h5py import File
from torch.nn.functional import pad
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

LABEL_LIST = [
            "Atelectasis",
            "Cardiomegaly",
            "Consolidation",
            "Edema",
            "Enlarged Cardiomediastinum",
            "Fracture",
            "Lung Lesion",
            "Lung Opacity",
            "No Finding",
            "Pleural Effusion",
            "Pleural Other",
            "Pneumonia",
            "Pneumothorax",
            "Support Devices",
        ]

class CXRMIMICDataset(Dataset):
    def __init__(self, split, args):
        super().__init__()
        if args.debug:
            split = "valid"

        self.args = args
        self.split = split
        self.num_tab_event = 1024 

        # Load table data
        print(f"Load mimiciv-cxr-filtered_{split}_openai.h5...")
        self.embeddings = File(
            os.path.join(args.tab_root_dir, f"mimiciv-cxr-filtered_{split}_openai.h5"),
            driver="sec2",
        )

        # Load label data
        label_data = pd.read_csv(os.path.join(args.img_meta_dir, "mimic-cxr-2.0.0-chexpert.csv"))
        img_meta = pd.read_csv(os.path.join(args.img_meta_dir, "mimic-cxr-2.0.0-metadata.csv"))
        img_meta = img_meta[img_meta.ViewPosition.isin(["PA", "AP"])]
        img_meta = img_meta[["dicom_id", "subject_id", "study_id"]]

        labels = []
        for pathology in LABEL_LIST:
            if pathology in label_data:
                mask = label_data[pathology]
            labels.append(mask.values)
        labels = np.asarray(labels).T
        labels = labels.astype(np.float32)

        if split != "test":
            labels = np.nan_to_num(labels, 0)
            labels[labels == -1] = 0
        else:
            labels[labels == -1] = np.nan

        label_data["label"] = [list(row) for row in labels]

        meta_df = pd.read_csv(os.path.join(args.tab_root_dir, "mimiciv_cohort_meta.csv"), usecols=["subject_id", "hadm_id", "dicom_id", "study_id"])
        meta_df["data_key"] = meta_df["dicom_id"] 
        meta_df[["dicom_id", "prev_dicom_id"]] = meta_df["data_key"].str.split("_", expand=True)

        meta_df = meta_df.merge(img_meta[["study_id", "dicom_id"]].add_prefix("prev_"), on=["prev_dicom_id"], how="inner")
        meta_df = meta_df.merge(label_data[["study_id", "label"]], on="study_id")
        meta_df = meta_df.merge(label_data[["study_id", "label"]].add_prefix("prev_"), on="prev_study_id", how="left")
        meta_df["prev_label"] = meta_df["prev_label"].apply(lambda x: [0] * labels.shape[-1] if isinstance(x, float) and np.isnan(x) else x)
        meta_df = meta_df.set_index("data_key")
        print(meta_df.shape)

        # only use intersection
        df_keys = meta_df.index
        h5_keys = list(self.embeddings["ehr"].keys())
        self.keys = list(set(df_keys) & set(h5_keys))
        meta_df = meta_df[meta_df.index.isin(self.keys)]
        print("# of datakey:", len(self.keys))

        table_info = pd.DataFrame({"data_key": list(self.keys)})
        print(meta_df.shape)
        print("Init table info: ", table_info.shape)

        table_info = table_info.drop_duplicates().reset_index(drop=True)
        self.table_info = table_info
        self.meta_df = meta_df
        print("# of data:", len(self.table_info))

    def __len__(self):
        return len(self.table_info)

    def __getitem__(self, idx):
        sample = self.table_info.iloc[idx]
        data_key = sample["data_key"]
        df_row = self.meta_df.loc[data_key]

        embedding = np.array(self.embeddings["ehr"][data_key])[-self.num_tab_event :]
        prev_label = np.nan_to_num(np.array(df_row["prev_label"]), 0)
        label = np.array(df_row["label"])
        
        if self.args.use_prev_label:
            attn_mask = np.zeros(len(embedding) + 1)  # 1 for prev_label
        else:
            attn_mask = np.zeros(len(embedding))

        return {
            "data_key": data_key,
            "embedding": torch.FloatTensor(embedding),
            "prev_label": torch.FloatTensor(prev_label),
            "label": torch.FloatTensor(label),
            "attn_mask": torch.BoolTensor(attn_mask),
        }

    def collate_fn(self, batch):
        # Emb -> S x E
        embeddings = [pad(b["embedding"], (0, 0, 0, self.num_tab_event - len(b["embedding"]))) for b in batch]
        if self.args.use_prev_label:
            attn_mask = [pad(b["attn_mask"], (0, self.num_tab_event + 1 - len(b["attn_mask"])), value=1) for b in batch]
        else:
            attn_mask = [pad(b["attn_mask"], (0, self.num_tab_event - len(b["attn_mask"])), value=1) for b in batch]

        return {
            "embedding": torch.stack(embeddings),
            "prev_label": torch.stack([b["prev_label"] for b in batch]),
            "label": torch.stack([b["label"] for b in batch]),
            "attn_mask": torch.stack(attn_mask).bool(),
            "data_key": [b["data_key"] for b in batch]
        }


class TableClassifier(nn.Module):
    def __init__(self, pretrained_path, use_prev_label=False):
        super().__init__()
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(1536, 24, 1536 * 4, batch_first=True), 2)
        self.fc = nn.Linear(1536, 14)
        if use_prev_label:
            self.label_embs = nn.Parameter(torch.randn(14, 1536))
        
        state_dict = torch.load(pretrained_path, map_location="cpu")["state_dict"]
        state_dict = {k.replace("model.transformer.transformer.", ""): v for k, v in state_dict.items() if "model.transformer." in k}
        missing, unexpected = self.transformer.load_state_dict(state_dict, strict=False)
        try:
            assert len(missing) == 0 and len(unexpected) == 0
        except:
            print("missing: ", missing)
            print("unexpected: ", unexpected)

    def forward(self, emb, attn_mask, prev_label=None):
        # B, 14 x 14x1536 -> B x 1536
        if prev_label is not None:
            prev_label_emb = prev_label.unsqueeze(-1) * self.label_embs
            prev_label_emb = prev_label_emb.mean(dim=1, keepdim=True)
            emb = torch.cat([prev_label_emb, emb], dim=1)  # B, S+1, E

        x = self.transformer(emb, src_key_padding_mask=attn_mask)  # B, S+1, E5
        x = x.sum(1) / (~attn_mask).float().sum(dim=1, keepdim=True)
        x = self.fc(x)
        return x


class Trainer:
    def __init__(self, args):
        self.args = args
        self.model = TableClassifier(pretrained_path=args.pretrained_path, use_prev_label=args.use_prev_label)
        self.model = nn.DataParallel(self.model).to("cuda")
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        self.scheduler = None
        self.accumulation_steps = args.accumulation_steps

        if args.eval:
            testset = CXRMIMICDataset("test", args)
            self.test_loader = DataLoader(
                testset, batch_size=args.batch_size, shuffle=False, num_workers=8, collate_fn=testset.collate_fn
            )
        else:
            trainset = CXRMIMICDataset("train", args)
            validset = CXRMIMICDataset("valid", args)
            testset = CXRMIMICDataset("test", args)
            self.train_loader = DataLoader(
                trainset, batch_size=args.batch_size, shuffle=True, num_workers=8, collate_fn=trainset.collate_fn
            )
            self.valid_loader = DataLoader(
                validset, batch_size=args.batch_size, shuffle=False, num_workers=8, collate_fn=validset.collate_fn
            )
            self.test_loader = DataLoader(
                testset, batch_size=args.batch_size, shuffle=False, num_workers=8, collate_fn=testset.collate_fn
            )
        self.loss = nn.BCEWithLogitsLoss()

    def step(self, batch):
        embedding = batch["embedding"].to("cuda")
        prev_label = batch["prev_label"].to("cuda")
        attn_mask = batch["attn_mask"].to("cuda")
        label = batch["label"].to("cuda").float()
        
        if self.args.use_prev_label:
            pred = self.model(embedding, attn_mask, prev_label)
        else:
            pred = self.model(embedding, attn_mask)
        loss = self.loss(pred, label)

        return loss, pred, label, prev_label

    def train(self):
        best_val_auc = 0
        for epoch in range(self.args.max_epoch):
            train_loss = []
            self.optimizer.zero_grad()

            for i, batch in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
                loss, _, _, _ = self.step(batch)
                loss = loss / self.accumulation_steps

                loss.backward()

                if (i + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    train_loss.append(loss.item() * self.accumulation_steps)
                    if self.scheduler:
                        self.scheduler.step()

            if self.scheduler:
                self.scheduler.step()
            avg_train_loss = np.mean(train_loss)
            wandb.log({"train_loss": avg_train_loss}, step=epoch)
            print(f"Epoch {epoch} Train Loss: {avg_train_loss}")
            torch.save(self.model.state_dict(), os.path.join(self.args.save_dir, "lastest.pt"))

            valid_metrics = {"loss": [], "score": [], "label": []}
            with torch.no_grad():
                for batch in tqdm(self.valid_loader):
                    loss, pred, label, _ = self.step(batch)
                    valid_metrics["loss"].append(loss.item())

                    score = torch.sigmoid(pred).cpu().numpy()
                    label = label.cpu().numpy()

                    valid_metrics["score"].append(score)
                    valid_metrics["label"].append(label)

                label = np.concatenate(valid_metrics["label"], axis=0)
                score = np.concatenate(valid_metrics["score"], axis=0)
                auc = roc_auc_score(label, score, average=None)
                macro_auc = roc_auc_score(label, score, average="macro")
                if macro_auc > best_val_auc:
                    best_val_auc = macro_auc
                    torch.save(self.model.state_dict(), os.path.join(self.args.save_dir, "best_val_auc.pt"))

            wandb.log(
                {"valid_loss": np.mean(valid_metrics["loss"]), "valid_micro_auc": auc, "valid_macro_auc": macro_auc}, step=epoch
            )

            print(f"Epoch {epoch} Valid Loss: {np.mean(valid_metrics['loss'])}")
            print(f"Epoch {epoch} Valid AUC: {auc}")
            print(f"Epoch {epoch} Valid Macro AUC: {macro_auc}")

        with torch.no_grad():
            test_metrics = {"loss": [], "score": [], "label": []}
            for batch in tqdm(self.test_loader):
                loss, pred, label, _ = self.step(batch)
                test_metrics["loss"].append(loss.item())

                score = nn.functional.sigmoid(pred).cpu().numpy()
                label = label.cpu().numpy()

                test_metrics["score"].append(score)
                test_metrics["label"].append(label)

            label = np.concatenate(test_metrics["label"], axis=0)
            score = np.concatenate(test_metrics["score"], axis=0)

            auc = roc_auc_score(label, score, average=None)
            macro_auc = roc_auc_score(label, score, average="macro")

        wandb.log({"test_loss": np.mean(test_metrics["loss"]), "test_macro_auc": macro_auc})
        print(f"Test Loss: {np.mean(test_metrics['loss'])}")
        print(f"Test AUC: {auc}")
        print(f"Test Macro AUC: {macro_auc}")

        torch.save(self.model.state_dict(), os.path.join(self.args.save_dir, "lastest.pt"))

    def test(self, args):
        print(f"Load checkpoint from {args.checkpoint_dir}")
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, "best_val_auc.pt"))
        self.model.load_state_dict(checkpoint)

        total_pred = torch.FloatTensor([]).cuda(non_blocking=True)
        total_label = torch.FloatTensor([]).cuda(non_blocking=True)
        total_prev_label = torch.FloatTensor([]).cuda(non_blocking=True)

        with torch.no_grad():
            for batch in tqdm(self.test_loader):
                loss, pred, label, prev_label = self.step(batch)
                total_pred = torch.cat([total_pred, pred], dim=0)
                total_label = torch.cat([total_label, label], dim=0)
                total_prev_label = torch.cat([total_prev_label, prev_label], dim=0)

            total_pred = total_pred.cpu()
            total_label = total_label.cpu()
            total_prev_label = total_prev_label.cpu()
            total_prev_label[total_prev_label == -1] = 0

        if args.use_prev_label_for_eval:
            total_pred = total_prev_label.clone()

        print(total_pred.shape)
        print(total_label.shape)
        print(total_prev_label.shape)
        
        import datetime
        from collections import Counter
        save_file = "test" if not args.use_prev_label_for_eval else "test_prev_label"
        save_path = os.path.join(args.checkpoint_dir, f"{save_file}.txt")
        
        mask = ~total_label.isnan()
        total_test_acc = accuracy_score(total_label[mask], total_pred[mask].round())
        total_test_auroc_micro = roc_auc_score(total_label[mask], total_pred[mask], average="micro")
        total_test_auprc_micro = average_precision_score(total_label[mask], total_pred[mask], average="micro")
        with open(save_path, "w") as f:
            f.write(f"Logging time: {datetime.datetime.now()}\n")
            f.write(f"total_test_auroc_micro : " + "{:.3f}\n".format(round(total_test_auroc_micro, 3)))
            f.write(f"total_test_auprc_micro : " + "{:.3f}\n".format(round(total_test_auprc_micro, 3)))
            f.write(f"total_test_acc_micro   : " + "{:.3f}\n".format(round(total_test_acc * 100, 3)))

        test_output_attr = {}
        for diff_label in ["all", "diff", "same"]:
            for i, cxr_class in enumerate(LABEL_LIST):
                if diff_label == "all":
                    mask = ~total_label[:, i].isnan()
                    total_probs_attr = total_pred[mask, i]
                    total_labels_attr = total_label[mask, i]
                elif diff_label == "diff":
                    mask = (total_label[:, i] != total_prev_label[:, i]) & (~total_label[:, i].isnan())
                    total_probs_attr = total_pred[mask, i]
                    total_labels_attr = total_label[mask, i]
                elif diff_label == "same":
                    mask = (total_label[:, i] == total_prev_label[:, i]) & (~total_label[:, i].isnan())
                    total_probs_attr = total_pred[mask, i]
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

            competition_class = ["Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
            macro_acc_attr = np.mean([v["acc"] for v in test_output_attr.values()])
            macro_f1_attr = np.mean([v["f1"] for v in test_output_attr.values()])
            macro_auc_attr = np.mean([v["auc"] for v in test_output_attr.values()])
            macro_prc_attr = np.mean([v["prc"] for v in test_output_attr.values()])
            macro_4acc_attr = np.mean([v["acc"] for k, v in test_output_attr.items() if k in competition_class])
            macro_4f1_attr = np.mean([v["f1"] for k, v in test_output_attr.items() if k in competition_class])
            macro_4auc_attr = np.mean([v["auc"] for k, v in test_output_attr.items() if k in competition_class])
            macro_4prc_attr = np.mean([v["prc"] for k, v in test_output_attr.items() if k in competition_class])
            
            with open(save_path, "a") as f:
                f.write("-" * 150 + "\n")
                f.write(f"diff_label = {diff_label}" + "\n")
                f.write("-" * 150 + "\n")
                f.write(f"{len(test_output_attr)} attribute pairs" + "\n")
                for attr in sorted(LABEL_LIST):
                    auc = test_output_attr[attr]["auc"] if attr in test_output_attr else -1 
                    prc = test_output_attr[attr]["prc"] if attr in test_output_attr else -1 
                    f1 = test_output_attr[attr]["f1"] if attr in test_output_attr else -1 
                    acc = test_output_attr[attr]["acc"] if attr in test_output_attr else -1 
                    support = test_output_attr[attr]["support"] if attr in test_output_attr else -1
                    f.write(f"{attr:80s} | auc & prc & f1 & acc score: {auc:.3f} / {prc:.3f} / {f1:.3f} / {acc:.3f} | supp: {support}" + "\n")
                f.write("-" * 150 + "\n")
                f.write(f"""macro AUC & PRC & f1 & acc (all attr)  : {macro_auc_attr:.3f} / {macro_prc_attr:.3f} / {macro_f1_attr:.3f} / {macro_acc_attr * 100:.3f}""" + "\n")
                f.write(f"""macro AUC & PRC & f1 & acc (4 attr)    : {macro_4auc_attr:.3f} / {macro_4prc_attr:.3f} / {macro_4f1_attr:.3f} / {macro_4acc_attr * 100:.3f}""" + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tab_root_dir", type=str, required=True, help="Directory containing EHR tabular data")
    parser.add_argument("--img_meta_dir", type=str, required=True, help="Directory containing MIMIC-CXR-JPG metadata")
    parser.add_argument("--pretrained_path", type=str, default="./EHRXDiff/trained_models/clip_vit32_256_1024.ckpt")
    parser.add_argument("--random_seed", default=42, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--accumulation_steps", default=1, type=int)
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--use_prev_label", action="store_true")
    parser.add_argument("--use_prev_label_for_eval", action="store_true")
    parser.add_argument("--checkpoint_dir", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    trainer = Trainer(args)
    if args.eval:
        trainer.test(args)
    else:
        # fix the seed
        seed = args.random_seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        args.exp_name = now + "_" + args.exp_name if args.exp_name else now
        args.save_dir = os.path.join("./logs", args.exp_name)
        os.makedirs(os.path.join("./logs", args.exp_name))
        wandb.init(project="xray_tab_classification", name=args.exp_name, config=args)
        trainer.train()


if __name__ == "__main__":
    main()
