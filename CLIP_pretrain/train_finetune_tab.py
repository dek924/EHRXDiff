import os
import sys
import clip
import torch
import datetime

torch.autograd.set_detect_anomaly(True)

from torch import nn
from argparse import ArgumentParser
from torchvision.models import resnet50
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from models import CustomTabViTCLIPWrapper
from dataset.tab_image_dm_transformer import TabImageDataModule


class TableClassifier(nn.Module):
    def __init__(self, n_embed=1536, feature_dim=768):
        super().__init__()
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(n_embed, 24, n_embed * 4, batch_first=True), 2)
        self.fc = nn.Linear(in_features=n_embed, out_features=feature_dim)

    def forward(self, emb, attn_mask, get_avg_emb=True):
        x = self.transformer(emb, src_key_padding_mask=attn_mask)
        x = self.fc(x)
        if get_avg_emb:
            x = x.sum(1) / (~attn_mask).float().sum(dim=1, keepdim=True)
        return x


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()


def main(args):
    seed_everything(42, workers=True)

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    now_name = now + args.expname if args.expname else now
    args.saved_dir = os.path.join(args.saved_dir, now_name)
    if not args.debug:
        wandb_logger = WandbLogger(project="cheff_v1", name=now_name, save_dir=args.saved_dir)
    else:
        wandb_logger = None

    checkpoint_callback = ModelCheckpoint(
        monitor="loss",  # or 'train_loss', ensure this matches your log name
        dirpath=os.path.join(args.saved_dir, "checkpoints"),  # Directory to save checkpoints
        filename="checkpoint-{epoch:02d}-{loss:.2f}",  # File name format
        save_top_k=1,  # Save all checkpoints
        save_last=True,
        save_on_train_epoch_end=True,  # Ensure saving happens at the end of the training epoch
    )

    tab_encoder = TableClassifier(n_embed=1536, feature_dim=768)
    model = CustomTabViTCLIPWrapper(
        tab_encoder,
        batch_size=args.batch_size,
        avg_word_embs=True,
    )

    dm = TabImageDataModule(
        batch_size=args.batch_size,
        img_root_dir=args.img_root_dir,
        img_meta_dir=args.img_meta_dir,
        tab_root_dir=args.tab_root_dir,
        tab_data_type=args.tab_data_type,
        max_event_len=args.max_event_len,
        num_workers=args.num_workers,
        image_size=args.image_size,
        resize_ratio=args.resize_ratio,
        shuffle=args.shuffle,
        phase="train" if not args.debug else "test",
    )
    trainer = Trainer.from_argparse_args(
        args,
        callbacks=[checkpoint_callback],
        precision=16,
        max_epochs=100,
        deterministic=True,
        strategy="ddp",
        logger=wandb_logger,
        accumulate_grad_batches=args.accumulate_grad_batches,
    )
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--saved_dir", type=str, default="./logs", help="Directory to save logs and checkpoints")
    parser.add_argument("--expname", type=str, default=None, help="Experiment name for logging and model saving")
    parser.add_argument("--img_root_dir", type=str, required=True, help="Directory containing preprocessed images")
    parser.add_argument("--img_meta_dir", type=str, required=True, help="Directory containing MIMIC-CXR-JPG metadata")
    parser.add_argument("--tab_root_dir", type=str, required=True, help="Directory containing tabular EHR data")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--image_size", type=int, default=256, help="Size of the images")
    parser.add_argument("--resize_ratio", type=float, default=0.75, help="Minimum size of images during random crop")
    parser.add_argument("--tab_data_type", type=str, default="openai_filtered", help="Type of tabular data preprocessing")
    parser.add_argument("--max_event_len", type=int, default=1024, help="Maximum length of event sequences in tabular data")
    parser.add_argument("--shuffle", type=bool, default=False, help="Whether to use shuffling during sampling")
    parser.add_argument("--debug", action="store_true", default=False, help="Whether to use debuging mode or not")

    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    torch.autograd.set_detect_anomaly(True)
    main(args)
