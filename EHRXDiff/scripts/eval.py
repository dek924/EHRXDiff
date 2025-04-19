import os
import sys
import glob
import torch
import random
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
os.chdir(os.path.dirname(os.path.dirname(__file__)))

from tqdm import tqdm
from cheff.ldm.inference import CheffLDMTab2IAdaptor, CheffLDMTab2I_Imgonly
from EHRXDiff.cheff.custom_dataset import CXREHRDataset
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from torchvision.transforms.functional import to_pil_image


def eval(args):
    device = args.device
    sdm_path = args.sdm_path
    ae_path = args.ae_path
    batch_size = args.batch_size

    n_embed = args.n_embed
    num_layer = args.num_layer

    # Fix random seed
    seed = args.random_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ldm_module = CheffLDMTab2I_Imgonly if args.condition_type == "prev_img" else CheffLDMTab2IAdaptor
    cheff_tab2i = ldm_module(
                        model_path=sdm_path,
                        ae_path=ae_path,
                        device=device,
                        num_layer=args.num_layer,
                        max_event_len=args.max_event_len,
                        n_embed=args.n_embed,
                        context_dim=args.context_dim,
                        condition_feat_dim=args.condition_feat_dim,
                        condition_type=args.condition_type,
                        conditioning_key=args.conditioning_key,
                        tab_encoder_architecture=args.tab_encoder_architecture,
                    )

    transforms = Compose([Resize(256), ToTensor(), Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    print("Setting CXREHRDataset")
    test_dataset = CXREHRDataset(
            phase="test",
            img_root_dir=args.img_root_dir,
            img_meta_dir=args.img_meta_dir,
            tab_root_dir=args.tab_root_dir,
            tab_data_type=args.tab_data_type,
            max_event_len=args.max_event_len,
            transforms=transforms,
            debug=False,
        )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        collate_fn=test_dataset.collate_fn,
    )

    os.makedirs(args.save_dir, exist_ok=True)


    for data in tqdm(test_dataloader, total=len(test_dataloader)):
        batch = data
        data_key = batch["data_key"]
        batch_size = len(data_key)

        if args.condition_type == "table, prev_img":
            if cheff_tab2i.model.model.conditioning_key == "hybrid":
                cond = {
                    "c_concat": batch["prev_img"].to(device),
                    "c_crossattn": {
                        "prev_img": batch["prev_img"].to(device),
                        "attn_mask": batch["attn_mask"].to(device),
                        "table": batch["table"].to(device),
                    }
                }
            elif cheff_tab2i.model.model.conditioning_key == "crossattn":
                cond = {
                  "c_crossattn": {
                    "prev_img": batch["prev_img"].to(device),
                    "attn_mask": batch["attn_mask"].to(device),
                    "table": batch["table"].to(device),
                  }
                }
        elif args.condition_type == "table":
            cond = {
                  "c_crossattn": {
                    "attn_mask": batch["attn_mask"].to(device),
                    "table": batch["table"].to(device),
                  }
                }
        elif args.condition_type == "prev_img":
            cond = batch["prev_img"].to(device)
        elif args.condition_type == "text":
            cond = batch["text"]
        else:
            raise NotImplementedError

        start_cond = batch["prev_img"].to(device)

        img = cheff_tab2i.sample(
            conditioning=cond,
            sampling_steps=100,
            eta=1.0,
            decode=True,
            batch_size=len(batch["data_key"]),
            start_cond=start_cond,
        )

        img.clamp_(-1, 1)
        img = (img + 1) / 2

        for i, _img in enumerate(img):
            to_pil_image(_img.detach().cpu()).save(os.path.join(args.save_dir, f"{data_key[i]}.png"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--sdm_path", type=str, required=True, help="Path to the checkpoint for model evaluation")
    parser.add_argument("--ae_path", type=str, default="trained_models/cheff_autoencoder.pt")
    parser.add_argument("--img_root_dir", type=str, required=True, help="Directory containing preprocessed images")
    parser.add_argument("--img_meta_dir", type=str, required=True, help="Directory containing MIMIC-CXR-JPG metadata")
    parser.add_argument("--tab_root_dir", type=str, required=True, help="Directory containing tabular EHR data")
    parser.add_argument("--tab_data_type", type=str, default="openai_filtered", help="Type of tabular data preprocessing")
    parser.add_argument("--save_dir", type=str, default="outputs_diff")
    parser.add_argument(
        "--condition_type",
        type=str,
        default="table, prev_img",
        choices=["table, prev_img", "table", "prev_img"],
    )
    parser.add_argument("--conditioning_key", type=str, default="hybrid")
    parser.add_argument("--num_layer", type=int, default=2)
    parser.add_argument("--n_embed", type=int, default=1536)
    parser.add_argument("--max_event_len", type=int, default=1024)
    parser.add_argument("--tab_encoder_architecture", type=str, default="MultiModalTransformerAdaptor")
    parser.add_argument("--context_dim", type=int, default=768)
    parser.add_argument("--condition_feat_dim", type=int, default=1024)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--delete_ratio", type=float, default=0.0)

    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    eval(args)
