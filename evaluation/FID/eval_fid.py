import os
import sys
import torch
import random
import skimage
import numpy as np
import torchxrayvision as xrv
import torchvision.transforms as TF

from tqdm import tqdm
from glob import glob
from PIL import Image
from scipy import linalg
from models.inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datautils import get_gt_path, load_mimic_cxr_meta

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, backbone, img_root_dir, img_meta_dir, transforms=None, use_prev_img_as_trg=False):
        self.files = files
        self.backbone = backbone
        self.transforms = transforms
        self.use_prev_img_as_trg = use_prev_img_as_trg

        img_meta = load_mimic_cxr_meta(img_meta_dir)
        img_meta["jpg_fpath"] = img_meta.apply(lambda x: img_root_dir + f"/p{str(x.subject_id)[:2]}/p{str(x.subject_id)}/s{str(x.study_id)}/{x.dicom_id}.jpg", axis=1)
        img_meta = img_meta[["dicom_id", "jpg_fpath"]]
        self.img_meta = img_meta

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        if self.use_prev_img_as_trg:
            assert len(path.split("/")[-1].split(".")[0].split("_")) == 2 # only for prediction
            prev_dicom_id = path.split("/")[-1].split(".")[0].split("_")[1]
            path = self.img_meta[self.img_meta.dicom_id == prev_dicom_id].jpg_fpath.values[0]

        if "xrv" in self.backbone:
            img = skimage.io.imread(path)
            img = xrv.datasets.normalize(img, 255)  # convert 8-bit image to [-1024, 1024] range
            if len(img.shape) == 3:
                img = img.mean(2)[None, ...]  # Make single color channel
            else:
                img = img[None, ...]

            transform = TF.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)])

            img = transform(img)
            img = torch.from_numpy(img)
        else:
            preprocess = Compose([
                Resize(299),
                CenterCrop(299),
                ToTensor(),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

            img = Image.open(path).convert('RGB')
            img = preprocess(img)

        return img


def get_activations(args, files, model, batch_size=50, dims=2048, device="cpu", num_workers=1, backbone=None, use_prev_img_as_trg=False):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    print("total len of files:", len(files))
    if batch_size > len(files):
        print(("Warning: batch size is bigger than the data size. " "Setting batch size to data size"))
        batch_size = len(files)

    dataset = ImagePathDataset(files, backbone, args.img_root_dir, args.img_meta_dir, transforms=TF.ToTensor(), use_prev_img_as_trg=use_prev_img_as_trg)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers
    )

    pred_arr = np.empty((len(files), dims))

    start_idx = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = batch.to(device)

            with torch.no_grad():
                if dims == 1024:
                    pred = model.features(batch)
                elif dims == 2048:
                    pred = model(batch)[0]  # 50, 2048, 1, 1

            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred = pred.squeeze(3).squeeze(2).cpu().numpy()

            pred_arr[start_idx : start_idx + pred.shape[0]] = pred

            start_idx = start_idx + pred.shape[0]

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ("fid calculation produces singular product; " "adding %s to diagonal of cov estimates") % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_activation_statistics(args, files, model, batch_size=50, dims=2048, device="cpu", num_workers=1, backbone=None, use_prev_img_as_trg=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(args, files, model, batch_size, dims, device, num_workers, backbone, use_prev_img_as_trg)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def compute_statistics_of_path(args, path, model, batch_size, dims, device, num_workers, backbone, use_prev_img_as_trg):
    m, s = calculate_activation_statistics(args, path, model, batch_size, dims, device, num_workers, backbone, use_prev_img_as_trg)

    return m, s


def calculate_fid_given_paths(args, gt_path, gen_path, batch_size, device, backbone, dims, num_workers=1):
    """Calculates the FID of two paths"""

    if backbone == "xrv_densenet_224_all":
        model = xrv.models.DenseNet(weights="densenet121-res224-all").to(device)
        model.eval()

    elif backbone == "inceptionv3":
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        model = InceptionV3([block_idx]).to(device)
    else:
        raise NotImplementedError

    m1, s1 = compute_statistics_of_path(args, gt_path, model, batch_size, dims, device, num_workers, backbone, use_prev_img_as_trg=False)
    m2, s2 = compute_statistics_of_path(args, gen_path, model, batch_size, dims, device, num_workers, backbone, use_prev_img_as_trg=args.use_prev_img_as_trg)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value


def main(args):
    set_seed(123)

    if args.backbone in ["xrv_densenet_224_all"]:
        args.dims = 1024
    else:
        args.dims = 2048

    args.gt_path = get_gt_path(
        phase="test",
        img_root_dir=args.img_root_dir,
        tab_root_dir=args.tab_root_dir,
        mimic_cxr_dir=args.img_meta_dir,
        tab_data_type=args.tab_data_type
    )

    infer_root = args.infer_root
    save_name = infer_root.split("/")[-1]
    os.makedirs(os.path.join(os.path.dirname(infer_root), "logs", "fid"), exist_ok=True)
    log_name = f"{save_name}_{args.tab_data_type}_log_{args.backbone}"
    if args.use_prev_img_as_trg:
        log_name += "_prev_img"
    log = open(os.path.join(os.path.dirname(infer_root), "logs", "fid", f"{log_name}_fid.txt"), "w")
    log.write("=" * 10 + "Info" + "=" * 10 + "\n")
    log.write("backbone: {} {}\n".format(args.backbone, args.dims))
    log.write("infer_root: {}\n".format(infer_root))

    args.gen_path = glob(os.path.join(infer_root, f"*.{args.img_ext}"))
    
    log.write("len of gt, gen: {} {}\n".format(len(args.gt_path), len(args.gen_path)))
    if len(args.gen_path) == 0:
        log.write("No generated images found in {}\n".format(infer_root))

    num_avail_cpus = len(os.sched_getaffinity(0))
    num_workers = min(num_avail_cpus, 8)

    with torch.no_grad():
        fid_value = calculate_fid_given_paths(
            args, args.gt_path, args.gen_path, args.batch_size, args.device, args.backbone, args.dims, num_workers
        )
    print("FID: {}\n".format(round(fid_value, 3)))
    log.write("FID: {}\n".format(round(fid_value, 3)))
    log.close()


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--backbone", type=str, default="xrv_densenet_224_all")
    parser.add_argument("--batch_size", type=int, default=200, help="Batch size to use")
    parser.add_argument("--num-workers", type=int, help=("Number of processes to use for data loading. Defaults to `min(8, num_cpus)`"))
    parser.add_argument("--device", type=str, default=None, help="Device to use. Like cuda, cuda:0 or cpu")
    parser.add_argument("--img_ext", type=str, default="png", help="Image extension to search for in folder")
    parser.add_argument("--infer_root", type=str, default=None)
    parser.add_argument("--img_root_dir", type=str, required=True, help="Directory containing preprocessed images")
    parser.add_argument("--img_meta_dir", type=str, required=True, help="Directory containing MIMIC-CXR-JPG metadata")
    parser.add_argument("--tab_root_dir", type=str, required=True, help="Directory containing tabular EHR data")
    parser.add_argument("--use_prev_img_as_trg", action="store_true")
    parser.add_argument("--tab_data_type", type=str, default="filtered")

    IMAGE_EXTENSIONS = {"bmp", "jpg", "jpeg", "pgm", "png", "ppm", "tif", "tiff", "webp"}

    args = parser.parse_args()
    args.device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    main(args)
