import os
import cv2
import parmap
import argparse
import numpy as np
import pandas as pd
import multiprocessing as mp

from tqdm import tqdm
from sklearn.metrics import *
from multiprocessing import Pool


def load_mimic_cxr_meta(mimic_cxr_db_dir):
    # load raw data
    cxr_meta = pd.read_csv(os.path.join(mimic_cxr_db_dir, "mimic-cxr-2.0.0-metadata.csv"))

    # Assumption: Use only frontal images
    cxr_meta = cxr_meta[cxr_meta["ViewPosition"].isin(["AP", "PA"])].reset_index()

    return cxr_meta

def transform_image(image, cropped=False):
    # Convert BGR to RGB
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if cropped:
        ret, thresh = cv2.threshold(gray, 0, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt = contours[max_index]
        x, y, w, h = cv2.boundingRect(cnt)
        crop_img = image[y : y + h, x : x + w]
    else:
        crop_img = image

    image = cv2.resize(crop_img, (1024, 1024), interpolation=cv2.INTER_CUBIC)

    # Resize to 1024x1024 then take the center crop
    center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
    cropped = image[center_y-512:center_y+512, center_x-512:center_x+512]
    
    # Resize cropped image to 256x256
    resized = cv2.resize(cropped, (256, 256), interpolation=cv2.INTER_CUBIC)
    
    # Convert back to uint8 range [0,255] for saving
    return resized

def _preprocess_image(rows, cropped):
    """
    main process:
    - load
    - crop
    - resize
    - save

    reference:
    - https://stackoverflow.com/questions/52979965/crop-images-with-different-black-margins
    - https://til.songyunseop.com/python/tqdm-with-multiprocessing.html
    """

    for row in tqdm(rows.iterrows(), total=len(rows)):
        # get a row information
        row = row[1]

        # Load an image
        pid, sid, iid = str(row.subject_id), str(row.study_id), str(row.dicom_id)
        image_path = os.path.join(args.mimic_cxr_db_dir, f"files/p{pid[:2]}/p{pid}/s{sid}/{iid}.jpg")
        
        img = cv2.imread(image_path)
        transformed_img = transform_image(img, cropped)

        save_image_dir = os.path.join(args.save_img_dir, f"p{pid[:2]}/p{pid}/s{sid}")
        os.makedirs(save_image_dir, exist_ok=True)

        save_image_path = os.path.join(args.save_img_dir, f"p{pid[:2]}/p{pid}/s{sid}/{iid}.jpg")
        cv2.imwrite(save_image_path, transformed_img)


def main(args):
    # load dataset
    meta_data = load_mimic_cxr_meta(mimic_cxr_db_dir=args.mimic_cxr_db_dir)

    # run multiprocessing
    num_cores = int(mp.cpu_count() / 4)
    splitted_data = np.array_split(meta_data, num_cores)
    
    # with mp.Pool(num_cores) as pool:
    parmap.map(_preprocess_image, splitted_data, args.cropped, pm_pbar=True, pm_processes=num_cores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # directory
    parser.add_argument("--mimic_cxr_db_dir", type=str, default="./physionet.org/files/mimic-cxr-jpg/2.0.0/")
    parser.add_argument("--save_img_dir", type=str, default="./re256")

    # image preprocessing
    parser.add_argument("--cropped", action="store_true", default=False)

    args = parser.parse_args()
    assert ("contour_cropped" if args.cropped else "") in os.path.basename(args.save_img_dir)

    main(args)
