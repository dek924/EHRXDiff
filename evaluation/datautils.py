import os
import h5py
import pickle
import pandas as pd

def load_mimic_cxr_meta(mimic_cxr_dir):
    # load raw data
    cxr_meta = pd.read_csv(os.path.join(os.path.dirname(mimic_cxr_dir), "mimic-cxr-2.0.0-metadata.csv"))
    cxr_meta = cxr_meta[["dicom_id", "subject_id", "study_id", "StudyDate", "StudyTime", "ViewPosition"]]
    print("Load mimic cxr metadata: ", cxr_meta.shape)

    # Use only frontal view image
    cxr_meta = cxr_meta[cxr_meta.ViewPosition.isin(["PA", "AP"])]
    cxr_meta = cxr_meta.drop(columns=["ViewPosition"])
    print("Use only frontal view image: ", cxr_meta.shape)

    return cxr_meta

def get_gt_path(phase, img_root_dir, tab_root_dir, mimic_cxr_dir, tab_data_type):
    # Load meta data
    img_meta = pd.read_csv(os.path.join(mimic_cxr_dir, "mimic-cxr-2.0.0-metadata.csv"))
    cohort = pd.read_csv(os.path.join(tab_root_dir, "mimiciv_cohort_meta.csv"))
    
    cohort["data_key"] = cohort["dicom_id"]
    cohort[["dicom_id", "prev_dicom_id"]] = cohort["data_key"].str.split("_", expand=True)

    cohort = cohort.merge(img_meta[["dicom_id", "study_id"]].rename(columns={"dicom_id": "prev_dicom_id", "study_id": "prev_study_id"}), on="prev_dicom_id", how="left")
    if tab_data_type == "all":
        tab_inputs = h5py.File(os.path.join(tab_root_dir, f"mimiciv-cxr_{phase}_openai.h5"), "r")["ehr"]
    elif tab_data_type == "filtered":
        tab_inputs = h5py.File(os.path.join(tab_root_dir, f"mimiciv-cxr-filtered_{phase}_openai.h5"), "r")["ehr"]
    else:
        raise ValueError
        
    data_keys = list(tab_inputs.keys())
    cohort = cohort[cohort.data_key.isin(data_keys)]

    def get_img_path(row, img_root_dir, study_col, dicom_col):
        subject_id = str(row["subject_id"])
        study_id = str(int(row[study_col]))
        dicom_id = row[dicom_col]

        return f"{img_root_dir}/p{subject_id[:2]}/p{subject_id}/s{study_id}/{dicom_id}.jpg"

    # For current img_path
    cohort["img_path"] = cohort.apply(get_img_path, img_root_dir=img_root_dir, study_col="study_id", dicom_col="dicom_id", axis=1)

    return cohort["img_path"].tolist()