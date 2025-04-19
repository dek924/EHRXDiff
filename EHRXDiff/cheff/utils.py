import os
import h5py
import pandas as pd


def load_mimic_cxr_meta(mimic_cxr_dir, img_root_dir):
    # load raw data
    cxr_meta = pd.read_csv(os.path.join(os.path.dirname(mimic_cxr_dir), "mimic-cxr-2.0.0-metadata.csv"))
    cxr_meta = cxr_meta[["dicom_id", "subject_id", "study_id", "StudyDate", "StudyTime", "ViewPosition"]]
    print("Load mimic cxr metadata: ", cxr_meta.shape)

    # Use only frontal view image
    cxr_meta = cxr_meta[cxr_meta.ViewPosition.isin(["PA", "AP"])]
    cxr_meta = cxr_meta.drop(columns=["ViewPosition"])
    print("Use only frontal view image: ", cxr_meta.shape)

    def make_jpath(data):
        _subject_id = data["subject_id"]
        _study_id = data["study_id"]
        _dicom_id = data["dicom_id"]
        return f"/p{_subject_id[:2]}/p{_subject_id}/s{_study_id}/{_dicom_id}.jpg"

    cxr_meta = cxr_meta.astype("str")
    cxr_meta["jpg_fpath"] = cxr_meta.apply(lambda x: make_jpath(x), axis=1)
    cxr_meta["jpg_fpath"] = img_root_dir + cxr_meta["jpg_fpath"]

    return cxr_meta


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

    tab_inputs = h5py.File(os.path.join(tab_root_dir, file_name), "r", driver="sec2")
    print("Load ", os.path.join(tab_root_dir, file_name))

    tab_input_key = tab_inputs["ehr"].keys()

    return tab_inputs, tab_input_key