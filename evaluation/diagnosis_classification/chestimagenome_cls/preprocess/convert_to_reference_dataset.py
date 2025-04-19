import os
import json
import argparse
import pandas as pd

CONFIG = {
    "REMOVED_OBJS": ["left arm", "right arm"],
    "REMOVED_ATTRS": [
        "artifact",
        "bronchiectasis",
        "pigtail catheter",
        "skin fold",
        "aortic graft/repair",
        "diaphragmatic eventration (benign)",
        "sternotomy wires",
    ]
}


USED_ATTRS = ['pleural effusion', 'enlarged cardiac silhouette', 'lung opacity',
            'pulmonary edema/hazy opacity', 'pneumonia', 'consolidation',
            'vascular congestion', 'pneumothorax', 'enlarged hilum',
            'mediastinal widening']

def process_dataset(dataset):
    dataset = dataset[~dataset["object"].isin(CONFIG["REMOVED_OBJS"])]
    dataset = dataset[~dataset["attribute"].isin(CONFIG["REMOVED_ATTRS"])]
    dataset = dataset.reset_index(drop=True)
    return dataset

def filterout_by_prevalence(dataset, prevalence_threshold=0.05):
    # Filter out objects by prevalence
    prevalence = dataset.groupby("object").size() / len(dataset)
    prevalence = prevalence[prevalence > prevalence_threshold]
    dataset = dataset[dataset["object"].isin(prevalence.index)]
    dataset = dataset.reset_index(drop=True)
    return dataset

def main(args):
    # Load dataset
    cohort = pd.read_csv(os.path.join(args.tab_root_dir, "mimiciv_cohort_meta.csv"), index_col=0)
    cohort["data_key"] = cohort["dicom_id"]
    cohort[["dicom_id", "prev_dicom_id"]] = cohort["data_key"].str.split("_", expand=True)
    
    with open(os.path.join(args.tab_root_dir, f"dataset_split.json"), "r") as f:
        dataset_split = json.load(f)

    mimic_meta = pd.read_csv(os.path.join(args.mimiccxr_dir, "mimic-cxr-2.0.0-metadata.csv"))
    mimic_meta = mimic_meta[mimic_meta.ViewPosition.isin(["PA", "AP"])]

    if args.split in ["train", "valid", "test"]:
        dataset = pd.read_csv(
            os.path.join(args.chest_imagenome_dir, "silver_dataset/scene_tabular/attribute_relations_tabular.txt"), 
            sep="\t",
            usecols=["study_id", "image_id", "sent_loc", "row_id", "bbox", "categoryID", "label_name", "relation"],
        )
        
        # Use label filtering
        dataset = dataset[dataset.categoryID != "nlp"]
        
        # Use report-level label
        dataset = dataset.sort_values(by=["study_id", "image_id", "sent_loc", "bbox", "label_name"], ascending=True)  # default ascending=True
        dataset = dataset.drop_duplicates(subset=["study_id", "image_id", "bbox", "label_name"], keep="last")

        # Use attribute-level label
        dataset = dataset.sort_values(by=["study_id", "image_id", "label_name", "relation"], ascending=True)
        dataset = dataset.drop_duplicates(subset=["study_id", "image_id", "label_name"], keep="last")
        
        dataset = pd.merge(dataset, mimic_meta[["study_id", "subject_id"]], on="study_id", how="inner")
        dataset = dataset[["subject_id", "study_id", "image_id", "categoryID", "label_name", "relation"]]

        # Filter dataset
        test_iids = list(cohort[cohort.data_key.isin(dataset_split["test"])].dicom_id.unique())
        test_sids = dataset[dataset.image_id.isin(test_iids)].subject_id.unique()

        valid_iis = list(cohort[cohort.data_key.isin(dataset_split["valid"])].dicom_id.unique())
        valid_sids = dataset[dataset.image_id.isin(valid_iis)].subject_id.unique()

        if args.split == "test":
            dataset = dataset[dataset.subject_id.isin(test_sids)]
        elif args.split == "valid":
            dataset = dataset[dataset.subject_id.isin(valid_sids)]
        else:
            dataset = dataset[~dataset.subject_id.isin(test_sids) & ~dataset.subject_id.isin(valid_sids)]
            gold_dataset = pd.read_csv(
                os.path.join(args.chest_imagenome_dir, "gold_dataset/gold_attributes_relations_500pts_500studies1st.txt"), 
                sep="\t",
                usecols=["patient_id"]
            )
            dataset = dataset[~dataset.subject_id.isin(gold_dataset.patient_id.unique())]
    elif args.split == "gold_test":
        dataset = pd.read_csv(
            os.path.join(args.chest_imagenome_dir, "gold_dataset/gold_attributes_relations_500pts_500studies1st.txt"), 
            sep="\t",
            usecols=["patient_id", "study_id", "image_id", "row_id", "bbox", "categoryID", "label_name", "relation"]
        )
        dataset = dataset.rename(columns={"patient_id": "subject_id"})
        dataset["image_id"] = dataset["image_id"].str.split(".").str[0]
        dataset["sent_loc"] = dataset["row_id"].apply(lambda x: float(x.split("|")[-1]))

        # Use report-level label
        dataset = dataset.sort_values(by=["study_id", "image_id", "sent_loc", "bbox", "label_name"])
        dataset = dataset.drop_duplicates(subset=["study_id", "image_id", "bbox", "label_name"], keep="last")

        # Use attribute-level label
        dataset = dataset.sort_values(by=["study_id", "image_id", "label_name", "relation"], ascending=True)
        dataset = dataset.drop_duplicates(subset=["study_id", "image_id", "label_name"], keep="last")

        dataset = dataset[["subject_id", "study_id", "image_id", "categoryID", "label_name", "relation"]]
    else:
        raise ValueError("Invalid split")

    dataset = dataset.rename(columns={"label_name": "attribute", "categoryID": "category"})
    dataset = dataset[dataset.attribute.isin(USED_ATTRS)]
    print(f"Load {len(dataset)} samples for {args.split} set ({dataset.image_id.nunique()} images, {dataset.attribute.nunique()} attributes)")

    # store new dataset
    if not os.path.exists(os.path.join(args.save_dir, "reference_exp_attr_level")):
        os.makedirs(os.path.join(args.save_dir, "reference_exp_attr_level"), exist_ok=True)

    # store new dataset
    dataset = dataset.to_dict("records")
    args.save_path = os.path.join(args.save_dir, "reference_exp_attr_level", f"{args.split}_ref.json")
    print(f"Saved new dataset to {args.save_path}")
    with open(args.save_path, "w") as f:
        json.dump(dataset, f, indent=4, default=str)  # use `default=str` to serialize int64


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="preprocessing upperbound dataset")

    # file directory
    parser.add_argument("--tab_root_dir", default="./data", type=str)
    parser.add_argument("--mimiccxr_dir", default="./mimic-cxr-jpg/2.0.0", type=str)
    parser.add_argument("--chest_imagenome_dir", default="./chest-imagenome", type=str)
    parser.add_argument("--save_dir", default=None, type=str)
    parser.add_argument("--split", default="valid", choices=["train", "valid", "test", "gold_test"], type=str)

    args = parser.parse_args()

    if args.save_dir is None:
        args.save_dir = args.tab_root_dir
    os.makedirs(args.save_dir, exist_ok=True)

    main(args)
    print("Done")
