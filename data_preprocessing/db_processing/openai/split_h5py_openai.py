import os
import json
import h5py
import shutil
import argparse
from tqdm import tqdm



def main(args):
    # Load dataset split information
    with open(os.path.join(args.save_dir, "dataset_split.json"), "r") as f:
        dataset_split = json.load(f)
    print("Loaded dataset split!")

    # Create new HDF5 files for each dataset split
    f_out_dict = {split: h5py.File(os.path.join(args.save_dir, f"{args.output_name}_{split}_openai.h5"), "w") for split in ["train", "valid", "test"]}
    for split_key in f_out_dict.keys():
        f_out_dict[split_key].create_group("ehr")
    print("Created new h5 files!")

    ehr_data = h5py.File(args.ehr_data_path, "r")
    print("Loaded ehr data!")

    # Loop through each subgroup and copy the dataset to the new file
    for i, dataset_key in tqdm(enumerate(ehr_data.keys()), total=len(ehr_data.keys())):
        if dataset_key in dataset_split["train"]:
            split_key = "train"
        elif dataset_key in dataset_split["valid"]:
            split_key = "valid"
        elif dataset_key in dataset_split["test"]:
            split_key = "test"
        else:
            raise ValueError("Invalid split key!")
        
        data_g_in = ehr_data[dataset_key][-1024:]
        f_out_dict[split_key]["ehr"].create_dataset(dataset_key, data=data_g_in, compression='lzf')
        
    for file in f_out_dict.values():
        file.close()

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MIMIC-IV CXR filtered data and store in HDF5 format.")
    
    parser.add_argument("--ehr_data_path", type=str, default="PREPROCESS_DIR/text_embeddings.h5", help="Path to the EHR text embeddings HDF5 file.")
    parser.add_argument("--save_dir", type=str, default="./data", help="Root directory for data storage.")
    parser.add_argument("--output_name", type=str, default="mimiciv-cxr-filtered", help="Output HDF5 file for training data.")
    args = parser.parse_args()

    main(args)