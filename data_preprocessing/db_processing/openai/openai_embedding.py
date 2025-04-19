import os
import re
import ast
import time
import h5py
import openai
import tiktoken
import requests
import numpy as np
import pandas as pd

from tqdm import tqdm
from transformers import AutoTokenizer

AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY", "")
RESOURCE_ENDPOINT = "YOUR_ENDPOINT"

def set_openai_env():
    openai.api_type = "azure"
    openai.api_key = AZURE_OPENAI_KEY
    openai.api_base = RESOURCE_ENDPOINT
    openai.api_version = "2022-12-01"
    url = openai.api_base + "/openai/deployments?api-version=2022-12-01"
    r = requests.get(url, headers={"api-key": AZURE_OPENAI_KEY})
    print(r.text)
    print("OpenAI environment set")


def main(args):
    ehr_data_path = os.path.join(args.datadir, "mimiciv.h5")
    ehr_data = h5py.File(ehr_data_path, "r")['ehr']
    cohort = pd.DataFrame({"dicom_id": list(ehr_data.keys())})

    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    cohort["text"] = cohort.dicom_id.apply(lambda x: tokenizer.batch_decode(ehr_data[x]['hi'][:, 0, :], skip_special_tokens=True))

    def normalize_text(s):
        # Combine multiple spaces and unwanted characters into one regex
        s = re.sub(r'\s+', ' ', s).strip()  # Normalize spaces
        s = re.sub(r'[.,]{2,}', '.', s)     # Replace multiple dots or commas with a single dot
        return s.strip()

    print("Load cohort: ", cohort.shape)
    print("Text preprocessing...")
    cohort["text"] = cohort["text"].apply(lambda x: [normalize_text(_x) for _x in x])
    cohort = cohort.explode("text").drop_duplicates()
    print(cohort.shape)

    print("Filtering by max tokens : ", args.max_tokens)
    tokenizer = tiktoken.get_encoding("cl100k_base")
    cohort["n_tokens"] = cohort["text"].apply(lambda x: len(tokenizer.encode(x)))
    cohort = cohort[cohort["n_tokens"] <= args.max_tokens]
    print(cohort.shape)

    print("Set OpenAI environment...")
    set_openai_env()

    print("OpenAI embedding...")
    result_dict = {}

    # batchsize embedding
    for i in tqdm(range(0, len(cohort), args.batch_size)):
        batch_cohort = cohort.iloc[i : i + args.batch_size]
        # start_time = time.time()
        result = None
        while result is None:
            try:
                result = openai.Embedding.create(
                    input=batch_cohort["text"].tolist(),
                    deployment_id="text-embedding-ada-002",
                )
                for _text, _embedding in zip(
                    batch_cohort["text"].tolist(), result["data"]
                ):
                    result_dict[_text] = _embedding["embedding"]
            except Exception as e:
                print(e, "Retry.")
                time.sleep(10)

    print("Save embedding...")
    result_df = pd.DataFrame.from_dict(result_dict, orient="index")
    result_df.to_csv(os.path.join(args.savedir, "openai_embedding.csv"))


def match_emb(args):
    cohort = pd.read_csv(os.path.join(args.datadir, "mimiciv_cohort.csv"), usecols=["dicom_id", "text"])
    
    def normalize_text(s):
        # Combine multiple spaces and unwanted characters into one regex
        s = re.sub(r'\s+', ' ', s).strip()  # Normalize spaces
        s = re.sub(r'[.,]{2,}', '.', s)     # Replace multiple dots or commas with a single dot
        return s.strip()

    print("Load cohort: ", cohort.shape)
    print("Text preprocessing...")
    cohort["text"] = cohort["text"].apply(ast.literal_eval)
    cohort["text"] = cohort["text"].apply(lambda x: [normalize_text(" ".join(_x)) for _x in x])
    
    result_df = pd.read_csv(os.path.join(args.datadir, "openai_embedding.csv"), index_col=0)
    print("Load embedding: ", result_df.shape)

    # HDF5 file to save embeddings
    output_file = os.path.join(args.savedir, 'text_embeddings.h5')

    # Create an HDF5 file
    with h5py.File(output_file, 'w') as h5f:
        # Processing and saving in batches
        batch_size = 4000  # Adjust based on your memory constraints
        for batch_start in tqdm(range(0, len(cohort), batch_size), total=len(cohort) // batch_size, desc="Saving batches"):
            batch_end = min(batch_start + batch_size, len(cohort))
            batch = cohort.iloc[batch_start:batch_end]

            for _, data in tqdm(batch.iterrows(), total=len(batch), desc="Saving embeddings"):
                try:
                    embeddings = result_df.loc[data['text']].values.astype(np.float16)
                    # Create a dataset for each dicom_id
                    h5f.create_dataset(str(data["dicom_id"]), data=embeddings, compression="lzf")
                except Exception as e:
                    print(f"Error processing {data['dicom_id']}: {str(e)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=str, required=True)
    parser.add_argument("--savedir", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    assert args.batch_size < 17, "batch_size should be less than 17"
    if args.savedir is None:
        args.savedir = args.datadir
    main(args)
    match_emb(args)
