# Data Preprocessing
This repository provides preprocessing scripts for the MIMIC-IV (v2.2) EHR tabular database and MIMIC-CXR-JPG (v2.0.0) images.
The preprocessing code in the `db_processing` folder is highly dependent on the code from [UniHPF](https://arxiv.org/abs/2207.09858) and utilizes components from the following repository:
- [Official UniHPF Preprocessing Implementation](https://github.com/Jwoo5/integrated-ehr-pipeline/blob/master/README.md)

# Install Requirements
- NOTE: This repository requires `python>=3.9` and `Java>=8`
- NOTE: Since there is a performance issue related to `transformers` library, it is recommended to use `transformers==4.29.1`.

To install the necessary dependencies, run:
   ```
   pip install tqdm treelib pyspark pandas==1.5.2 transformers==4.29.1 numpy==1.26.4 tiktoken openai
   ```

# Usage Guidance
## EHR Tabular Data Preprocessing
1) Download the datasets:  
   Obtain MIMIC-IV (v2.2) and MIMIC-CXR-JPG (v2.0.0) after obtaining credentialing on PhysioNet. Follow the instructions in [README.md](../README.md) for details.

2) Run the preprocessing script:  
   Execute the following command to preprocess the EHR tabular data and CXR images:
   ```
   cd db_processing
   python main.py \
       --ehr mimicivfiltered \
       --data $MIMIC_IV_DIR \
       --img_datadir $MIMIC_CXR_DIR \
       --dest $SAVE_DIR \
       --ext ".csv.gz" \
       --use_image \
       --use_admission_table \
       --use_more_tables \
       --valid-percent 0.05
   ```

3) Generate OpenAI Embeddings:  
   To generate OpenAI embeddings for each sample, run:
    ```
    python openai/openai_embedding.py --datadir $SAVE_DIR
    python openai/split_h5py_openai.py --save_dir $SAVE_DIR --ehr_data_path $SAVED_DIR/text_embeddings.h5 
    ```
    **Note**: You should use the Azure OpenAI service and opt out of human review for the data.

After preprocessing, set your $SAVE_DIR as $TAB_ROOT_DIR when running training and evaluation scripts.

## CXR Image Preprocessing
Run the following command to preprocess CXR images:
```
cd img_preprocessing
python img_preprocessing.py --mimic_cxr_db_dir $mimic_cxr_image_dir --savedir $SAVED_DIR
```