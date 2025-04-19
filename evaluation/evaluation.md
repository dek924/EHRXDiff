# Evaluation of Predicted Chest X-rays
We evaluate the quality of the predicted X-rays using three main categories of metrics: (1) the preservation of medical information, ensuring disease details are accurately represented, (2) the preservation of demographic information, maintaining key patient attributes (age, gender, race) from the input image, and (3) overall image quality, ensuring realistic, high-quality chest radiographs.

** We will provide the classifier model weights.

## Preservation of Medical Information
### 1.1 Chest ImaGenome Classification
We classify images using the Chest ImaGenome model, which requires preprocessing, downloading the model, and running inference.
1) Preprocess Chest ImaGenome Data
    Run the following command:
    ```
    python ./diagnosis_classification/chestimagenome_cls/preprocess/convert_to_reference_dataset.py \
        --tab_root_dir=${TAB_ROOT_DIR} \  # Directory containing tabular data
        --mimiccxr_dir=${IMG_META_DIR} \  # Directory containing metadata for MIMIC-CXR-JPG \
        --chest_imagenome_dir=${IMG_ROOT_DIR} \  # Directory of Chest ImaGenome
        --split=${SPLIT}  # Data split (train/val/test)
    ```

2) Download the Classification Model
    - Download the Chest ImaGenome classification model.
    - Move the downloaded file to `./diagnosis_classification/chestimagenome_cls/checkpoint/`.

3) Run the Model for Evaluation
    Run the following command to evaluate the model:
    ```
    MASTER_PORT=$MASTER_PORT python ./diagnosis_classification/chestimagenome_cls/finetune_attr.py \
        --batch_size_per_gpu=1024 \
        --infer_root=${INFER_ROOT} \  # Directory of predicted images
        --dataroot=${TAB_ROOT_DIR} \  # Tabular data
        --imgmetaroot=${IMG_META_DIR} \  # Metadata for MIMIC-CXR-JPG
        --imgroot=${IMG_ROOT_DIR} \  # Preprocessed images
        --output_dir=./diagnosis_classification/chestimagenome_cls/checkpoint/ \
        --only_test --eval --full_finetune
    ```
    Optional Flags:
    - `--prev_img_as_trg` → Uses "previous image" as the prediction ("Previous image" baseline).
    - `--eval_prev_data` → Uses "previous label" as the prediction ("Previous label" baseline).


### 1.2 CheXpert Classification
- For CheXpert diagnosis classification, we use the torchxrayvision library to evaluate the predicted X-rays. Run the following command to evaluate:
    ```
    python diagnosis_classification/eval_classification_mimic_test.py \
        --infer_root=${PREDICTIED_IMG_DIR} \  # Directory of predicted images
        --batch_size ${BATCH_SIZE}         \
        --img_meta_dir=${IMG_META_DIR}     \  # Metadata for MIMIC-CXR-JPG
        --img_root_dir=${IMG_ROOT_DIR}     \  # Preprocessed images
        --tab_root_dir=${TAB_ROOT_DIR}        # Tabular data
    ```
    Optional Flags:
    - `--use_prev_img_as_trg` → Uses "previous image" as the prediction baseline.
    - `--use_prev_label_for_eval` → Uses "previous label" as the prediction baseline.


## 2. Preservation of Demographic Information
### 2.1 Race & Age Prediction
- For race and age classification, we use SOTA (state-of-the-art) models from the torchxrayvision library. Run the following command to evaluate:
    ```
    python demographics/eval_demographic.py \
        --infer_root=${PREDICTIED_IMG_DIR} \  # Directory of predicted images
        --batch_size ${BATCH_SIZE} \
        --img_meta_dir=${IMG_META_DIR} \  # Metadata for MIMIC-CXR-JPG
        --img_root_dir=${IMG_ROOT_DIR} \  # Preprocessed images
        --tab_root_dir=${TAB_ROOT_DIR} \  # Tabular data
        --tab_db_dir=${TAB_DB_DIR} \  # MIMIC-IV metadata
        --demographic_type=${DEMOGRAPHIC_TYPE}  # Race or gender
    ```
    Optional Flag:
    - `--use_prev_img_as_trg` → Uses "previous image" as the prediction ("Previous image" baseline).

### 2.2 Gender Prediction
1) Download the Gender Classification Model
    - Download the gender classification model.
    - Move the file to `./evaluation/demographics/checkpoint/`.

2) Run the Evaluation
    ```
    python demographics/gender_classification_openai.py \
        --eval \
        --infer_root=${PREDICTIED_IMG_DIR} \  # Directory of predicted images
        --batch_size ${BATCH_SIZE} \
        --img_meta_dir=${IMG_META_DIR} \  # Metadata for MIMIC-CXR-JPG
        --img_root_dir=${IMG_ROOT_DIR} \  # Preprocessed images
        --tab_root_dir=${TAB_ROOT_DIR} \  # Tabular data
        --checkpoint_path=./demographics/checkpoint/best_val_auc.pth.tar
    ```
    Optional Flag:
    - `--use_prev_img_as_trg` → Uses "previous image" as the prediction ("Previous image" baseline).


## 3. Overall Image Quality
- To measure the realism and quality of the predicted chest radiographs, we compute FID (Fréchet Inception Distance). Run the following command to evaluate:
    ```
    python FID/eval_fid.py \
        --infer_root=${PREDICTIED_IMG_DIR} \  # Directory of predicted images
        --batch_size ${BATCH_SIZE} \
        --img_meta_dir=${IMG_META_DIR} \  # Metadata for MIMIC-CXR-JPG
        --img_root_dir=${IMG_ROOT_DIR} \  # Preprocessed images
        --tab_root_dir=${TAB_ROOT_DIR}  # Tabular data
    ```
    Optional Flag:
    - `--use_prev_img_as_trg` → Uses "previous image" as the prediction ("Previous image" baseline).


## 4. Baseline: Table Classifier
We compare performance using a Table Classifier trained on tabular data.

### 4.1 Chest ImaGenome Baseline
1) Preprocess Data
    - Use the same preprocessing approach as in the Chest ImaGenome classification pipeline.
        ```
        python ./diagnosis_classification/chestimagenome_cls/preprocess/convert_to_reference_dataset.py \
            --tab_root_dir=${TAB_ROOT_DIR} \  # Directory containing tabular data
            --mimiccxr_dir=${IMG_META_DIR} \  # Directory containing metadata for MIMIC-CXR-JPG \
            --chest_imagenome_dir=${IMG_ROOT_DIR} \  # Directory of Chest ImaGenome
            --split=${SPLIT}  # Data split (train/val/test)
        ```

2) Train the Table Classifier
    ```
    python ./baseline/table_classifier.py \
        --tab_root_dir=${TAB_ROOT_DIR} \
        --img_meta_dir=${IMG_META_DIR} \
        --pretrained_path=${PRETRAINED_PATH} \  # Path to EHR-CXR CLIP pretrain checkpoint
        --exp_name=${EXP_NAME}  # Experiment name
    ```
    Optional Flag:
    - `--use_prev_label` → Uses previous label embedding as a additional input ("Table classifier (w/ prev label)").

3) Evaluate the Table Classifier
    ```
    python ./baseline/table_classifier.py \
        --tab_root_dir=${TAB_ROOT_DIR} \
        --img_meta_dir=${IMG_META_DIR} \
        --pretrained_path=${PRETRAINED_PATH} \
        --checkpoint_dir=${CHECKPOINT_PATH} \
        --eval
    ```
    
### 4.2 CheXpert Baseline
1) Train the Table Classifier
    ```
    python ./baseline/table_classifier_chexpert.py \
        --tab_root_dir=${TAB_ROOT_DIR} \
        --img_meta_dir=${IMG_META_DIR} \
        --pretrained_path=${PRETRAINED_PATH} \  # Path to EHR-CXR CLIP pretrain checkpoint
        --exp_name=${EXP_NAME}  # Experiment name
    ```
    Optional Flag:
    - `--use_prev_label` → Uses previous label embedding as a additional input ("Table classifier (w/ prev label)").


2) Evaluate the Table Classifier
    ```
    python ./baseline/table_classifier_chexpert.py \
        --tab_root_dir=${TAB_ROOT_DIR} \
        --img_meta_dir=${IMG_META_DIR} \
        --pretrained_path=${PRETRAINED_PATH} \
        --checkpoint_dir=${CHECKPOINT_PATH} \
        --eval
    ```

## Acknowledgements
This implementation uses code from following repositories:
- [Official Dino implementation](https://github.com/facebookresearch/dino)

We thank the authors for their open-sourced code.
