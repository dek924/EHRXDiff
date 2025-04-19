# CLIP Pretraining for EHR-CXR
This repository provides code for pretraining a CLIP model using EHR tabular data and CXR images.
Our implementation is based on the train-CLIP repository, which uses PyTorch Lightning.

## Usage
To pretrain the model, run:
```
python train_finetune_tab.py \
    --img_root_dir=${IMG_ROOT_DIR} # Directory containing preprocessed images \
    --img_meta_dir=${IMG_META_DIR} # Directory containing metadata for MIMIC-CXR-JPG \
    --tab_root_dir=${TAB_ROOT_DIR} # Directory containing tabular data \
    --batch_size=${BATCH_SIZE} \
    --accumulate_grad_batch=${accumulate_grad_batch} \
    --gpus=${NUM_GPUS}  \
    --benchmark=False   \
    --expname=${EXP_NAME}
```

## Requirements
Ensure that you use the same environment (`ehrxdiff`) as specified in [README.md](../README.md).

## Acknowledgments
This implementation is based on the following repository:
```
@misc{cg2021trainCLIP,
  author = {Cade Gordon},
  title = {train-CLIP},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  doi = {10.5281/zenodo.4915843},
  howpublished = {\url{https://github.com/Zasder3/train-CLIP}}
}
```
We thank the authors for their open-sourced code. <br /><br />