#!/usr/bin/env bash
set -e
set -x

# Use specified checkpoint path, otherwise, default value
unet=$1
archs=$2
ckpt=${3:-"pretrained_weights/stable-diffusion-2-1"}
subfolder=${4:-"eval"}
scheduler=${5:-"hf_configs/scheduler_beta_0.00085_0.012"}
fix_timesteps=${6:-None}

python infer.py  \
    --checkpoint $ckpt \
    --scheduler $scheduler \
    --unet $unet \
    --archs $archs \
    --fix_timesteps $fix_timesteps \
    --seed 1234 \
    --base_data_dir $BASE_DATA_DIR \
    --denoise_steps 10 \
    --ensemble_size 1 \
    --dataset_config config/dataset/eval/data_eth3d.yaml \
    --output_dir output_eval_sd21/${subfolder}/eth3d/prediction \
    --processing_res 756 \
    --resample_method bilinear \