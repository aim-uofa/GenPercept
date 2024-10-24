#!/usr/bin/env bash
set -e
set -x

subfolder=${1:-"eval"}

python eval.py \
    --base_data_dir $BASE_DATA_DIR \
    --dataset_config config/dataset/eval/data_scannet_val.yaml \
    --alignment least_square_disparity \
    --prediction_dir output_eval_sd21/${subfolder}/scannet/prediction \
    --output_dir output_eval_sd21/${subfolder}/scannet/eval_metric \
