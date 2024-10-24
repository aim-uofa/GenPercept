#!/usr/bin/env bash
set -e
set -x

subfolder=${1:-"eval"}

python eval.py \
    --base_data_dir $BASE_DATA_DIR \
    --dataset_config config/dataset/eval/data_eth3d.yaml \
    --alignment least_square \
    --prediction_dir  output_eval_sd21/${subfolder}/eth3d/prediction \
    --output_dir output_eval_sd21/${subfolder}/eth3d/eval_metric \
    --alignment_max_res 1024 \