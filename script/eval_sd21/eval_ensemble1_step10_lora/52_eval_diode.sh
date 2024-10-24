#!/usr/bin/env bash
set -e
set -x

subfolder=${1:-"eval"}

python eval.py \
    --base_data_dir $BASE_DATA_DIR \
    --dataset_config config/dataset/eval/data_diode_val_all.yaml \
    --alignment least_square \
    --prediction_dir output_eval_sd21/${subfolder}/diode/prediction \
    --output_dir output_eval_sd21/${subfolder}/diode/eval_metric \
