#!/usr/bin/env bash
set -e
set -x

subfolder=${1:-"eval"}

python eval.py \
    --base_data_dir $BASE_DATA_DIR \
    --dataset_config config/dataset/eval/data_nyu_test.yaml \
    --alignment least_square_disparity \
    --prediction_dir output_eval_sd21/${subfolder}/nyu_test/prediction \
    --output_dir output_eval_sd21/${subfolder}/nyu_test/eval_metric \
