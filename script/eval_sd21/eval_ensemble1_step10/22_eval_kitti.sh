#!/usr/bin/env bash
set -e
set -x

subfolder=${1:-"eval"}

python eval.py \
    --base_data_dir $BASE_DATA_DIR \
    --dataset_config config/dataset/eval/data_kitti_eigen_test.yaml \
    --alignment least_square \
    --prediction_dir output_eval_sd21/${subfolder}/kitti_eigen_test/prediction \
    --output_dir output_eval_sd21/${subfolder}/kitti_eigen_test/eval_metric \
