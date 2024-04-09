CUDA_VISIBLE_DEVICES='1'
export PYTHONPATH=./

python tools/inference_genpercept.py \
--input_rgb_dir 'input' \
--output_dir 'output' \
--mode 'normal' \
--checkpoint "weights/normal_v1"

