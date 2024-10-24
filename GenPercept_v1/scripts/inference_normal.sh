# CUDA_VISIBLE_DEVICES='0'
export PYTHONPATH=./

python tools/inference_genpercept.py \
--input_rgb_dir 'input/normal' \
--output_dir 'output/normal' \
--mode 'normal' \
--checkpoint "weights/v1" \
--unet_ckpt_path "weights/v1/unet_normal_v1" \

