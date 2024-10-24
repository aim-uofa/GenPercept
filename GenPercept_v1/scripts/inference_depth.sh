# CUDA_VISIBLE_DEVICES='0'
export PYTHONPATH=./

python tools/inference_genpercept.py \
--input_rgb_dir 'input/depth' \
--output_dir 'output/depth' \
--mode 'depth' \
--checkpoint "weights/v1" \
--unet_ckpt_path "weights/v1/unet_depth_v1" \

