# CUDA_VISIBLE_DEVICES='0'
export PYTHONPATH=./

python tools/inference_genpercept.py \
--input_rgb_dir 'input/dis' \
--output_dir 'output/dis' \
--mode 'seg' \
--checkpoint "weights/v1" \
--unet_ckpt_path "weights/v1/unet_dis_v1" \

