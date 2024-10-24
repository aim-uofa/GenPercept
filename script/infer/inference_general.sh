mode=$1
unet=$2
archs=$3
scheduler=${4:-"hf_configs/scheduler_beta_1.0_1.0"}
input_rgb_dir=${5:-"input"}

unet_basename=$(basename "$unet")
scheduler_basename=$(basename "$scheduler")

output_dir=${6:-"output_inference/${archs}_${mode}_${scheduler_basename}/$unet_basename"}
denoise_steps=${7:-"1"}
ensemble_size=${8:-"1"}
sd_pretrain=${9:-"pretrained_weights/stable-diffusion-2-1"}

python run.py \
    --archs $archs \
    --checkpoint $sd_pretrain \
    --unet $unet \
    --denoise_steps $denoise_steps \
    --ensemble_size $ensemble_size \
    --input_rgb_dir $input_rgb_dir \
    --output_dir $output_dir \
    --scheduler $scheduler \
    --mode $mode
