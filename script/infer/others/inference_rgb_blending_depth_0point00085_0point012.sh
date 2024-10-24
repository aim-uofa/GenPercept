input_rgb_dir=${1:-"input/depth_images"}
unet=${2:-"weights/genpercept-exps/ablation/beta_values/sd21_train_marigold_exr_1card_ensure_rgb_blending_0point00085_0point012"}
scheduler=${3:-"hf_configs/scheduler_beta_0.00085_0.012"} 
mode="depth"
archs="rgb_blending"

unet_basename=$(basename "$unet")
scheduler_basename=$(basename "$scheduler")

output_dir=${4:-"output_inference/${archs}_${mode}_${scheduler_basename}/$unet_basename"}
denoise_steps=${7:-"10"}
ensemble_size=${8:-"1"}


source script/infer/inference_general.sh $mode $unet $archs $scheduler $input_rgb_dir $output_dir $denoise_steps $ensemble_size
