input_rgb_dir=${1:-"input/normal_images"}
unet=${2:-"weights/genpercept-models/unet_normal_v2"}
scheduler="hf_configs/scheduler_beta_1.0_1.0"
mode="normal"
output_dir=${3:-"output_inference/$mode"}

source script/infer/inference_general.sh $mode $unet genpercept $scheduler $input_rgb_dir $output_dir
