#!/usr/bin/env bash
# set -e
# set -x

export BASE_DATA_DIR="datasets_eval"
eval_folder="eval_ensemble1_step10"
ckpt="pretrained_weights/stable-diffusion-2-1"

unet_list=(
    "weights/genpercept-exps/ablation/beta_values/sd21_train_marigold_exr_1card_ensure_rgb_blending_0point00085_0point012"
    "weights/genpercept-exps/ablation/beta_values/sd21_train_marigold_exr_1card_ensure_rgb_blending_0point00340_0point048"
    "weights/genpercept-exps/ablation/beta_values/sd21_train_marigold_exr_1card_ensure_rgb_blending_0point000425_0point006"
    "weights/genpercept-exps/ablation/beta_values/sd21_train_marigold_exr_1card_ensure_rgb_blending_0point1360_0point192"
    "weights/genpercept-exps/ablation/beta_values/sd21_train_marigold_exr_1card_ensure_rgb_blending_0point0002125_0point003"
    "weights/genpercept-exps/ablation/beta_values/sd21_train_marigold_exr_1card_ensure_rgb_blending_0point05440_0point768"

    "weights/genpercept-exps/ablation/beta_values/sd21_train_marigold_exr_1card_ensure_wo_rgb_blending_0point00085_0point012"
    "weights/genpercept-exps/ablation/beta_values/sd21_train_marigold_exr_1card_ensure_wo_rgb_blending_0point00340_0point048"
    "weights/genpercept-exps/ablation/beta_values/sd21_train_marigold_exr_1card_ensure_wo_rgb_blending_0point000425_0point006"
    "weights/genpercept-exps/ablation/beta_values/sd21_train_marigold_exr_1card_ensure_wo_rgb_blending_0point1360_0point192"
    "weights/genpercept-exps/ablation/beta_values/sd21_train_marigold_exr_1card_ensure_wo_rgb_blending_0point0002125_0point003"
    "weights/genpercept-exps/ablation/beta_values/sd21_train_marigold_exr_1card_ensure_wo_rgb_blending_0point05440_0point768"

    "weights/genpercept-exps/ablation/beta_values/sd21_train_marigold_exr_1card_ensure_wo_rgb_blending_0point00085_0point012_wo_multi_res_noise"
    "weights/genpercept-exps/ablation/beta_values/sd21_train_marigold_exr_1card_ensure_wo_rgb_blending_0point00340_0point048_wo_multi_res_noise"
    "weights/genpercept-exps/ablation/beta_values/sd21_train_marigold_exr_1card_ensure_wo_rgb_blending_0point000425_0point006_wo_multi_res_noise"
    "weights/genpercept-exps/ablation/beta_values/sd21_train_marigold_exr_1card_ensure_wo_rgb_blending_0point1360_0point192_wo_multi_res_noise"
    "weights/genpercept-exps/ablation/beta_values/sd21_train_marigold_exr_1card_ensure_wo_rgb_blending_0point0002125_0point003_wo_multi_res_noise"
    "weights/genpercept-exps/ablation/beta_values/sd21_train_marigold_exr_1card_ensure_wo_rgb_blending_0point05440_0point768_wo_multi_res_noise"
)


archs_list=(
    "rgb_blending"
    "rgb_blending"
    "rgb_blending"
    "rgb_blending"
    "rgb_blending"
    "rgb_blending"

    "marigold"
    "marigold"
    "marigold"
    "marigold"
    "marigold"
    "marigold"

    "marigold"
    "marigold"
    "marigold"
    "marigold"
    "marigold"
    "marigold"
)

scheduler_list=(
    "hf_configs/scheduler_beta_0.00085_0.012"
    "hf_configs/scheduler_beta_0.00340_0.048"
    "hf_configs/scheduler_beta_0.000425_0.006"
    "hf_configs/scheduler_beta_0.1360_0.192"
    "hf_configs/scheduler_beta_0.0002125_0.003"
    "hf_configs/scheduler_beta_0.5440_0.768"

    "hf_configs/scheduler_beta_0.00085_0.012"
    "hf_configs/scheduler_beta_0.00340_0.048"
    "hf_configs/scheduler_beta_0.000425_0.006"
    "hf_configs/scheduler_beta_0.1360_0.192"
    "hf_configs/scheduler_beta_0.0002125_0.003"
    "hf_configs/scheduler_beta_0.5440_0.768"
    
    "hf_configs/scheduler_beta_0.00085_0.012"
    "hf_configs/scheduler_beta_0.00340_0.048"
    "hf_configs/scheduler_beta_0.000425_0.006"
    "hf_configs/scheduler_beta_0.1360_0.192"
    "hf_configs/scheduler_beta_0.0002125_0.003"
    "hf_configs/scheduler_beta_0.5440_0.768"
)




for ((i=1; i<=${#unet_list[@]}; i++)); do
    unet="${unet_list[$i]}"
    unet_name=$(basename "$unet")
    save_subfolder="$eval_folder/$unet_name"
    archs="${archs_list[$i]}"
    scheduler="${scheduler_list[$i]}"

    bash script/eval_sd21/$eval_folder/11_infer_nyu.sh $unet $archs $ckpt $save_subfolder $scheduler
    bash script/eval_sd21/$eval_folder/12_eval_nyu.sh $save_subfolder

    bash script/eval_sd21/$eval_folder/21_infer_kitti.sh $unet $archs $ckpt $save_subfolder $scheduler
    bash script/eval_sd21/$eval_folder/22_eval_kitti.sh $save_subfolder

    bash script/eval_sd21/$eval_folder/41_infer_scannet.sh $unet $archs $ckpt $save_subfolder $scheduler
    bash script/eval_sd21/$eval_folder/42_eval_scannet.sh $save_subfolder

    bash script/eval_sd21/$eval_folder/51_infer_diode.sh $unet $archs $ckpt $save_subfolder $scheduler
    bash script/eval_sd21/$eval_folder/52_eval_diode.sh $save_subfolder

    bash script/eval_sd21/$eval_folder/31_infer_eth3d.sh $unet $archs $ckpt $save_subfolder $scheduler
    bash script/eval_sd21/$eval_folder/32_eval_eth3d.sh $save_subfolder

done
