#!/usr/bin/env bash
# set -e
# set -x

export BASE_DATA_DIR="datasets_eval"
eval_folder="eval_ensemble1_step10_disparity"
ckpt="pretrained_weights/stable-diffusion-2-1"

unet_list=(
    "weights/genpercept-exps/main_paper/sd21_train_genpercept_exr_1card_ensure_disparity_bs4_per_accu_768_pixel_mse_ssi_grad_loss"
    "weights/genpercept-exps/main_paper/sd21_train_genpercept_exr_1card_ensure_disparity_bs8_per_accu_768_pixel_mse_ssi_grad_loss_dpt_head_identity_decoder_3e-5"
)


archs_list=(
    "genpercept"
    "genpercept"
)

scheduler_list=(
    "hf_configs/scheduler_beta_1.0_1.0"
    "hf_configs/scheduler_beta_1.0_1.0"
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
