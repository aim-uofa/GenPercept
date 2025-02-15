
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
num_workers=2

echo $num_gpus

accelerate launch --num_processes $num_gpus --num_machines 1 \
--main_process_port $(( RANDOM % 1000 + 50000 )) accelerate_train.py \
 --mixed_precision "fp16" \
 --config config/ablation/beta_values/sd21_train_marigold_exr_1card_ensure_wo_rgb_blending_0point000425_0point006_wo_multi_res_noise.yaml \
 --base_data_dir datasets \
 --base_ckpt_dir pretrained_weights