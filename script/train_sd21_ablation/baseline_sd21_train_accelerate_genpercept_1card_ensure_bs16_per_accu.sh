
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
num_workers=2

echo $num_gpus

accelerate launch --num_processes $num_gpus --num_machines 1 \
--main_process_port $(( RANDOM % 1000 + 50000 )) accelerate_train.py \
 --mixed_precision "fp16" \
 --config config/ablation/baseline_sd21_train_genpercept_exr_1card_ensure_bs16_per_accu.yaml \
 --base_data_dir datasets \
 --base_ckpt_dir pretrained_weights