python -m torch.distributed.launch --master_port=29500 --nproc_per_node=2 main.py \
--model vit --drop_path 0.1 \
--batch_size 256 --lr 4e-3 --update_freq 4 \
--model_ema true --model_ema_eval true \
--data_path /ImageNet1K \
--epochs 120 \
--log_dir ./tensorboard_log/mix_s_vit \
--output_dir ./save_results_mix_vit_s \
--mode small