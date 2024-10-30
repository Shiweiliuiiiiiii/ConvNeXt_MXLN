# python -m torch.distributed.launch --nproc_per_node=4 main.py \
# --model convnext_tiny --drop_path 0.1 \
# --batch_size 128 --lr 4e-3 --update_freq 4 \
# --model_ema true --model_ema_eval true \
# --data_path /ImageNet1K \
# --epochs 120 \
# --output_dir ./save_results
torchrun --nproc_per_node=2 --master_port=29501 main.py \
--model vit --drop_path 0.1 \
--batch_size 1024 --lr 4e-3 --update_freq 2 \
--model_ema true --model_ema_eval true \
--data_path /scratch/shiwei/data/imagenet1k/ \
--epochs 120 \
--log_dir ./tensorboard_log/mixln_vit_t \
--output_dir /scratch/shiwei/shiwei/mixln_vit_tiny_pre \
--mode tiny