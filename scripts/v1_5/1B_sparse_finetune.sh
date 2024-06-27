#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

deepspeed --master_port 21243 llava/train/sparse_train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path mtgv/MobileLLaMA-1.4B-Chat \
    --teacher_model_name_or_path /mnt/bn/ic-vlm/personal/zhangyuan/SparseVLM/checkpoints/llava-v1.5-1b-base \
    --version v1 \
    --data_path /mnt/bn/ic-vlm/personal/zhangyuan/llavaData/llava_v1_5_mix665k.json \
    --image_folder /mnt/bn/ic-vlm/personal/zhangyuan/llavaData \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-v1.5-1b-pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/test \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
