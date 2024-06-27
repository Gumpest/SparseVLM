#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
MODEL=llava-v1.5-1b-sparse-by-text
python3 -m llava.eval.model_vqa_loader \
    --model-path /mnt/bn/ic-vlm/personal/zhangyuan/SparseVLM/checkpoints/$MODEL \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/$MODEL.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd ./playground/data/eval/MME

python3 convert_answer_to_mme.py --experiment $MODEL

cd eval_tool

python3 calculation.py --results_dir answers/$MODEL
