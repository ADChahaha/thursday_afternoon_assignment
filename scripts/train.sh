#!/bin/bash

train_datasets=(
    "data/datasets/train_ForenSynths/train" \
)
eval_datasets=(
    "data/datasets/train_ForenSynths/val" \
)

MODEL="ResNet-50"

for train_dataset in "${train_datasets[@]}" 
do
    for eval_dataset in "${eval_datasets[@]}" 
    do

        current_time=$(date +"%Y%m%d_%H%M%S")
        OUTPUT_PATH="results/$MODEL/$current_time"
        mkdir -p $OUTPUT_PATH

        python main_finetune.py \
            --input_size 256 \
            --transform_mode 'crop' \
            --model $MODEL \
            --data_path "$train_dataset" \
            --eval_data_path "$eval_dataset" \
            --save_ckpt_freq 1 \
            --batch_size 32 \
            --blr 1e-2 \
            --weight_decay 0.01 \
            --warmup_epochs 1 \
            --epochs 20 \
            --output_dir $OUTPUT_PATH \
            --log_dir $OUTPUT_PATH/tensorboard \
            --pretrained False \
        2>&1 | tee -a $OUTPUT_PATH/log_train.txt

    done
done
