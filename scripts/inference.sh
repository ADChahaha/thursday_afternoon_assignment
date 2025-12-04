MODEL="SAFE"
RESUME_PATH="checkpoint"

inference_images="test_image"

python inference.py \
        --input_size 256 \
        --transform_mode 'crop' \
        --model $MODEL \
        --eval_data_path $inference_images \
        --batch_size 1 \
        --output_dir $RESUME_PATH \
        --resume '/home/yukino/MachineLearningPractice/results/ResNet-50/20251203_161301/checkpoint-best.pth' \
        --eval True