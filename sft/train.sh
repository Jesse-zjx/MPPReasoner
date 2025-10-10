GPUS=${1:-0,1,2,3,4,5,6,7}
EXPERIMENT_NAME="sft"

dataset_root=data/train
output_path=ckpts/${EXPERIMENT_NAME}

MAX_PIXELS=1003520 SIZE_FACTOR=8 NPROC_PER_NODE=8 CUDA_VISIBLE_DEVICES=${GPUS} swift sft \
        --model Qwen/Qwen2.5-VL-7B-Instruct \
        --model_type qwen2_5_vl \
        --train_type full \
        --dataset ${dataset_root}/bace.jsonl \
                ${dataset_root}/bbbp.jsonl \
                ${dataset_root}/hiv.jsonl \
                ${dataset_root}/sider.jsonl \
        --torch_dtype bfloat16 \
        --freeze_vit true \
        --num_train_epochs 3 \
        --per_device_train_batch_size 2 \
        --learning_rate 1e-5 \
        --gradient_accumulation_steps 4 \
        --eval_steps 50 \
        --save_steps 100 \
        --save_total_limit 1 \
        --logging_steps 20 \
        --max_length 16384 \
        --max_grad_norm 0.3 \
        --warmup_ratio 0.03 \
        --dataloader_num_workers 8 \
        --dataset_num_proc 1 \
        --save_only_model true \
        --output_dir ${output_path} \
        --use_liger_kernel true \
        --padding_free true \
        --attn_impl flash_attn \
        --gradient_checkpointing False \
        --lr_scheduler_type linear \
        --deepspeed zero2 \
        --tf32 True
