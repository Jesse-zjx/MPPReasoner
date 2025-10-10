set -x

# ==== User-configurable parameters ====
MODEL_PATH=$1
TRAIN_ROOT="./data/train"
TRAIN_FILES="[$TRAIN_ROOT/bace_train.parquet,$TRAIN_ROOT/bbbp_train.parquet,$TRAIN_ROOT/sider_train.parquet,$TRAIN_ROOT/hiv_train.parquet"

EXPERIMENT_NAME="rl"
BATCH_SIZE=128
MAX_PROMPT_LENGTH=4096
MAX_RESPONSE_LENGTH=4096

export HYDRA_FULL_ERROR=1
export WANDB_MODE="disabled"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILES \
    data.train_batch_size=$BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=24 \
    data.truncation='right' \
    reward_model.reward_manager='dapo' \
    custom_reward_function.path=rl/rlpgr.py \
    custom_reward_function.name=compute_reward \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.dtype='bfloat16' \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.disable_log_stats=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    trainer.default_local_dir=ckpts/${EXPERIMENT_NAME} \
    trainer.critic_warmup=0 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.val_before_train=False \
    trainer.test_freq=50 \
    trainer.total_training_steps=300
