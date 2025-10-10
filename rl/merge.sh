step=$1
EXPERIMENT_NAME="rl"
LOCAL_DIR=ckpts/${EXPERIMENT_NAME}/global_step_${step}/actor
TARGET_DIR=ckpts/${EXPERIMENT_NAME}/merged_${step}


python ./rl/model_merger.py merge \
       --backend fsdp \
       --local_dir ${LOCAL_DIR} \
       --target_dir ${TARGET_DIR}