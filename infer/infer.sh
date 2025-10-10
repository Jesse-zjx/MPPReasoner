MODEL_PATH=$1
EXPERIMENT_NAME="rl"


python ./infer/infer_new.py \
    --model Qwen2.5-VL-7B-Instruct \
    --model_path "${MODEL_PATH}" \
    --data_dir "./data/test" \
    --dataset bace,bbbp,sider,hiv,bioavailability_ma,cyp2c9_veith,cyp2d6_veith,ames \
    --gpus 0,1,2,3 \
    --batch_size 256 \
    --temperature 0.0 \
    --output_dir "./results/${EXPERIMENT_NAME}" \
    --use_images True

