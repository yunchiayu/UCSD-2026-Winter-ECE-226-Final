
SCRIPT_DIR="$(dirname "$0")"
PRJ_DIR="$SCRIPT_DIR/.."

# ------ Python script: ------
PY="$PRJ_DIR/gpu_simulation/gpu_sim.py"

MODEL="Qwen/Qwen2.5-3B-Instruct"
# MODEL="state-spaces/mamba2-2.7b"
MODEL_CONFIG_PATH="$PRJ_DIR/gpu_simulation/model_config/Qwen2.5-3B-Instruct.json"
HARDWARE_CONFIG_PATH="$PRJ_DIR/gpu_simulation/hardware_config/RTX4090.yaml"




SUM_SEQ_LEN=8192
GEN_SEQ_LEN=64
BATCH_SIZE=1


export CUDA_VISIBLE_DEVICES=1 # set gpu id

python "$PY" \
    --model $MODEL \
    --model-config-path $MODEL_CONFIG_PATH \
    --hardware-config-path $HARDWARE_CONFIG_PATH \
    --sum-seq-len $SUM_SEQ_LEN \
    --gen-seq-len $GEN_SEQ_LEN \
    --batch-size $BATCH_SIZE 
