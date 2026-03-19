
SCRIPT_DIR="$(dirname "$0")"
PRJ_DIR="$SCRIPT_DIR/.."

# ------ Python script: ------
PY="$PRJ_DIR/gpu_profile/torch_profile_finegrained.py"

MODEL_NAME="Qwen/Qwen2.5-3B-Instruct"

# MODEL_NAME="state-spaces/mamba2-2.7b"
# MODEL="state-spaces/mamba-2.8b"
# MODEL_NAME="state-spaces/mamba-130m"
# MODEL_NAME="state-spaces/mamba-2.8b-hf"
SUM_SEQ_LEN=8192
GEN_SEQ_LEN=64
BATCH_SIZE=1
WARMUP_ITERS=20
SEED=42

export CUDA_VISIBLE_DEVICES=0 # set gpu id

python "$PY" \
    --model-name $MODEL_NAME \
    --sum-seq-len $SUM_SEQ_LEN \
    --gen-seq-len $GEN_SEQ_LEN \
    --batch-size $BATCH_SIZE \
    --warmup-iters $WARMUP_ITERS \
    --seed $SEED