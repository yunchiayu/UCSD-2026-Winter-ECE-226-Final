
SCRIPT_DIR="$(dirname "$0")"
PRJ_DIR="$SCRIPT_DIR/.."

# ------ Python script: ------
PY="$PRJ_DIR/gpu_profile/torch_profile.py"

MODEL="Qwen/Qwen2.5-3B-Instruct"
SUM_SEQ_LEN=1024
GEN_SEQ_LEN=64
BATCH_SIZE=1
WARMUP_ITERS=20
SEED=42

export CUDA_VISIBLE_DEVICES=1 # set gpu id

python "$PY" \
    --model $MODEL \
    --sum-seq-len $SUM_SEQ_LEN \
    --gen-seq-len $GEN_SEQ_LEN \
    --batch-size $BATCH_SIZE \
    --warmup-iters $WARMUP_ITERS \
    --seed $SEED