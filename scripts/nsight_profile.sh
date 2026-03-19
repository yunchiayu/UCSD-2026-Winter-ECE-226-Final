
SCRIPT_DIR="$(dirname "$0")"
PRJ_DIR="$SCRIPT_DIR/.."

# ------ Python script: ------
PY="$PRJ_DIR/gpu_profile/nsight_profile.py"

MODEL="Qwen/Qwen2.5-3B-Instruct"
SUM_SEQ_LEN=1024
GEN_SEQ_LEN=64
BATCH_SIZE=1
WARMUP_ITERS=20
SEED=42

export CUDA_VISIBLE_DEVICES=1 # set gpu id  

OUTPUT_DIR="$PRJ_DIR/results/nsight_profile/$MODEL/batch-size-$BATCH_SIZE/sum-seq-len-$SUM_SEQ_LEN/gen-seq-len-$GEN_SEQ_LEN"
mkdir -p $OUTPUT_DIR

MODEL_NAME_CLEANED=$(echo $MODEL | tr "/" "-")
OUTPUT_FILE="$OUTPUT_DIR/nsys_result-$MODEL_NAME_CLEANED-batch-size-$BATCH_SIZE-sum-seq-len-$SUM_SEQ_LEN-gen-seq-len-$GEN_SEQ_LEN"

nsys profile \
    -o $OUTPUT_FILE \
    --force-overwrite=true \
    --trace=cuda,nvtx \
    --stop-on-exit=true \
    --cuda-memory-usage=true \
    python "$PY" \
    --model $MODEL \
    --sum-seq-len $SUM_SEQ_LEN \
    --gen-seq-len $GEN_SEQ_LEN \
    --batch-size $BATCH_SIZE \
    --warmup-iters $WARMUP_ITERS \
    --seed $SEED