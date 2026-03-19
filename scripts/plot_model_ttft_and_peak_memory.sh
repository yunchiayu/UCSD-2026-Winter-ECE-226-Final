
SCRIPT_DIR="$(dirname "$0")"
PRJ_DIR="$SCRIPT_DIR/.."

# ------ Python script: ------
PY="$PRJ_DIR/plot/plot_model_ttft_and_peak_memory.py"

OUTPUT_DIR="$PRJ_DIR/results/torch_profile_finegrained"
# MODEL="Qwen/Qwen2.5-3B-Instruct"
MODEL="state-spaces/mamba-2.8b-hf"



export CUDA_VISIBLE_DEVICES=1 # set gpu id

python "$PY" \
    --output-dir $OUTPUT_DIR \
    --model $MODEL
