
SCRIPT_DIR="$(dirname "$0")"
PRJ_DIR="$SCRIPT_DIR/.."

# ------ Python script: ------
PY="$PRJ_DIR/plot/plot_model_ttft_and_peak_memory.py"

OUTPUT_DIR="$PRJ_DIR/figures/model_ttft_and_peak_memory"
# MODEL="Qwen/Qwen2.5-3B-Instruct"
MODEL="state-spaces/mamba-2.8b-hf"
# MODEL="fla-hub/rwkv7-2.9B-world"



export CUDA_VISIBLE_DEVICES=1 # set gpu id

python "$PY" \
    --output-dir $OUTPUT_DIR \
    --model $MODEL
