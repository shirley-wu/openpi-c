MODEL_TYPE=${1:-"bart"}
MODEL_NAME=${2:-"facebook/bart-large"}

WEIGHT_DECAY="0.0"
WARMUP="0"
LR="5e-05"
ACCUM="2"
NUM_EPOCHS="7"
MAX_LEN="20"
BATCH_SIZE_TRAIN="4"
BATCH_SIZE_EVAL="16"
BLOCK_SIZE="300"

EXP=exps/econd_${MODEL_NAME//\//-}
mkdir -p $EXP

PYTHONPATH=$PYTHONPATH:. python training/run_trainer.py \
    --seed 44 \
    --output_dir=$EXP \
    --warmup_steps $WARMUP \
    --data_type econd \
    --model_type="$MODEL_TYPE" \
    --model_name_or_path=${MODEL_NAME} \
    --do_train \
    --train_data_file=data/train.jsonl \
    --per_gpu_train_batch_size $BATCH_SIZE_TRAIN \
    --per_gpu_eval_batch_size $BATCH_SIZE_EVAL \
    --gradient_accumulation_steps $ACCUM \
    --overwrite_output_dir \
    --length $MAX_LEN \
    --block_size "$BLOCK_SIZE" \
    --save_total_limit 3 \
    --save_steps 1000 \
    --learning_rate $LR \
    --weight_decay $WEIGHT_DECAY \
    --num_train_epochs $NUM_EPOCHS \
    --overridden_model_configs '{"resid_pdrop": 0.1, "attn_dropout": 0.1}'