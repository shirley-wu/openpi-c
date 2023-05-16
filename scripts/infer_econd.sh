INPUT_PRED=$1
MODEL_TYPE=${2:-"bart"}
MODEL_NAME=${3:-"facebook/bart-large"}

EXP=exps/econd_${MODEL_NAME//\//-}
export PYTHONPATH=.

python scripts/econd/convert_prediction_to_econd_input.py ${INPUT_PRED}

python training/generation_econd.py --model_path $EXP --model_type "$MODEL_TYPE" \
    --test_input_file ${INPUT_PRED}.econd-input.jsonl --outpath gen-out

python scripts/econd/convert_econd_output_to_prediction.py $EXP/gen-out.formatted.jsonl
python eval/simple_eval.py -g data/test.jsonl -p $EXP/gen-out.formatted.jsonl.pred