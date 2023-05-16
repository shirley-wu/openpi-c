MODEL_TYPE=${1:-"bart-emem"}
MODEL_NAME=${2:-"facebook/bart-large"}

EXP=exps/emem_${MODEL_NAME//\//-}
export PYTHONPATH=.

python training/generation_emem.py --model_path $EXP --model_type "$MODEL_TYPE" \
  --test_input_file data/test.jsonl --outpath gen-out

python eval/simple_eval.py -g data/test.jsonl -p $EXP/gen-out.formatted.jsonl