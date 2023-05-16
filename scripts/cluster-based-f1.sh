PRED=$1

python eval/cluster_answers.py $PRED
python eval/simple_eval_by_cluster.py -g data/test.jsonl.clustered -p $PRED.clustered