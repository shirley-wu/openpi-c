import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('inp')
parser.add_argument('--gold_fn', default='data/test.jsonl')
args = parser.parse_args()

gold_ids = []
grouped = dict()
with open(args.gold_fn) as f:
    for line in f:
        line = json.loads(line)
        id_ = "{:s}||{:d}".format(*line['id'])
        gold_ids.append(id_)
        grouped[id_] = []

with open(args.inp) as f:
    for line in f:
        line = json.loads(line)
        id_, _ = line['id'].split("__")
        if line['answers'] != ["There will be no change.", ]:
            grouped[id_] += line['answers']

with open(args.inp + '.pred', 'w') as f:
    for id_ in gold_ids:
        f.write(json.dumps({
            "id": id_,
            "answers": grouped[id_],
        }) + '\n')
