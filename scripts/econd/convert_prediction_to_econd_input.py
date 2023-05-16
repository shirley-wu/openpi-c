import argparse
import json

from training.data_reader.openpi import RegularConstructor
from training.split_util import split_parts


def get_entity_to_state_change_questions(id_, question, entities):
    question = ' '.join(q for q in question if q != '')
    assert question.endswith("Now, what happens?")
    question = question[:-len("Now, what happens?")]
    ret = []
    for entity in entities:
        ret.append({
            "id": id_ + '__' + entity.replace(" ", "-"),
            "question": question + "Now, what happens to {}?".format(entity)
        })
    return ret


def get_gold_questions(gold):
    questions = []
    last_id = None
    constructor = RegularConstructor()
    for item in gold:
        title = item['id'][0].split('.com/')[1].replace("-", " ") + "."
        if item['id'][0] != last_id:
            last_id = item['id'][0]
            constructor = RegularConstructor()
        questions.append([title, ] + constructor.construct_question(item['query']))
        constructor.update_query(item['query'])
    return questions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('pred_fn')
    parser.add_argument('--gold_fn', default='data/test.jsonl')
    args = parser.parse_args()

    pred_fn = args.pred_fn
    gold_fn = args.gold_fn

    with open(gold_fn) as f:
        gold = [json.loads(line) for line in f]
    with open(pred_fn) as f:
        pred = [json.loads(line) for line in f]
    assert len(pred) == len(gold)

    pred_entities = []
    for p in pred:
        entities = set()
        for ans in p['answers']:
            ret = split_parts(ans, normalize=False)
            if ret is not None:
                entities.add(ret[0])
        pred_entities.append(sorted(entities))

    gold_questions = get_gold_questions(gold)

    questions = []
    for p, e, q in zip(pred, pred_entities, gold_questions):
        questions += get_entity_to_state_change_questions(p['id'], q, e)

    with open(pred_fn + '.econd-input.jsonl', 'w') as f:
        for q in questions:
            f.write(json.dumps(q) + '\n')
