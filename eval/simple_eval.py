import json
import os
from typing import List, Tuple

from eval.eval_util import get_content_from_predicted_effect
from eval.generation_metric import GenerationMetric, ExactMetric, BLEUMetric, ROUGEMetric


def f1_emnlp2020(
        predictions: List[str],
        gold: List[str],
        generation_metric: GenerationMetric) -> Tuple[float, float, float]:
    if len(gold) == 0 and len(predictions) == 0:
        return (1.0, 1.0, 1.0)
    if len(gold) == 0 and len(predictions) > 0:
        return (0.0, 1.0, 0.0)
    if len(predictions) == 0:
        return (1.0, 0.0, 0.0)

    predictions = list(set(predictions))

    # Compute precision score based on best gold match for each prediction
    tp = 0.0  # true positive score
    for p in predictions:
        best_gold_match = 0.0
        matched_gold = "n/a"
        norm_p = get_content_from_predicted_effect(p)
        for g in gold:
            norm_g = get_content_from_predicted_effect(g)
            gold_match = generation_metric.match_score(gold=norm_g, predicted=norm_p)
            if gold_match > best_gold_match:
                best_gold_match = gold_match
                matched_gold = g
        tp += best_gold_match
    precision = tp / len(predictions)

    # Compute recall score based on best prediction match for each gold
    tr = 0.0
    for g in gold:
        norm_g = get_content_from_predicted_effect(g)
        best_prediction_match = 0.0
        for p in predictions:
            norm_p = get_content_from_predicted_effect(p)
            prediction_match = generation_metric.match_score(gold=norm_g, predicted=norm_p)
            if prediction_match > best_prediction_match:
                best_prediction_match = prediction_match
        tr += best_prediction_match
    recall = tr / len(gold)

    f1_denominator = precision + recall

    if f1_denominator == 0:
        return (0.0, 0.0, 0.0)
    return (precision, recall, 2 * precision * recall / (precision + recall))


def evaluate(all_predictions, all_gold_answers, generation_metric: GenerationMetric) -> dict:
    assert len(all_predictions) == len(all_gold_answers)

    metric_main_p_sum = 0.0
    metric_main_r_sum = 0.0
    metric_main_f1_sum = 0.0

    for predictions, gold_answers in zip(all_predictions, all_gold_answers):
        if len(predictions["answers"]) == 1 and predictions["answers"][0].lower().strip().startswith(
                "there will be no change"):
            predictions["answers"] = []
        assert predictions['id'] == gold_answers['id']

        # Main metric
        (p, r, f1) = f1_emnlp2020(
            predictions=predictions["answers"],
            gold=gold_answers["answers"],
            generation_metric=generation_metric,
        )
        metric_main_p_sum += p
        metric_main_r_sum += r
        metric_main_f1_sum += f1

    return {
        "main_P": '%.2f' % (100.0 * metric_main_p_sum / len(all_predictions)),
        "main_R": '%.2f' % (100.0 * metric_main_r_sum / len(all_predictions)),
        "main_F1": '%.2f' % (100.0 * metric_main_f1_sum / len(all_predictions)),
    }


def format_gold_answer(gold_answer):
    gold_answer['id'] = "{:s}||{:d}".format(*gold_answer['id'])
    gold_answer['answers'] = sorted(set([
        f"{attr} of {ent} was {statepre} before and {statepost} afterwards"
        for ent, attr, statepre, statepost in gold_answer['answers']
    ]))
    return gold_answer


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate OpenPI predictions.')
    parser.add_argument('--gold-file', '-g', help='Filename with gold answers', required=True)
    parser.add_argument('--prediction-file', '-p', help='Filename with predictions', required=True)
    args = parser.parse_args()

    if not args.gold_file or not os.path.exists(args.gold_file):
        print(f"WARNING: Not performing any evaluation because input gold file does not exist: {args.gold_file}")
        return

    if not args.prediction_file or not os.path.exists(args.prediction_file):
        print(f"WARNING: Not performing any evaluation because prediction file does not exist: {args.prediction_file}")
        return

    with open(args.prediction_file) as f:
        all_predictions = [json.loads(line) for line in f]
    with open(args.gold_file) as f:
        all_gold_answers = [format_gold_answer(json.loads(line)) for line in f]

    generation_metrics = [
        ExactMetric(),
        BLEUMetric(),
        ROUGEMetric()
    ]

    all_metrics = dict()
    formatted_scores = []

    for metric_num, current_metric in enumerate(generation_metrics):
        print(f"Evaluating current metric ({1 + metric_num}/{len(generation_metrics)}) : {current_metric.name()} ...")
        current_metric_score = evaluate(all_predictions=all_predictions,
                                        all_gold_answers=all_gold_answers,
                                        generation_metric=current_metric)

        for k, v in current_metric_score.items():
            # prepare all metrics as json entries.
            all_metrics[f"{k.replace('main_', '')}_{current_metric.name()}"] = v
        formatted_scores.append(f"{current_metric.name()}"
                                f"\t{current_metric_score['main_P']}"
                                f"\t{current_metric_score['main_R']}"
                                f"\t{current_metric_score['main_F1']}")

    print(f"\n\n================================\n Evaluation results original F1 \n"
          "================================")
    print(f"Predictions: {args.prediction_file}")
    print(f"Gold: {args.gold_file}")
    print(f"\n\t\tprec\trecall\tf1")
    for fs in formatted_scores:
        print(fs)


if __name__ == '__main__':
    main()
