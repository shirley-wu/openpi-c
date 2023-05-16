from eval.eval_metric.bleu.bleu import Bleu
from eval.eval_metric.rouge.rouge import Rouge
from eval.eval_util import normalize_and_stem


class GenerationMetric:
    metric = None

    def match_score(self, gold: str, predicted: str):
        if not gold and not predicted:
            # both empty then return 1.0
            return 1.0

        if not gold:  # gold empty then return 0.0
            return 0.0

        if not predicted:  # predicted empty then return 0.0
            return 0.0

        return self.compute_score(gold=gold, predicted=predicted)

    def compute_score(self, gold: str, predicted: str):
        pass

    def name(self):
        pass


class ExactMetric(GenerationMetric):
    def __init__(self):
        self.metric = None

    def compute_score(self, gold: str, predicted: str):
        return 1.0 if normalize_and_stem(gold) == normalize_and_stem(predicted) else 0.0

    def name(self):
        return "ExactMetric"


class BLEUMetric(GenerationMetric):
    def __init__(self, n: int = 4):
        self.metric = Bleu(n)

    def compute_score(self, gold: str, predicted: str):
        # Reference: gold , Hypothesis: predicted
        score, score_info = self.metric.compute_score(
            gts={0: [normalize_and_stem(gold)]},
            res={0: [normalize_and_stem(predicted)]}
        )
        # return average of Bleu_1, Bleu_2, Bleu_3, Bleu_4
        # return sum(score)/len(score)/100.0
        return score[1] / 100.0

    def name(self):
        return "BLEUMetric"


class ROUGEMetric(GenerationMetric):
    def __init__(self):
        self.metric = Rouge()

    def compute_score(self, gold: str, predicted: str):
        # Reference: gold , Hypothesis: predicted
        score, score_info = self.metric.compute_score(
            gts={0: [normalize_and_stem(gold)]},
            res={0: [normalize_and_stem(predicted)]}
        )

        return score / 100.0

    def name(self):
        return "ROUGEMetric"
