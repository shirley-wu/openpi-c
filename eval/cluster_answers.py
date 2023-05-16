import json

import numpy as np
import torch
import torch.nn as nn
import tqdm
from sentence_transformers import SentenceTransformer


class BertSentenceSimilarity:
    def __init__(self):
        self.model = SentenceTransformer('stsb-distilroberta-base-v2').cuda()
        self.similarity = nn.CosineSimilarity(dim=2, eps=1e-6).cuda()

    def __call__(self, answers):
        if len(answers) == 0:
            return torch.zeros(0, 0)

        embeddings = self.model.encode(answers)
        embeddings = torch.tensor(embeddings).cuda()
        return self.similarity(embeddings[:, None, :], embeddings[None, :, :])


def strict_connected_components(edges):
    edges = edges.cpu().numpy()
    n = edges.shape[0]
    n_clusters = 0
    labels = np.repeat(-1, n)
    for i in range(n):
        for j in range(n_clusters):
            if np.all(edges[i][labels == j]):
                labels[i] = j
                break
        if labels[i] < 0:
            labels[i] = n_clusters
            n_clusters += 1
    return n_clusters, labels


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate OpenPI predictions.')
    parser.add_argument("prediction")
    parser.add_argument("--th", default=0.7, type=float)
    args = parser.parse_args()

    with open(args.prediction) as f:
        all_predictions = [json.loads(line) for line in f]
    all_answers = [sorted(set(line['answers'])) for line in all_predictions]

    similarity = BertSentenceSimilarity()

    n_clusters = 0
    n_total = 0
    for i in tqdm.trange(len(all_answers)):
        scores = similarity(all_answers[i]).cpu()
        n_labels, labels = strict_connected_components(scores > args.th)

        all_predictions[i]['answers'] = all_answers[i]
        all_predictions[i]['answer_clusters'] = labels.tolist()

        n_clusters += n_labels
        n_total += len(all_answers[i])

    with open(args.prediction + ".clustered", 'w') as f:
        for pred in all_predictions:
            f.write(json.dumps(pred) + '\n')

    print("Total: %d clusters for %d answers (%.2f%%). %.2f clusters for each answer in average" % (
        n_clusters, n_total, n_clusters / n_total * 100, n_clusters / len(all_answers)
    ))


if __name__ == '__main__':
    main()
