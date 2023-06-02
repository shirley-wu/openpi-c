# Introduction

This repository is for our ACL2023 findings paper:

[OpenPI-C: A Better Benchmark and Stronger Baseline for Open-Vocabulary State Tracking](https://arxiv.org/abs/2306.00887)

The code is based on the original OpenPI dataset: https://allenai.org/data/openpi

## Dataset

OpenPI Dataset files are available in JSON format under `data`. There are four files: `{train,dev,test}.jsonl` and `test.jsonl.clustered`.

The three files `{train,dev,test}.jsonl` are of the same format representing the training, development and test sets respectively. Each line is a json representing a data point, i.e., one step in a process. An example is as follows:
```json
{
  "id": [
    "www.wikihow.com/Stop-a-Mosquito-Bite-from-Itching-Using-a-Spoon",
    1
  ],
  "query": "It\u2019s always a good idea to disinfect an area that has been bitten or stung by an insect.",
  "answers": [
    [
      "skin",
      "cleanness",
      "clean",
      "covered in disinfectant"
    ],
    [
      "disinfectant",
      "location",
      "in bottle",
      "on bite"
    ]
  ]
}
```
The json contains three fields:
1. `id`: a two-tuple. The first item of the tuple is the ID for the process, and the second item represents which step it is in the process.
2. `query`: the description of the current step.
3. `answers`: the 4-tuples state tracking results. This field is a list where each item is a 4-tuple representing (entity, attribute, pre-state, post-state).

To facilitate cluster-based F1 evaluation, we pre-cluster `test.jsonl` into `test.jsonl.clustered`. It's basically the same as `test.jsonl` except an additional field named `answer_clusters`.
The field is a list of integers with the same length as `answers`. Each item in `answer_clusters` represents which cluster the corresponding item in `answers` belongs to.

## Environment Setup

To set up the environment, it's recommended to create a new conda environment as follows:
```bash
conda create -y -n openpi-c python=3.8
conda activate openpi-c
```

Then install `pytorch==1.7.0` as in https://pytorch.org/get-started/previous-versions/. For example, with CUDA 10.1:
```bash
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.1 -c pytorch
```

Finally, install the requirements via pip:
```bash
pip install -r requirements.txt
```

Unfortunately, due to version conflict, this environment doesn't support installing `sentence-transformers` and thus doesn't support cluster-based F1 evaluation. We need to create a separate environment for cluster-based evaluation. The requirements for this environment are in `requirements.cluster-f1.txt`. I'll recommend creating the environment as follows:
```bash
conda create -y -n cluster-f1 python=3.8
conda activate cluster-f1
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.1 -c pytorch # depending on your cuda version
pip install -r requirements.cluster-f1.txt
```

Before running cluster-based F1 scripts (as in [Cluster-based F1](#cluster-based-f1)), you should first activate this `cluster-f1` environment.

## Training and Evaluation

### BART

For quickly reproducing our experiments, you can run the following scripts:
```bash
bash scripts/train_baseline.sh  # for training BART baseline
bash scripts/train_concat-states.sh  # for training BART+concat states model
bash scripts/train_econd.sh  # for training ECond model
bash scripts/train_emem.sh  # for training EMem model
```

Similarly, for evaluating BART baseline, BART+concat states and EMem model, simply run:
```bash
bash scripts/infer_baseline.sh  # for evaluating BART baseline
bash scripts/infer_concat-states.sh  # for evaluating BART+concat states model
bash scripts/infer_emem.sh  # for evaluating EMem model
```

However, for evaluating BART+ECond, you first need to run evaluation for the BART baseline.
Then, the generation outputs would be at `exps/baseline_facebook-bart-large/gen-out.formatted.jsonl`.
Then run evaluation for BART+ECond as follows:
```bash
bash scripts/infer_econd.sh exps/baseline_facebook-bart-large/gen-out.formatted.jsonl
```

Similarly, for evaluating BART+EMem+ECond, first run evaluation for BART+EMem and get the generation outputs at `exps/emem_facebook-bart-large/gen-out.formatted.jsonl`.
Then, run evaluation for BART+EMem+ECond as follows:
```bash
bash scripts/infer_econd.sh exps/emem_facebook-bart-large/gen-out.formatted.jsonl
```

### Cluster-based F1

Unfortunately, due to the version conflict mentioned in the [Environment Setup](#environment-setup) section, previous scripts are unable to produce cluster-based F1 numbers.
To calculate cluster-based F1 results, first activate the corresponding environment such as:
```bash
conda activate cluster-f1
```

Then run `scripts/cluster-based-f1.sh` based on the generation outputs. For example, for the BART baseline:
```bash
bash scripts/cluster-based-f1.sh exps/baseline_facebook-bart-large/gen-out.formatted.jsonl
```
