import json
import os

import torch
from torch.utils.data import Dataset

from .parse_utils import parse_example


class Constructor:
    def __init__(self):
        self.past_queries = []

    def update_query(self, query):
        self.past_queries.append(query)

    def update_answers(self, answer):
        pass

    def construct_question(self, query):
        raise NotImplementedError


class RegularConstructor(Constructor):
    def construct_question(self, query):
        return [' '.join(self.past_queries), query + ' Now, what happens?', ]


class ConcatStatesConstructor(Constructor):
    def __init__(self):
        super(ConcatStatesConstructor, self).__init__()
        self.past_answers = []

    def update_answers(self, answer):
        if len(answer) == 0:
            answer = "There will be no change."
        else:
            answer = ["{1} of {0} was {2} before and {3} afterwards".format(*parts) for parts in answer]
            answer = ', '.join(answer)
        self.past_answers.append(answer)

    def construct_question(self, query):
        return [' '.join([q + ' ' + a for q, a in zip(self.past_queries, self.past_answers)]),
                query + ' Now, what happens?', ]


CONSTRUCTOR_CLASSES = {"regular": RegularConstructor, "concat-states": ConcatStatesConstructor}


class OpenPIDatasetBase(Dataset):
    constructor_name = None

    def __init__(self, tokenizer, file_path='train', block_size=512):
        self.tokenizer = tokenizer
        self.block_size = block_size

        self.data = []
        constructor_cls = CONSTRUCTOR_CLASSES[self.constructor_name]
        constructor = constructor_cls()
        last_id = None
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                input_json = json.loads(line)
                if input_json['id'][0] != last_id:
                    last_id = input_json['id'][0]
                    constructor = constructor_cls()

                question = constructor.construct_question(input_json['query'])
                if len(input_json['answers']) == 0:
                    answer = "There will be no change."
                else:
                    answer = ["{1} of {0} was {2} before and {3} afterwards".format(*parts)
                              for parts in input_json['answers']]
                    answer = ', '.join(answer)

                constructor.update_query(input_json['query'])
                constructor.update_answers(input_json['answers'])

                title = input_json['id'][0].split('.com/')[1].replace("-", " ") + "."

                self.data.append(([title, ] + question, answer))

        print("Example data from", os.path.basename(file_path))
        for i in range(10):
            print("# {:d}".format(i))
            print("Question:\n\t", self.data[i][0])
            print("Answer:\n\t", self.data[i][1])
            print()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item_idx):
        question, answer = self.data[item_idx]
        ret = parse_example(question, answer, self.tokenizer, self.block_size)
        ret = {k: torch.tensor(v) for k, v in ret.items()}
        ret["metadata"] = torch.tensor([item_idx])
        return ret


class OpenPIDataset(OpenPIDatasetBase):
    constructor_name = "regular"


class OpenPIDatasetConcatStates(OpenPIDatasetBase):
    constructor_name = "concat-states"
