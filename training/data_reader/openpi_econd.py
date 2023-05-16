import json
import logging
import os

import torch
from torch.utils.data import Dataset

from .openpi import RegularConstructor
from .parse_utils import parse_example

logger = logging.getLogger(__name__)


class OpenPIECond(Dataset):
    def __init__(self, tokenizer, file_path='train', block_size=512):
        self.tokenizer = tokenizer

        self.data = []
        constructor_cls = RegularConstructor
        constructor = constructor_cls()
        last_id = None
        with open(file_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                input_json = json.loads(line)
                if input_json['id'][0] != last_id:
                    last_id = input_json['id'][0]
                    constructor = constructor_cls()

                question = constructor.construct_question(input_json['query'])
                constructor.update_query(input_json['query'])

                title = input_json['id'][0].split('.com/')[1].replace("-", " ") + "."

                entities = sorted(set(x[0] for x in input_json['answers']))

                for entity in entities:
                    answer = [
                        f"{attr} of {ent_} was {statepre} before and {statepost} afterwards"
                        for ent_, attr, statepre, statepost in input_json['answers'] if ent_ == entity
                    ]
                    self.data.append([[title, ] + question, entity, answer, i, ])

        self.block_size = block_size

        print("Example data from", os.path.basename(file_path))
        for i in range(10):
            print("# {:d}".format(i))
            print("Question:\n\t", self.data[i][0])
            print("Answer:\n\t", self.data[i][1])
            print("History indexes:\n\t", self.data[i][2])
        print("\nExample tensors (of index 9):")
        for k, v in self[9].items():
            print('-' * 10, k)
            print(v)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item_idx):
        question, entity, answers, main_idx = self.data[item_idx]
        assert question[-1].endswith("Now, what happens?")
        entity_question = question[-1][:-len("Now, what happens?")] + "Now, what happens to {}?".format(entity)
        question = question[:-1] + [entity_question, ]
        answer = ", ".join(answers)
        ret = self.parse_example(question, answer, self.tokenizer, self.block_size)
        ret = {k: torch.tensor(v) for k, v in ret.items()}
        ret["main_idx"] = torch.tensor([main_idx])
        ret["metadata"] = torch.tensor([item_idx])
        return ret

    def parse_example(self, question, answer, tokenizer, block_size):
        return parse_example(question, answer, tokenizer, block_size)
