import copy
import json
import os

import torch
from torch.utils.data import Dataset

from .parse_utils_emem import parse_example


class BaselineEntityStateConstructor:
    def __init__(self):
        self.past_queries = []
        self.states = []
        self.inds = dict()

    def construct_and_update_question(self, query):
        self.past_queries.append(query)
        return [[], ' '.join(self.past_queries) + ' Now, what happens?', ]

    def construct_and_update_answer(self, answer):
        # Update answer
        for ent, attr, _, post_state in answer:
            if (ent, attr) not in self.inds:
                self.inds[(ent, attr)] = len(self.states)
                self.states.append([ent, attr, post_state])
            else:
                self.states[self.inds[(ent, attr)]] = [ent, attr, post_state]

        # Construct answer
        if len(answer) == 0:
            answer_ret = [("There will be no change.", -1), ]
        else:
            answer_ret = []
            for ent, attr, pre_state, post_state in answer:
                answer_ret.append((f"{attr} of {ent} was {pre_state} before and {post_state} afterwards,",
                                   self.inds[(ent, attr)]))
            if len(answer) > 0:
                answer_ret[-1] = (answer_ret[-1][0][:-1], answer_ret[-1][1])

        return answer_ret

    def construct_and_update_question_answer(self, query, answer):
        question_ret = self.construct_and_update_question(query)
        answer_ret = self.construct_and_update_answer(answer)
        return question_ret, answer_ret


class OpenPIEMem(Dataset):
    constructor_cls = BaselineEntityStateConstructor

    def __init__(self, tokenizer, file_path='train', block_size=512):
        self.tokenizer = tokenizer
        self.block_size = block_size

        self.data = []
        constructor = self.constructor_cls()
        last_id = None
        history = []
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                input_json = json.loads(line)
                if input_json['id'][0] != last_id:
                    last_id = input_json['id'][0]
                    constructor = self.constructor_cls()
                    history = []

                question, answer = constructor.construct_and_update_question_answer(
                    input_json['query'], input_json['answers']
                )

                title = input_json['id'][0].split('.com/')[1].replace("-", " ") + "."

                self.data.append(([title, ] + question, answer, copy.copy(history)))
                history.append(len(self.data) - 1)

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
        ret = self._getitem_no_history(item_idx)
        history_inds = self.data[item_idx][2]
        history_data = []
        for i in history_inds:
            history_data.append(self._getitem_no_history(i))
        ret["history_data"] = history_data
        return ret

    def _getitem_no_history(self, item_idx):
        question, answer, _ = self.data[item_idx]
        ret = parse_example(question, answer, self.tokenizer, self.block_size)
        ret = {k: torch.tensor(v) for k, v in ret.items()}
        ret["metadata"] = torch.tensor([item_idx])
        return ret
