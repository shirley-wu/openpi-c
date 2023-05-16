#!/usr/bin/env python
# coding: utf-8
import argparse
import json
import logging
import os
import sys

import torch
from tqdm import tqdm

from training.data_reader import BaselineEntityStateConstructor
from training.data_reader.parse_utils_emem import parse_question, parse_answer
from training.gen_ans_to_list import aggregate_predictions
from training.models import MODEL_CLASSES
from training.split_util import split_parts

# to avoid "src.xxx" not found error.
sys.path.insert(0, '..')

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)


def map_bool(x):
    return {"y": True, "True": True, "n": False, "False": False}[x]


class OpenPIPredictor:
    def __init__(self, args, model_type: str, model_path: str, stop_token: str = "<|endoftext|>"):
        self.args = args

        assert model_type.startswith("bart")
        _, model_class, tokenizer_class = MODEL_CLASSES[model_type]
        self.stop_token = stop_token
        self.tokenizer = tokenizer_class.from_pretrained(model_path)
        self.model = model_class.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info(f"Loaded model for generation.")

    def get_predictions(self, max_len, input_ctxt_and_query, memory_update_info=None, gen_kwargs=None):
        title, end_states, current_step = input_ctxt_and_query
        question_ids = parse_question(title, end_states, current_step, self.tokenizer, block_size=1024)
        return self.generate_nexttokens_for_sent(max_len=max_len,
                                                 text_so_far="",
                                                 encoded_prompt=torch.LongTensor([question_ids, ]).to(self.device),
                                                 memory_update_info=memory_update_info,
                                                 gen_kwargs=gen_kwargs if gen_kwargs is not None else {})

    def generate_nexttokens_for_sent(self,
                                     max_len: int,
                                     text_so_far: str,
                                     encoded_prompt: torch.Tensor,
                                     memory_update_info,
                                     gen_kwargs: dict):
        '''
        :param text_so_far: text generated so far.
        :param encoded_prompt: `tf.Tensor` of `dtype=tf.int32` of shape `(batch_size, sequence_length)`
                The sequence used as a prompt for the generation. If `None` the method initializes
                it as an empty `tf.Tensor` of shape `(1,)`.
        :return: generated next token.
        '''
        answer: str = ""
        with torch.no_grad():
            out = self.model.generate(
                input_ids=encoded_prompt,
                max_length=min(1024, max_len + encoded_prompt.size(-1)),
                memory_update_info=memory_update_info,
                **gen_kwargs,
            )
            assert len(out) == 1

            for out_seq in out:
                text = self.tokenizer.decode(out_seq, clean_up_tokenization_spaces=True)
                if text.startswith(text_so_far):  # True for GPT-style model, False for seq2seq-style model
                    text = text[len(text_so_far):]
                text = text.replace("</s>", "").replace("<s>", "").strip()  # clean-up for BART
                if self.stop_token is not None and self.stop_token in text:
                    text = text[: text.find(self.stop_token)]
                answer += text

        return {"answer": answer, }

    def update_memory(self, question, answer, memory_update_info):
        # Parse question
        title, end_states, current_step = question
        question_ids = parse_question(title, end_states, current_step, self.tokenizer, block_size=1024)
        # Parse answer
        answer_ids, answer_segment_ids = parse_answer(answer, self.tokenizer, block_size=1024)
        # Forward
        input_kwargs = dict(
            input_ids=torch.LongTensor([question_ids, ]).to(self.device),
            memory_update_info=memory_update_info,
            decoder_input_ids=torch.LongTensor([answer_ids, ]).to(self.device),
            labels_segment_ids=torch.LongTensor([answer_segment_ids, ]).to(self.device),
        )
        outs = self.model.model(**input_kwargs)
        return self.model.model.get_memory_update_info(input_kwargs, outs, memory_update_info)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default="gpt2")
    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        help="model path",
        required=True
    )
    parser.add_argument(
        "--max_len",
        default=400,
        type=int,
    )
    parser.add_argument(
        "--stop_token",
        type=str,
        default='<|endoftext|>',
        help="model path",
    )
    parser.add_argument(
        "--test_input_file",
        default=None,
        type=str,
        help="jsonl file containing id (str) and question (str) keys",
        required=True
    )
    parser.add_argument(
        "--outpath",
        default=None,
        type=str,
        help="path to store unformatted model predictions",
        required=True
    )

    parser.add_argument('--temperature', default=1.0, type=float)
    parser.add_argument('--top_k', default=0, type=int)
    parser.add_argument('--top_p', default=1.0, type=float)
    parser.add_argument('--do_sample', default=False, type=map_bool)
    parser.add_argument('--num_return_sequences', default=1, type=int)
    parser.add_argument('--num_beams', default=4, type=int)
    parser.add_argument('--no_repeat_ngram_size', default=0, type=int)
    parser.add_argument('--early_stopping', default=False, type=map_bool)
    parser.add_argument('--length_penalty', default=1.0, type=float)
    parser.add_argument('--repetition_penalty', default=1.0, type=float)

    parser.add_argument('--use_gold_answers', default=False, action="store_true")

    args = parser.parse_args()
    if not os.path.exists(os.path.dirname(args.outpath)):
        args.outpath = os.path.join(args.model_path, args.outpath)
    if args.stop_token == "None":
        print("Stop token is None")
        args.stop_token = None

    gen_kwargs = {}
    for k in ['temperature', 'top_k', 'top_p', 'do_sample', 'num_return_sequences', 'num_beams', 'no_repeat_ngram_size',
              'early_stopping', 'length_penalty', 'repetition_penalty', ]:
        v = getattr(args, k)
        print(k, v)
        gen_kwargs[k] = v

    if not args.model_path or not os.path.exists(args.model_path):
        print(
            f"WARNING: Not performing any non_batched generation "
            f"because generation model file/dir does not exist: {args.model_path}")
        return

    if not args.test_input_file or not os.path.exists(args.test_input_file):
        print(
            f"WARNING: Not performing any non_batched generation "
            f"because generation input file does not exist: {args.test_input_file}")
        return

    if not args.outpath:
        print(
            f"WARNING: Not performing any non_batched generation "
            f"because generation output file is empty: {args.outpath}")
        return

    args.model_path = args.model_path.strip()
    args.outpath = args.outpath.strip()
    args.test_input_file = args.test_input_file.strip()

    print(f"Generation task, input = {args.test_input_file}, output = {args.outpath} ...")

    predictor = OpenPIPredictor(args, model_type=args.model_type, model_path=args.model_path,
                                stop_token=args.stop_token)

    with open(args.test_input_file, 'r') as open_file:
        lines = open_file.readlines()

    last_id = None
    memory_update_info = None
    constructor_cls = BaselineEntityStateConstructor
    constructor = constructor_cls()
    with open(args.outpath, 'w') as open_file:
        for i, item in tqdm(enumerate(lines), total=len(lines)):
            if i in [1, 5, 10, 100, 200, 400, ]:
                open_file.flush()
            item = json.loads(item)
            title = item['id'][0].split('.com/')[1].replace("-", " ") + "."
            if item['id'][0] != last_id:
                last_id = item['id'][0]
                memory_update_info = None
                constructor = constructor_cls()

            query = constructor.construct_and_update_question(item['query'])
            output = predictor.get_predictions(
                input_ctxt_and_query=[title, ] + query,
                memory_update_info=memory_update_info, max_len=args.max_len, gen_kwargs=gen_kwargs,
            )

            if args.use_gold_answers:
                answer = constructor.construct_and_update_answer(item['answers'])
            else:
                answer_parts = []
                for ans in output['answer'].split(', '):
                    ans = split_parts(ans.strip(), normalize=False)
                    if ans is not None:
                        answer_parts.append(ans)
                answer = constructor.construct_and_update_answer(answer_parts)

            memory_update_info = predictor.update_memory([title, ] + query, answer, memory_update_info)

            output['id'] = item['id'][0] + "||" + str(item['id'][1])
            json.dump(output, open_file)
            open_file.write('\n')

    formatted_fp = args.outpath + ".formatted.jsonl"
    logger.info(f"Done generating. Aggregating and formatting to {formatted_fp}")
    aggregate_predictions(prediction_fp=args.outpath, out_fp=formatted_fp)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
    )
    with torch.no_grad():
        main()
