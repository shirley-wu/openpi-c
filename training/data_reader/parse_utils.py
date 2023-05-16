def parse_question(tokenizer, question, block_size=None):
    question_ids = []
    for iq, q in enumerate(question):
        if len(q) > 0:
            q = tokenizer.encode(q)[:-1]
            if tokenizer._bos_token is not None:
                q = q[1:]
            question_ids += q
    question_ids = question_ids + [tokenizer.eos_token_id, ]
    if tokenizer._bos_token is not None:
        question_ids = [tokenizer.bos_token_id, ] + question_ids

    if block_size is not None:
        # truncate and handle special tokens
        question_ids = question_ids[:-1][-(block_size - 1):] + [question_ids[-1], ]
    return question_ids


def parse_example(question, answer, tokenizer, block_size):
    if isinstance(question, str):
        question = [question, ]
    assert answer != ""

    if isinstance(block_size, int):
        input_block_size = output_block_size = block_size
    else:
        input_block_size, output_block_size = block_size

    # Question
    question_ids = parse_question(tokenizer, question, block_size=input_block_size)
    # pad
    question_mask = [1, ] * len(question_ids) + [0, ] * (input_block_size - len(question_ids))
    question_ids = question_ids + [tokenizer.pad_token_id, ] * (input_block_size - len(question_ids))

    # Answer
    labels = tokenizer.encode(answer)
    # truncate and handle special tokens
    if tokenizer._bos_token is not None:
        labels = labels[1:]  # remove <BOS>
    labels = labels[:-1][:output_block_size - 1] + [labels[-1], ]  # truncate
    answer = [labels[-1], ] + labels[:-1]  # <EOS> as prepending token
    # pad
    answer_mask = [1, ] * len(answer) + [0, ] * (output_block_size - len(answer))
    labels = labels + [-100, ] * (output_block_size - len(answer))
    answer = answer + [tokenizer.pad_token_id, ] * (output_block_size - len(answer))

    ret = dict(
        input_ids=question_ids,
        attention_mask=question_mask,
        decoder_input_ids=answer,
        decoder_attention_mask=answer_mask,
        lm_labels=labels,
    )
    return ret
