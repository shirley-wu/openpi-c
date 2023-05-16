def parse_question(title, end_states, current_step, tokenizer, block_size: int):
    question_ids = []

    # Title
    q = tokenizer.encode(title)[:-1]
    if tokenizer._bos_token is not None:
        q = q[1:]
    question_ids += q

    # End states
    for q, iq in end_states:
        if len(q) > 0:
            q = tokenizer.encode(q)[:-1]
            if tokenizer._bos_token is not None:
                q = q[1:]
            question_ids += q

    # Current step
    q = tokenizer.encode(current_step)[:-1]
    if tokenizer._bos_token is not None:
        q = q[1:]
    question_ids += q

    # BOS & EOS
    question_ids = question_ids + [tokenizer.eos_token_id, ]
    if tokenizer._bos_token is not None:
        question_ids = [tokenizer.bos_token_id, ] + question_ids

    # Truncate and handle special tokens
    question_ids = question_ids[:-1][-(block_size - 1):] + [question_ids[-1], ]

    return question_ids


def parse_answer(answer, tokenizer, block_size: int):
    answer_ids = []
    segment_ids = []

    # End states
    for a, ia in answer:
        assert len(a) > 0
        a = tokenizer.encode(a)[:-1]
        if tokenizer._bos_token is not None:
            a = a[1:]
        answer_ids += a
        segment_ids += [ia + 1, ] * len(a)

    # Append EOS
    answer_ids = answer_ids + [tokenizer.eos_token_id, ]
    segment_ids = segment_ids + [0, ]

    # Truncate and handle special tokens
    answer_ids = answer_ids[:-1][-(block_size - 1):] + [answer_ids[-1], ]
    segment_ids = segment_ids[:-1][-(block_size - 1):] + [segment_ids[-1], ]

    return answer_ids, segment_ids


def parse_example(question, answer, tokenizer, block_size):
    if isinstance(block_size, int):
        input_block_size = output_block_size = block_size
    else:
        input_block_size, output_block_size = block_size

    # Question
    title, end_states, current_step = question
    question_ids = parse_question(title, end_states, current_step, tokenizer, input_block_size)
    # pad
    question_mask = [1, ] * len(question_ids) + [0, ] * (input_block_size - len(question_ids))
    question_ids = question_ids + [tokenizer.pad_token_id, ] * (input_block_size - len(question_ids))

    # Answer
    answer_labels, answer_segment_ids = parse_answer(answer, tokenizer, block_size=output_block_size)
    answer = [answer_labels[-1], ] + answer_labels[:-1]  # <EOS> as prepending token
    # pad
    answer_mask = [1, ] * len(answer) + [0, ] * (output_block_size - len(answer))
    labels = answer_labels + [-100, ] * (output_block_size - len(answer))
    answer_segment_ids = answer_segment_ids + [0, ] * (output_block_size - len(answer_segment_ids))
    answer = answer + [tokenizer.pad_token_id, ] * (output_block_size - len(answer))

    ret = dict(
        input_ids=question_ids,
        attention_mask=question_mask,
        decoder_input_ids=answer,
        decoder_attention_mask=answer_mask,
        lm_labels=labels,
        labels_segment_ids=answer_segment_ids,
    )
    return ret
