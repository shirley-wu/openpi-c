import torch
from torch import nn
from torch.nn import functional as F


class AttnAggregator(nn.Module):
    def __init__(self, hidden_states):
        super().__init__()
        self.linear = nn.Linear(hidden_states, 1)

    def forward(self, states):
        scores = self.linear(states).squeeze(dim=1)
        scores = F.softmax(scores, dim=-1)
        return (states * scores[:, None]).sum(dim=0)


def get_bsz(input_ids, encoder_outputs):
    if input_ids is not None:
        bsz = input_ids.shape[0]
    else:
        import pdb
        pdb.set_trace()
        assert encoder_outputs is not None
        bsz = encoder_outputs[0].shape[0]
    return bsz


def fill_dummy_memory(memory_update_info, config, bsz):
    feat_dim = config.d_model
    if memory_update_info is None:  # Initialized "empty" memory
        memory_update_info = [torch.zeros((0, feat_dim)).cuda() for _ in range(bsz)]
    return memory_update_info


def augment_decoder_cache_with_memory(decoder_outputs, memory_update_info):
    if len(decoder_outputs) >= 2:
        assert decoder_outputs[1] is not None
        cache = (decoder_outputs[1], memory_update_info)
        return decoder_outputs[:1] + (cache,) + decoder_outputs[2:]
    return decoder_outputs


def merge_states(*states_to_merge):
    merged_states = []
    for states in states_to_merge:
        if len(states) == 2 and (states[1] is None or states[1].dtype == torch.long):
            states, mask = states
            if mask is None:
                states = list(states)
            else:
                states = [s[~m] for s, m in zip(states, mask)]
        for i in range(len(states)):
            if i == len(merged_states):
                merged_states.append([])
            merged_states[i].append(states[i])

    merged_states = [torch.cat(s, dim=0) for s in merged_states]

    max_len = max(s.shape[0] for s in merged_states)
    ret_mask = torch.stack([
        torch.cat([torch.ones(s.shape[0], ), torch.zeros(max_len - s.shape[0], ), ], dim=0)
        for s in merged_states
    ], dim=0).long().cuda()
    ret_states = torch.stack([
        torch.cat([s, torch.zeros((max_len - s.shape[0], s.shape[1])).cuda(), ], dim=0)
        for s in merged_states
    ], dim=0)

    return ret_states, ret_mask
