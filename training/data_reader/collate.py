import torch


def collate(examples):
    def collate_history_samples(samples, ex):
        samples_ = []
        max_len = max(len(s) for s in samples)
        for s in samples:
            len_s = len(s)
            if len_s > 0:
                s = collate(s)
                for k, v in s.items():
                    v_pad = torch.zeros([max_len - len_s, ] + list(v.shape)[1:], dtype=v.dtype, device=v.device)
                    s[k] = torch.cat([v, v_pad, ], dim=0)
            else:
                s = {}
                for k, v in ex.items():
                    if k not in ['history_data', ]:
                        s[k] = torch.zeros([max_len, ] + list(v.shape), dtype=v.dtype, device=v.device)
            s["length"] = torch.LongTensor([len_s, ])
            samples_.append(s)
        return collate(samples_)

    def collate_individual(samples):
        length_of_first = samples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in samples)
        assert are_tensors_same_length, "Key {} different input size {}".format(key, [x.shape for x in samples])
        return torch.stack(samples, dim=0)

    ret = {}
    for key in examples[0].keys():
        samples = [example[key] for example in examples]
        if key == "history_data":
            history_ret = collate_history_samples(samples, examples[0])
            for k, v in history_ret.items():
                ret["history_" + k] = v
        else:
            ret[key] = collate_individual(samples)
    return ret
