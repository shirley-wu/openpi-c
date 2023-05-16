from eval.eval_util import normalize_and_stem, normalize_nostem

split_parts_cache = dict()


def split_parts(s: str, normalize=True, strict=True):
    if (s, normalize) in split_parts_cache:
        return split_parts_cache[(s, normalize)]

    s_orig = s
    if normalize:
        s = normalize_nostem(s)

    def get_first_part(s, kw):
        s = s.split(kw)
        return s[0], kw.join(s[1:])

    attr, s = get_first_part(s, ' of ')
    entity, s = get_first_part(s, ' was ')
    state_prev, s = get_first_part(s, ' before and ')
    state_post, s = get_first_part(s, ' afterwards')

    if strict and (len(attr) == 0 or len(entity) == 0 or len(state_prev) == 0 or len(state_post) == 0):
        print("Warning: invalid string: %s" % s_orig)
        ret = None
    else:
        ret = [entity, attr, state_prev, state_post]
        if normalize:
            ret = [normalize_and_stem(x) for x in ret]

    split_parts_cache[(s_orig, normalize)] = ret

    return ret
