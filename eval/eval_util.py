import string

from eval.porter import PorterStemmer

porter_stemmer = PorterStemmer()


def normalize_nostem(s: str) -> str:
    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s.strip())))


# Evaluation speed is important.
# Stemming cache for orders of magnitude faster computation
# e.g., stemming of gold phrases was repeated during precision computation
# and stemming of prediction phrases was repeated during recall computation
# The cache is supposed to be very small (at most vocab size map: word-> stem).
# Phrases are generally not queried and even if they were, they wouldn't blow
# up computation at eval time.
cache = dict()
leads = ["a ", "an ", "the ", "your ", "his ", "their ", "my ", "another ", "other ", "this ", "that "]


def stem(w: str):
    if not w or len(w.strip()) == 0:
        return ""
    w_lower = w.lower()
    if w_lower in cache:
        return cache[w_lower]
    # Remove leading articles from the phrase (e.g., the rays => rays).
    for lead in leads:
        if w_lower.startswith(lead):
            w_lower = w_lower[len(lead):]
            break
    # Porter stemmer: rays => ray
    if not w_lower or len(w_lower.strip()) == 0:
        return ""
    ans = porter_stemmer.stem(w_lower).strip()
    cache[w_lower] = ans
    return ans


def normalize_and_stem(s: str) -> str:
    def stemmed(text):
        stemmed_tokens = []
        for token in text.split():
            token_stem = stem(token)
            if token_stem:
                stemmed_tokens.append(token_stem)
        return ' '.join(stemmed_tokens)

    return stemmed(normalize_nostem(s))


EFFECT_STOP_WORDS = {"and", "was", "is", "before", "afterwards", "after", "of"}


def get_content_from_predicted_effect(s):
    def is_stop(cand: str):
        return cand.lower() in EFFECT_STOP_WORDS

    words_prev = normalize_nostem(s).split(" ")
    words = []
    for word in words_prev:
        if not is_stop(word):
            words.append(word)
    return ' '.join(words)
