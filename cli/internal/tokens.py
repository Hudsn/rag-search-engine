import nltk.stem as stem
import string


def tokenize_text(text: str, stemmer: stem.PorterStemmer, stopwords: list[str]) -> list[str]:
    text = text.lower()
    text = str.translate(text, str.maketrans("", "", string.punctuation))
    toks = text.lower().split(None)
    toks = remove_stops(toks, stopwords)
    toks = list(map(lambda entry: stemmer.stem(entry), toks))
    return toks

def remove_stops(tokens: list[str], stops: list[str]) -> list[str]:
    return list(filter(lambda entry: entry not in stops, tokens))

def is_token_match(q_tokens: list[str], t_tokens: list[str]) -> bool:
    for qtok in q_tokens:
        for ttok in t_tokens:
            if qtok in ttok:
                return True
    return False