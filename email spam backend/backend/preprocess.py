import re
import string

_URL = re.compile(r"(https?://\S+|www\.\S+)")
_EMAIL = re.compile(r"\S+@\S+")
_NUM = re.compile(r"\d+")
_PUNCT = str.maketrans("", "", string.punctuation)


def clean_text(text: str) -> str:
    """Lower-case, replace URLs/emails/numbers with tokens, strip punctuation."""
    text = str(text).lower()
    text = _URL.sub(" urltoken ", text)
    text = _EMAIL.sub(" emailtoken ", text)
    text = _NUM.sub(" numtoken ", text)
    text = text.translate(_PUNCT)
    return " ".join(text.split())
