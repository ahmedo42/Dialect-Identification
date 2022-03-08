import re
import pandas as pd
import string

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def normalize_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'EMOJI', text)

def normalize_url(text):
    url_pattern = re.compile(r"https?://\S+|www\.\S+")
    return url_pattern.sub(r"URL", text)

def normalize_mentions(text):
  mention_pattern = re.compile(r"/\B@\w+/g")
  return mention_pattern.sub(r"MENTION",text)

def preprocess(text):
    text = remove_punctuation(text)
    text = normalize_emoji(text)
    text = normalize_url(text)
    text = normalize_mentions(text)
    tokens = text.split()
    return ' '.join(tokens)