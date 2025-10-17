import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")

SRC = "data/segments.csv"
DEST = "data/segments_clean.csv"

df = pd.read_csv(SRC)

stop_words = set(stopwords.words("english"))
punct = set(string.punctuation)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    return " ".join(tokens)

df["clean_text"] = df["text"].apply(clean_text)

df.to_csv(DEST, index=False)
print("Done")
