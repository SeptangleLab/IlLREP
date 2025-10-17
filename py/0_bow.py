
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

SRC = "data/segments_clean.csv"
TOP_N = 30  # change if needed
df = pd.read_csv(SRC)

df["clean_text"] = df["clean_text"].fillna("")
df = df[df["clean_text"].str.strip().astype(bool)]

# Vectorise
vectorizer = CountVectorizer(min_df=1)
X = vectorizer.fit_transform(df["clean_text"])

# Sum and Rank
word_counts = X.sum(axis=0).A1  # convert to flat array
vocab = vectorizer.get_feature_names_out()

bow_freq = pd.DataFrame({"word": vocab, "count": word_counts})
bow_freq = bow_freq.sort_values("count", ascending=False)


print(f"Top {TOP_N} most frequent words:\n")
print(bow_freq.head(TOP_N).to_string(index=False))
bow_freq.to_csv("data/bow_word_frequencies.csv", index=False)
print(f"\n✓ Full BoW frequencies saved to data/bow_word_frequencies.csv")
