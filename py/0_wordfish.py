
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

nltk.download("stopwords")

SRC = "data/segments.csv"
CLEANED = "data/segments_clean.csv"
PROJ_CSV = "data/speaker_wordfish_projection.csv"

df = pd.read_csv(SRC)

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    return " ".join(tokens)

df["clean_text"] = df["text"].astype(str).apply(clean_text)
df.to_csv(CLEANED, index=False)
print(f"✓ Cleaned text saved to {CLEANED}")

vectorizer = CountVectorizer(min_df=2)
X = vectorizer.fit_transform(df["clean_text"])

print(f"✓ Vectorized {X.shape[0]} documents and {X.shape[1]} terms")
pca = PCA(n_components=1)
positions = pca.fit_transform(X.toarray())

df["wordfish_position"] = positions[:, 0]

speaker_positions = df.groupby("speaker")["wordfish_position"].mean().reset_index()
speaker_positions = speaker_positions.sort_values("wordfish_position")

speaker_positions.to_csv(PROJ_CSV, index=False)
print(f"✓ Speaker projections saved to {PROJ_CSV}")

# Plotting
plt.figure(figsize=(10, 5))
plt.barh(speaker_positions["speaker"], speaker_positions["wordfish_position"], color="slateblue")
plt.xlabel("Wordfish Dimension (1st PC)")
plt.title("Estimated Speaker Positions via Wordfish (PCA)")
plt.tight_layout()
plt.savefig("data/wordfish_projection.png", dpi=300)
plt.show()


# component weights for PC1
pc1 = pca.components_[0]
terms = vectorizer.get_feature_names_out()

top_idx = pc1.argsort()
top_negative = [(terms[i], pc1[i]) for i in top_idx[:15]]
top_positive = [(terms[i], pc1[i]) for i in reversed(top_idx[-15:])]

print("\n🔹 Top words driving the NEGATIVE direction:")
for word, weight in top_negative:
    print(f"{word:15s} {weight:.4f}")

print("\n🔹 Top words driving the POSITIVE direction:")
for word, weight in top_positive:
    print(f"{word:15s} {weight:.4f}")
