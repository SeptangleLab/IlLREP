
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
from scipy.sparse import hstack
DATA_DIR = Path("data")
LABELS_FILE = DATA_DIR / "labels.csv"
MODEL_FILE = DATA_DIR / "model_artifacts.joblib"
PLOT_OUTPUT_DIR = DATA_DIR / "analysis_plots"
PLOT_OUTPUT_DIR.mkdir(exist_ok=True)

TOP_N_FEATURES = 15 # customsie when needed or whatever. this file isn't meant for replcation, bit you're welcome to use the given resources.

df = pd.read_csv(LABELS_FILE)
df["clean_text"] = df["clean_text"].fillna("").astype(str)
artifacts = joblib.load(MODEL_FILE)

tfidf = artifacts["tfidf"]
dict_vect = artifacts["dict_vect"]
clf = artifacts["clf"]
dict_terms = artifacts["dict_terms"]
y = artifacts["labels"]

tfidf_features = tfidf.get_feature_names_out()
dict_features = dict_vect.get_feature_names_out()
all_features = np.concatenate([tfidf_features, dict_features])

X_all = hstack([
    tfidf.transform(df["clean_text"]),
    dict_vect.transform(df["clean_text"])
])
df["pred_label"] = clf.predict(X_all)

speaker_counts = df.groupby(["speaker", "pred_label"]).size().reset_index(name="count")
speaker_totals = speaker_counts.groupby("speaker")["count"].transform("sum")
speaker_counts["pct"] = (speaker_counts["count"] / speaker_totals * 100).round(2)

plt.figure(figsize=(10, 6))
sns.barplot(data=speaker_counts, x="speaker", y="pct", hue="pred_label")
plt.ylabel("% of segments")
plt.title("Distribution of Predicted Labels by Speaker")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(PLOT_OUTPUT_DIR / "speaker_distribution.png", dpi=300)
plt.close()

if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    monthly = df.groupby([pd.Grouper(key="date", freq="ME"), "pred_label"]).size().reset_index(name="count")
    monthly_totals = monthly.groupby("date")["count"].transform("sum")
    monthly["pct"] = (monthly["count"] / monthly_totals * 100).round(2)

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=monthly, x="date", y="pct", hue="pred_label", marker="o")
    plt.ylabel("% of segments")
    plt.title("Monthly Share of Predicted Labels")
    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT_DIR / "time_trends.png", dpi=300)
    plt.close()

desc_cols = [c for c in df.columns if c.startswith("desc_")]
subst_cols = [c for c in df.columns if c.startswith("subst_")]

desc_cov = (df[desc_cols].sum() / len(df) * 100).round(2).sort_values()
subst_cov = (df[subst_cols].sum() / len(df) * 100).round(2).sort_values()

plt.figure(figsize=(8, 5))
sns.barplot(x=desc_cov.values, y=desc_cov.index, hue=desc_cov.index, palette="Blues_r", legend=False)
plt.xlabel("% of corpus")
plt.ylabel("Descriptive subclass")
plt.title("Descriptive Subclass Coverage")
plt.tight_layout()
plt.savefig(PLOT_OUTPUT_DIR / "desc_subclass_coverage.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
sns.barplot(x=subst_cov.values, y=subst_cov.index, hue=subst_cov.index, palette="Reds_r", legend=False)
plt.xlabel("% of corpus")
plt.ylabel("Substantive subclass")
plt.title("Substantive Subclass Coverage")
plt.tight_layout()
plt.savefig(PLOT_OUTPUT_DIR / "subst_subclass_coverage.png", dpi=300)
plt.close()

subclass_cols = desc_cols + subst_cols
cooc_matrix = pd.DataFrame(0, index=subclass_cols, columns=subclass_cols)
for i in subclass_cols:
    for j in subclass_cols:
        cooc_matrix.loc[i, j] = ((df[i] == 1) & (df[j] == 1)).sum()

plt.figure(figsize=(12, 10))
sns.heatmap(cooc_matrix, annot=False, cmap="coolwarm", cbar=True)
plt.title("Subclass Co-occurrence Heatmap")
plt.tight_layout()
plt.savefig(PLOT_OUTPUT_DIR / "subclass_cooccurrence_heatmap.png", dpi=300)
plt.close()

def top_terms_for_class(class_label, top_n=TOP_N_FEATURES):
    class_index = list(clf.classes_).index(class_label)
    coefs = clf.coef_[class_index]
    top_idx = np.argsort(coefs)[::-1][:top_n]
    return [(all_features[i], coefs[i]) for i in top_idx]

for label in clf.classes_:
    top_terms = top_terms_for_class(label, TOP_N_FEATURES)
    terms, weights = zip(*top_terms)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(weights), y=list(terms), hue=list(terms), palette="viridis", legend=False)
    plt.title(f"Top {TOP_N_FEATURES} Features for Class {label}")
    plt.xlabel("Weight")
    plt.ylabel("Term")
    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT_DIR / f"top_features_{label}.png", dpi=300)
    plt.close()
