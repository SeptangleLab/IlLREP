
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.metrics import classification_report
from scipy.sparse import hstack
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = Path("data")
LABELS_FILE = DATA_DIR / "labels.csv"
MODEL_FILE = DATA_DIR / "model_artifacts.joblib"

TOP_N_FEATURES = 15
PLOT_OUTPUT_DIR = DATA_DIR / "analysis_plots"
PLOT_OUTPUT_DIR.mkdir(exist_ok=True)

print("Loading data...")
df = pd.read_csv(LABELS_FILE)
artifacts = joblib.load(MODEL_FILE)
df["clean_text"] = df["clean_text"].fillna("").astype(str)

tfidf = artifacts["tfidf"]
dict_vect = artifacts["dict_vect"]
clf = artifacts["clf"]
dict_terms = artifacts.get("dict_terms", None)
y = pd.Series(artifacts["labels"])

assert len(df) == len(y), f"Row mismatch: df={len(df)} vs labels={len(y)}"

tfidf_features = tfidf.get_feature_names_out()
dict_features = dict_vect.get_feature_names_out()
all_features = np.concatenate([tfidf_features, dict_features])

mask_nonN = (y != "N")
df_nonN = df.loc[mask_nonN].copy()
y_nonN = y.loc[mask_nonN].reset_index(drop=True)
df_nonN = df_nonN.reset_index(drop=True)

print(f"\nCorpus (gold) size: {len(df)}; Neutral (N) removed: {len(df) - len(df_nonN)}")
print(f"Non‑neutral corpus size: {len(df_nonN)}")

print("\n=== Classification Report (Non‑Neutral Only: D, H, S) ===")
X_nonN = hstack([
    tfidf.transform(df_nonN["clean_text"]),
    dict_vect.transform(df_nonN["clean_text"])
])
y_pred_nonN = clf.predict(X_nonN)

labels_eval = [c for c in ["D", "H", "S"] if c in np.unique(y_nonN)]
print(classification_report(
    y_nonN, y_pred_nonN,
    labels=labels_eval,
    target_names=labels_eval,
    digits=3
))

desc_cols = [c for c in df_nonN.columns if c.startswith("desc_")]
subst_cols = [c for c in df_nonN.columns if c.startswith("subst_")]

print("\n-- Descriptive subclass coverage (% of non‑N corpus) --")
if desc_cols:
    desc_cov = (df_nonN[desc_cols].sum() / len(df_nonN) * 100).round(2)
    print(desc_cov.sort_values(ascending=False))
else:
    print("[no desc_* columns found]")

print("\n-- Substantive subclass coverage (% of non‑N corpus) --")
if subst_cols:
    subst_cov = (df_nonN[subst_cols].sum() / len(df_nonN) * 100).round(2)
    print(subst_cov.sort_values(ascending=False))
else:
    print("[no subst_* columns found]")

def top_terms_for_class(class_label, top_n=TOP_N_FEATURES):
    # model was trained on all classes; just display D/H/S slices
    class_index = list(clf.classes_).index(class_label)
    coefs = clf.coef_[class_index]
    top_idx = np.argsort(coefs)[::-1][:top_n]
    return [(all_features[i], coefs[i]) for i in top_idx]

classes_to_show = [c for c in clf.classes_ if c in ["D", "H", "S"]]
print("\n=== Top Features Per Class (Non‑Neutral focus) ===")
for label in classes_to_show:
    print(f"\n-- {label} --")
    for term, weight in top_terms_for_class(label, TOP_N_FEATURES):
        print(f"{term:25} {weight:.4f}")

print("\nGenerating speaker distribution plot (non‑N only)...")
df_pred = df_nonN.copy()
df_pred["pred_label"] = y_pred_nonN

speaker_counts = (
    df_pred
    .groupby(["speaker", "pred_label"])
    .size()
    .reset_index(name="count"))

speaker_counts = speaker_counts[speaker_counts["pred_label"].isin(["D", "H", "S"])]

speaker_totals = speaker_counts.groupby("speaker")["count"].transform("sum")
speaker_counts["pct"] = (speaker_counts["count"] / speaker_totals * 100).round(2)

plt.figure(figsize=(10, 6))
sns.barplot(data=speaker_counts, x="speaker", y="pct", hue="pred_label")
plt.ylabel("% of segments")
plt.title("Distribution of Predicted Labels by Speaker (Non‑Neutral Corpus)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(PLOT_OUTPUT_DIR / "speaker_distribution_nonN.png", dpi=300)
plt.close()

if "date" in df_pred.columns:
    print("Generating time trend plot (non‑N only)...")
    df_pred["date"] = pd.to_datetime(df_pred["date"], errors="coerce")
    monthly = (
        df_pred
        .groupby([pd.Grouper(key="date", freq="M"), "pred_label"])
        .size()
        .reset_index(name="count")
    )
    monthly = monthly[monthly["pred_label"].isin(["D", "H", "S"])]
    monthly_totals = monthly.groupby("date")["count"].transform("sum")
    monthly["pct"] = (monthly["count"] / monthly_totals * 100).round(2)

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=monthly, x="date", y="pct", hue="pred_label", marker="o")
    plt.ylabel("% of segments")
    plt.title("Monthly Share of Predicted Labels (Non‑Neutral Corpus)")
    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT_DIR / "time_trends_nonN.png", dpi=300)
    plt.close()

print("\n=== Subclass Co‑occurrence Matrix (Non‑Neutral Corpus) ===")
subclass_cols = desc_cols + subst_cols
if subclass_cols:
    cooc_matrix = pd.DataFrame(0, index=subclass_cols, columns=subclass_cols, dtype=int)
    for i in subclass_cols:
        for j in subclass_cols:
            cooc_matrix.loc[i, j] = ((df_nonN[i] == 1) & (df_nonN[j] == 1)).sum()
    print(cooc_matrix)
else:
    print("[no desc_* or subst_* columns found]")

print("\n✓ Analysis complete (non‑N). Plots saved to", PLOT_OUTPUT_DIR)
