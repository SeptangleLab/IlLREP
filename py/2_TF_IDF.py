
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix, save_npz

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import StratifiedShuffleSplit
import joblib

DATA_DIR = Path("data")
SRC = DATA_DIR / "segments_clean.csv"

DESCRIPTIVE = {
    "territorial": [
        "district", "back home", "home state", "county", "southern border",
        "statehouse", "state legislature", "folks back home"
    ],
    "socio_cultural": [
        "workers"
    ],
    "occupation": [
        "veteran", "law enforcement", "sheriff", "doctor", "dentist",
        "principal", "researcher", "attorney", "police", "mom"
    ],
    "demographics": [
        "people","white","women","woman","men","parents","families","students",
        "children","kids","black","asian","tribal","christians"
    ],
    "constituencies": [
        "small businesses", "border patrol", "veterans", "taxpayers"
    ],
}

SUBSTANTIVE = {
    "disruptive": [
        "object","defund","dismantle","weaponized","fraud","stolen","deep state",
        "unelected bureaucrats","censorship","subpoena","protect","refuse"
    ],
    "maximalist": [
        "zero","end"
    ],
    "policy_issue": [
        # keep given list + fix common variants
        "seurity", "security", "border security", "illegal", "alien", "aliens",
        "immigration", "migrants", "rape", "crime", "insurance", "inflation",
        "fentanyl", "cartels", "cartel", "trafficking", "crisis", "gas", "taxes",
        "budget", "debt", "spending", "drugs", "overdoses", "schools", "school",
        "rights", "gender", "ideology", "health care", "medicare",
        "social security", "censorship", "401k", "china", "life", "abortion", "dhs"
    ],
    "advocacy": [
        "introduce","bill","demand","press","push","request","accountable",
        "subpoena","oversight","defend","protect","block","stop"
    ],
    "conventional_coop": [
        "bipartisan","enacted"
    ],
    "general": [
        "law","laws"
    ],
    "own_states": [
        "georgia","colorado","arizona","florida","south carolina"
    ],
    "red_states": [
        "alabama","alaska","arkansas","florida","idaho","indiana","iowa","kansas",
        "kentucky","louisiana","mississippi","missouri","montana","nebraska",
        "north dakota","oklahoma","south carolina","south dakota","tennessee",
        "texas","utah","west virginia","wyoming"
    ],
    "blue_states": [
        "california","colorado","connecticut","delaware","hawaii","illinois",
        "maine","maryland","massachusetts","new jersey","new mexico","new york",
        "oregon","rhode island","vermont","washington"
    ],
    "contested_states": [
        "arizona","georgia","michigan","minnesota","nevada","new hampshire",
        "north carolina","ohio","pennsylvania","virginia","wisconsin"
    ],
}

# Text vectoriser
TFIDF_PARAMS = dict(
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
    lowercase=True,
    token_pattern=r"(?u)\b\w+\b",
)

# Dictionary vectoriser
DICT_VECT_PARAMS = dict(
    ngram_range=(1, 3),
    lowercase=True,
    token_pattern=r"(?u)\b\w+\b",
)

SAMPLE_FRAC = 0.05   # 5%
SAMPLE_RANDOM_STATE = 42

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["clean_text"] = df["clean_text"].fillna("").astype(str).str.lower()
    if "speaker" not in df.columns:
        raise ValueError("Expected 'speaker' column in segments_clean.csv")
    return df


def flatten_dict(d):
    """Return flat list of all terms (lower-case, unique) from nested dict."""
    out = []
    for _, terms in d.items():
        out.extend([t.lower() for t in terms])

    seen = set()
    uniq = []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def dict_match_flags(texts: pd.Series, buckets: dict) -> pd.DataFrame:
    """
    For each bucket (subclass), return binary presence (>=1 hit) in each text.
    Uses regex with word boundaries; supports multi-word phrases.
    """
    data = {}
    for bucket, terms in buckets.items():
        term_pats = [rf"\b{re.escape(t.lower())}\b" for t in terms]
        pat = re.compile("|".join(term_pats))
        data[bucket] = texts.str.contains(pat, regex=True)
    return pd.DataFrame(data, index=texts.index).astype(int)


def build_labels(desc_flags: pd.DataFrame, subst_flags: pd.DataFrame) -> pd.DataFrame:
    D = (desc_flags.sum(axis=1) > 0).astype(int)
    S = (subst_flags.sum(axis=1) > 0).astype(int)
    H = ((D == 1) & (S == 1)).astype(int)
    return pd.DataFrame({"D": D, "S": S, "H": H})


def build_vectorizers_and_matrices(df: pd.DataFrame,
                                   tfidf_params: dict,
                                   dict_vect_params: dict,
                                   dict_terms: list):
    tfidf = TfidfVectorizer(**tfidf_params)
    X_text = tfidf.fit_transform(df["clean_text"])

    dict_vect = CountVectorizer(vocabulary=dict_terms, **dict_vect_params)
    X_dict = dict_vect.fit_transform(df["clean_text"])

    X_model = hstack([X_text, X_dict], format="csr")
    return tfidf, dict_vect, X_text, X_dict, X_model


def make_stratified_sample(df: pd.DataFrame, labels: pd.DataFrame, frac=0.05, seed=42):
    bucket = np.where(labels["H"] == 1, "H",
              np.where(labels["D"] == 1, "D",
              np.where(labels["S"] == 1, "S", "N")))
    df = df.copy()
    df["bucket"] = bucket

    # join with speaker to form strata
    strata = df["speaker"].astype(str).str.lower() + "||" + df["bucket"]
    n = len(df)
    test_size = max(1, int(round(frac * n)))

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx = np.arange(n)
    try:
        train_idx, sample_idx = next(sss.split(idx.reshape(-1, 1), strata))
    except ValueError:
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, sample_idx = next(sss2.split(idx.reshape(-1,1), df["bucket"]))
    return df.iloc[sample_idx].drop(columns=["bucket"])


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data(SRC)

    desc_flags = dict_match_flags(df["clean_text"], DESCRIPTIVE)
    subst_flags = dict_match_flags(df["clean_text"], SUBSTANTIVE)

    labels = build_labels(desc_flags, subst_flags)

    labels_out = pd.concat([df[["date", "speaker", "source_file"]], labels, desc_flags.add_prefix("desc_"), subst_flags.add_prefix("subst_"), df[["clean_text"]]], axis=1)
    labels_out.to_csv(DATA_DIR / "labels.csv", index=False)

    dict_terms = flatten_dict(DESCRIPTIVE) + flatten_dict(SUBSTANTIVE)
    # ensure unique
    dict_terms = list(dict.fromkeys([t.lower() for t in dict_terms]))

    tfidf, dict_vect, X_text, X_dict, X_model = build_vectorizers_and_matrices(
        df, TFIDF_PARAMS, DICT_VECT_PARAMS, dict_terms
    )

    save_npz(DATA_DIR / "X_text_tfidf.npz", csr_matrix(X_text))
    save_npz(DATA_DIR / "X_dict_terms.npz", csr_matrix(X_dict))
    save_npz(DATA_DIR / "X_model.npz", csr_matrix(X_model))
    joblib.dump({"tfidf": tfidf, "dict_vect": dict_vect, "dict_terms": dict_terms}, DATA_DIR / "vectorizers.joblib")

    with open(DATA_DIR / "dictionaries.json", "w", encoding="utf-8") as f:
        json.dump({"descriptive": DESCRIPTIVE, "substantive": SUBSTANTIVE}, f, indent=2, ensure_ascii=False)

    sample = make_stratified_sample(df, labels, frac=SAMPLE_FRAC, seed=SAMPLE_RANDOM_STATE)
    sample.to_csv(DATA_DIR / "sample_5pct.csv", index=False)

    n = len(df)
    print(f"Segments loaded: {n}")
    print(f"Labels: D={labels['D'].sum()}  S={labels['S'].sum()}  H={labels['H'].sum()}")
    print(f"Saved labels to: {DATA_DIR/'labels.csv'}")
    print(f"Vector shapes: X_text={X_text.shape}, X_dict={X_dict.shape}, X_model={X_model.shape}")
    print(f"Saved matrices: X_text_tfidf.npz, X_dict_terms.npz, X_model.npz")


if __name__ == "__main__":
    main()
