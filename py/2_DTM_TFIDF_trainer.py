
import re
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import joblib
DATA_DIR = Path("data")
SRC = DATA_DIR / "segments_clean.csv"

DESCRIPTIVE = {
    "territorial": [
        "district", "back home", "home state", "county", "southern border",
        "statehouse", "state legislature", "folks back home"
    ],
    "socio_cultural": ["workers"],
    "occupation": [
        "veteran", "law enforcement", "sheriff", "doctor", "dentist",
        "principal", "researcher", "attorney", "police", "mom"
    ],
    "demographics": [
        "people", "white", "women", "woman", "men", "parents", "families",
        "students", "children", "kids", "black", "asian", "tribal", "christians"
    ],
    "constituencies": ["small businesses", "border patrol", "veterans", "taxpayers"],
}

SUBSTANTIVE = {
    "disruptive": [
        "object", "defund", "dismantle", "weaponized", "fraud", "stolen",
        "deep state", "unelected bureaucrats", "censorship", "subpoena",
        "protect", "refuse"
    ],
    "maximalist": ["zero", "end"],
    "policy_issue": [
        "seurity", "security", "border security", "illegal", "alien", "aliens",
        "immigration", "migrants", "rape", "crime", "insurance", "inflation",
        "fentanyl", "cartels", "cartel", "trafficking", "crisis", "gas",
        "taxes", "budget", "debt", "spending", "drugs", "overdoses",
        "schools", "school", "rights", "gender", "ideology", "health care",
        "medicare", "social security", "censorship", "401k", "china",
        "life", "abortion", "dhs"
    ],
    "advocacy": [
        "introduce", "bill", "demand", "press", "push", "request",
        "accountable", "subpoena", "oversight", "defend", "protect",
        "block", "stop"
    ],
    "conventional_coop": ["bipartisan", "enacted"],
    "general": ["law", "laws"],
    "own_states": ["georgia", "colorado", "arizona", "florida", "south carolina"],
    "red_states": [
        "alabama", "alaska", "arkansas", "florida", "idaho", "indiana",
        "iowa", "kansas", "kentucky", "louisiana", "mississippi",
        "missouri", "montana", "nebraska", "north dakota", "oklahoma",
        "south carolina", "south dakota", "tennessee", "texas", "utah",
        "west virginia", "wyoming"
    ],
    "blue_states": [
        "california", "colorado", "connecticut", "delaware", "hawaii",
        "illinois", "maine", "maryland", "massachusetts", "new jersey",
        "new mexico", "new york", "oregon", "rhode island", "vermont",
        "washington"
    ],
    "contested_states": [
        "arizona", "georgia", "michigan", "minnesota", "nevada",
        "new hampshire", "north carolina", "ohio", "pennsylvania",
        "virginia", "wisconsin"
    ],
}

TFIDF_PARAMS = dict(
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
    lowercase=True,
    token_pattern=r"(?u)\b\w+\b",
)

DICT_VECT_PARAMS = dict(
    ngram_range=(1, 3),
    lowercase=True,
    token_pattern=r"(?u)\b\w+\b",
)

def flatten_dict(d):
    terms = []
    for v in d.values():
        terms.extend([t.lower() for t in v])
    return list(dict.fromkeys(terms))

def dict_match_flags(texts, buckets):
    data = {}
    for bucket, terms in buckets.items():
        pat = re.compile("|".join([rf"\b{re.escape(t.lower())}\b" for t in terms]))
        data[bucket] = texts.str.contains(pat, regex=True).astype(int)
    return pd.DataFrame(data, index=texts.index)

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SRC)
    df["clean_text"] = df["clean_text"].fillna("").astype(str).str.lower()

    desc_flags = dict_match_flags(df["clean_text"], DESCRIPTIVE)
    subst_flags = dict_match_flags(df["clean_text"], SUBSTANTIVE)

    D = (desc_flags.sum(axis=1) > 0).astype(int)
    S = (subst_flags.sum(axis=1) > 0).astype(int)
    y = np.where((D == 1) & (S == 0), "D",
        np.where((S == 1) & (D == 0), "S",
        np.where((S == 1) & (D == 1), "H", "N")))  # N = neither

    dict_terms = flatten_dict(DESCRIPTIVE) + flatten_dict(SUBSTANTIVE)
    tfidf = TfidfVectorizer(**TFIDF_PARAMS)
    X_tfidf = tfidf.fit_transform(df["clean_text"])

    dict_vect = CountVectorizer(vocabulary=dict_terms, **DICT_VECT_PARAMS)
    X_dict = dict_vect.fit_transform(df["clean_text"])

    X = hstack([X_tfidf, X_dict], format="csr")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = LinearSVC(class_weight="balanced", random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("\n=== Main Labels: D / S / H / N ===")
    print(classification_report(y_test, y_pred, digits=3))

    print("\n=== Subclassifier Presence Stats ===")
    print("\n-- Descriptive modes (percent of corpus) --")
    print((desc_flags.sum() / len(df) * 100).round(2).sort_values(ascending=False))
    print("\n-- Substantive modes (percent of corpus) --")
    print((subst_flags.sum() / len(df) * 100).round(2).sort_values(ascending=False))

    joblib.dump({"tfidf": tfidf, "dict_vect": dict_vect, "clf": clf,
                 "dict_terms": dict_terms, "labels": y},
                DATA_DIR / "model_artifacts.joblib")
    print(f"\n✓ Model + vectorizers saved to {DATA_DIR/'model_artifacts.joblib'}")
    

if __name__ == "__main__":
    main()
