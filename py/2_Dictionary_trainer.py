import re
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import StratifiedShuffleSplit
import joblib

DATA_DIR = Path("data")
SRC = DATA_DIR / "segments_clean.csv"

# dictionaries
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

SAMPLE_FRAC = 0.05
SAMPLE_RANDOM_STATE = 42


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

    # 1) Flags
    desc_flags = dict_match_flags(df["clean_text"], DESCRIPTIVE)
    subst_flags = dict_match_flags(df["clean_text"], SUBSTANTIVE)

    # 2) Labels
    D = (desc_flags.sum(axis=1) > 0).astype(int)
    S = (subst_flags.sum(axis=1) > 0).astype(int)
    H = ((D == 1) & (S == 1)).astype(int)
    labels = pd.DataFrame({"D": D, "S": S, "H": H})

    labels_out = pd.concat(
        [df[["date", "speaker", "source_file"]], labels,
         desc_flags.add_prefix("desc_"), subst_flags.add_prefix("subst_"),
         df[["clean_text"]]],
        axis=1
    )
    labels_out.to_csv(DATA_DIR / "labels.csv", index=False)

    # 3) Vectors
    dict_terms = flatten_dict(DESCRIPTIVE) + flatten_dict(SUBSTANTIVE)
    tfidf = TfidfVectorizer(**TFIDF_PARAMS)
    X_tfidf = tfidf.fit_transform(df["clean_text"])

    dict_vect = CountVectorizer(vocabulary=dict_terms, **DICT_VECT_PARAMS)
    X_dict = dict_vect.fit_transform(df["clean_text"])

    X_model = hstack([X_tfidf, X_dict], format="csr")

    save_npz(DATA_DIR / "X_tfidf.npz", csr_matrix(X_tfidf))
    save_npz(DATA_DIR / "X_dict.npz", csr_matrix(X_dict))
    save_npz(DATA_DIR / "X_model.npz", csr_matrix(X_model))
    joblib.dump({"tfidf": tfidf, "dict_vect": dict_vect, "dict_terms": dict_terms},
                DATA_DIR / "vectorizers.joblib")

    # 4) Stratified 5% sample
    bucket = np.where(H == 1, "H", np.where(D == 1, "D", np.where(S == 1, "S", "N")))
    strata = df["speaker"].astype(str).str.lower() + "||" + bucket
    n = len(df)
    test_size = max(1, int(round(SAMPLE_FRAC * n)))

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=SAMPLE_RANDOM_STATE)
    try:
        _, sample_idx = next(sss.split(np.arange(n).reshape(-1, 1), strata))
    except ValueError:
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=SAMPLE_RANDOM_STATE)
        _, sample_idx = next(sss2.split(np.arange(n).reshape(-1, 1), bucket))

    sample = df.iloc[sample_idx]
    sample.to_csv(DATA_DIR / "sample_5pct.csv", index=False)

    # 5) Save dictionaries
    with open(DATA_DIR / "dictionaries.json", "w", encoding="utf-8") as f:
        json.dump({"descriptive": DESCRIPTIVE, "substantive": SUBSTANTIVE},
                  f, indent=2, ensure_ascii=False)

    print(f"✓ Segments: {len(df)}")
    print(f"✓ Labels: D={D.sum()} S={S.sum()} H={H.sum()}")
    print(f"✓ Matrices: X_tfidf={X_tfidf.shape} X_dict={X_dict.shape} X_model={X_model.shape}")
    print(f"✓ Sample: {len(sample)} rows → data/sample_5pct.csv")


if __name__ == "__main__":
    main()
