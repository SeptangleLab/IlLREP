df = pd.read_csv(SRC)
df["clean_text"] = df["clean_text"].fillna("")

if not CASE_SENSITIVE:
    df["clean_text"] = df["clean_text"].str.lower()
    TARGET_TERMS = [t.lower() for t in TARGET_TERMS]

patterns = {t: re.compile(rf"\b{re.escape(t)}\b") for t in TARGET_TERMS}

for term, pat in patterns.items():
    mask = df["clean_text"].str.contains(pat)
    subset = df[mask]

    term_count = mask.sum()
    total_occurrences = subset["clean_text"].str.count(pat).sum()

    print("="*60)
    print(f"TERM: '{term}'")
    print(f"Segments containing term: {term_count}")
    print(f"Total occurrences in corpus: {total_occurrences}")

    tokens = [row.split() for row in subset["clean_text"]]
    co_tokens = list(chain.from_iterable(tokens))
    co_counts = Counter(co_tokens)
    co_counts.pop(term, None)

    print(f"\nTop {TOP_COOCUR} co-occurring words with '{term}':")
    for w, c in co_counts.most_common(TOP_COOCUR):
        print(f"{w:20} {c}")
