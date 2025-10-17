
import re
import pathlib
import csv

SRC_DIR = pathlib.Path("data/sbclean")
OUT_CSV = pathlib.Path("data/segments.csv")
SEGMENT_RE = re.compile(r"^([a-z]+):\s*$")  # e.g., "greene:"

rows = []

for txt_file in sorted(SRC_DIR.glob("*.txt")):
    with open(txt_file, encoding="utf-8") as f:
        lines = [ln.rstrip() for ln in f]

    date = None
    current_speaker = None
    current_block = []

    for ln in lines:
        if ln.startswith("# "):
            date = ln[2:].strip()
            continue

        match = SEGMENT_RE.match(ln)
        if match:
            # flush current block
            if current_speaker and current_block:
                rows.append({
                    "date": date,
                    "speaker": current_speaker,
                    "text": "\n".join(current_block).strip(),
                    "source_file": txt_file.name
                })
            current_speaker = match.group(1)
            current_block = []
        else:
            current_block.append(ln)

    # flush last block
    if current_speaker and current_block:
        rows.append({
            "date": date,
            "speaker": current_speaker,
            "text": "\n".join(current_block).strip(),
            "source_file": txt_file.name
        })

# Write to CSV
with open(OUT_CSV, "w", encoding="utf-8", newline="") as out_f:
    writer = csv.DictWriter(out_f, fieldnames=["date", "speaker", "text", "source_file"])
    writer.writeheader()
    writer.writerows(rows)

print(f"✓ Wrote {len(rows)} segments to {OUT_CSV}")
