
import re
import pathlib
import bs4
import html5lib 

HEARING_ROOT = pathlib.Path("data/hearings")
DEST_DIR = pathlib.Path("data/sbclean")
DEST_DIR.mkdir(parents=True, exist_ok=True)

FOLDERS = {
    "biggs":      "BiggsAZ05",
    "boebert":    "BoebertCO03",
    "gosar":      "GosarAZ09",
    "greene":     "GreeneGA14",
    "luna":       "LunaFL13",
    "mace":       "MaceSC01",
}

TARGETS = {
    "biggs":   re.compile(r"^(Representative\s+)?(Mr\.|Chairman)?\s*Biggs$",   re.I),
    "boebert": re.compile(r"^(Representative\s+)?(Ms\.?|Mrs\.?)?\s*Boebert$",  re.I),
    "gosar":   re.compile(r"^(Representative\s+)?(Mr\.?)?\s*Gosar$",           re.I),
    "greene":  re.compile(r"^(Representative\s+)?(Ms\.?|Mrs\.?)?\s*Greene$",   re.I),
    "luna":    re.compile(r"^(Representative\s+)?(Ms\.?|Mrs\.?)?\s*Luna$",     re.I),
    "mace":    re.compile(r"^(Representative\s+)?(Ms\.?|Mrs\.?)?\s*Mace$",     re.I),
}

SPEECH_HEAD = re.compile(r"^([A-Z][A-Za-z .'-]{0,40})\.\s*(.*)$")
DATE_PAT = re.compile(r"(\d{4}-\d{2}-\d{2})")

for speaker, folder in FOLDERS.items():
    src_dir = HEARING_ROOT / folder
    if not src_dir.exists():
        print(f"⚠️ Folder not found: {src_dir}")
        continue

    for html_file in src_dir.glob("*_Hearing.html"):
        date_match = DATE_PAT.search(html_file.name)
        if not date_match:
            print(f"⚠️ No ISO date in filename: {html_file.name}; skipped")
            continue
        date = date_match.group(1)

        with open(html_file, encoding="utf-8") as fh:
            soup = bs4.BeautifulSoup(fh, "html5lib")
            lines = soup.get_text("\n").splitlines()

        current_speaker = None
        current_block = []
        collected_blocks = []

        for ln in lines:
            ln = ln.strip()
            if not ln:
                continue

            head = SPEECH_HEAD.match(ln)
            if head:
                if current_speaker:
                    for target, pattern in TARGETS.items():
                        if pattern.match(current_speaker):
                            block_text = "\n".join(current_block).strip()
                            if block_text:
                                collected_blocks.append((target, block_text))
                            break
                current_speaker = head.group(1).strip()
                current_block = [head.group(2).strip()]
            else:
                current_block.append(ln)

        if current_speaker:
            for target, pattern in TARGETS.items():
                if pattern.match(current_speaker):
                    block_text = "\n".join(current_block).strip()
                    if block_text:
                        collected_blocks.append((target, block_text))
                    break

        if not collected_blocks:
            print(f"—  {html_file.name}: no target speakers found")
            continue

        out_path = DEST_DIR / f"{date}_{speaker}.txt"
        with open(out_path, "w", encoding="utf-8") as out:
            out.write(f"# {date}\n\n")
            for surname, block in collected_blocks:
                out.write(f"{surname}:\n{block}\n\n")

        print(f"✓  {out_path}  ({len(collected_blocks)} blocks)")

print("Done.")
