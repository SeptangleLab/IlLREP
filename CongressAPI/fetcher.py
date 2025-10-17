import os
import re
import time
import textwrap
import pathlib
import requests
import xml.etree.ElementTree as ET
from urllib.parse import urlencode

API_KEY = ":)"
BASE_URL = "https://api.congress.gov/v3"
CONGRESS  = 119
REQUEST_DELAY = 0.25

TARGET_REPS = {
    "BiggsAZ05" : ("B001302", "Biggs"),
    "BoebertCO03": ("B000825", "Boebert"),
    "GosarAZ09"  : ("G000565", "Gosar"),
    "GreeneGA14" : ("G000596", "Greene"),
    "LunaFL13"   : ("L000596", "Luna"),
    "MaceSC01"   : ("M000194", "Mace"),
}

ROOT_DIR   = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT_DIR / "data"
HEAR_DIR   = DATA_DIR / "hearings"
BILL_DIR   = DATA_DIR / "bills"
for p in (HEAR_DIR, BILL_DIR): p.mkdir(parents=True, exist_ok=True)

HEADERS = {"X-API-Key": API_KEY, "User-Agent": "research-script/1.0 (+https://example.com)"}
PARAMS  = {"format": "xml"}

def fetch(url, **extra):
    """GET wrapper with API-key, retries & polite delay."""
    params = PARAMS.copy(); params.update(extra)
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=40)
            r.raise_for_status()
            return r.content
        except requests.HTTPError as e:
            if r.status_code == 404: # just in case
                raise
            if attempt == 2:
                raise
            time.sleep(1.5)
    return b""

def safe_filename(s):
    return re.sub(r"[^A-Za-z0-9_.\-]+", "_", s.strip())[:250]

def save_bytes(path: pathlib.Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)

def harvest_hearings():
    print("🔎 Scanning House hearings…")
    page_size = 250
    offset = 0
    found_cnt = 0
    while True:
        list_url = f"{BASE_URL}/hearing/{CONGRESS}/house"
        xml_blob = fetch(list_url, offset=offset, limit=page_size)
        root = ET.fromstring(xml_blob)
        items = root.findall("./hearings/item")
        if not items:
            break

        for it in items:
            detail_url = it.findtext("./url").strip()
            jacket     = it.findtext("./jacketNumber") or "UNKNOWN"
            try:
                detail_xml = fetch(detail_url)
            except requests.HTTPError:
                continue
            droot = ET.fromstring(detail_xml)
            f_url = None
            for f in droot.findall(".//formats/item"):
                if (f.findtext("./type") or "").strip() == "Formatted Text":
                    f_url = f.findtext("./url").strip()
                    break
            if not f_url:
                continue
            html_blob = None
            for attempt in range(3):
                try:
                    html_blob = requests.get(f_url, headers=HEADERS, timeout=60 * (attempt + 1)).content
                    break  # success
                except requests.exceptions.Timeout:
                    print(f"Timeout")
                    time.sleep(2)
            if html_blob is None:
                print(f"Timeout 2")
                continue

            html_txt = html_blob.decode("utf-8", errors="ignore")
            matched_reps = [key for key, (_, last) in TARGET_REPS.items()
                            if re.search(rf"\b{re.escape(last)}\b", html_txt, re.I)]
            if not matched_reps:
                continue  # skip unrelated hearings

            title = droot.findtext("./title") or "Hearing"
            date  = droot.findtext(".//dates/item/date") or "n.d."
            file_base = f"{date}_{safe_filename(title[:120])}.html"

            for rep_key in matched_reps:
                out_path = HEAR_DIR / rep_key / file_base
                save_bytes(out_path, html_blob)
            found_cnt += 1
            print(f"   ✔ saved transcript {file_base}   (mentions: {', '.join(matched_reps)})")

        offset += page_size
        time.sleep(REQUEST_DELAY)

    print(f"Done")

# ────────────────────────────────  BILLS  ────────────────────────────────
def bills_for_rep(bioguide_id: str):
    """Return a list of <item> elements for bills *sponsored* by the given member."""
    items = []
    offset = 0
    page  = 250
    while True:
        q = {"congress": CONGRESS, "type": "hr", "offset": offset, "pageSize": page,
             "sponsorship": "sponsored"}   # key field!
        list_url = f"{BASE_URL}/member/{bioguide_id}/bills"
        try:
            xml_blob = fetch(list_url, **q)
        except requests.HTTPError:
            break
        root = ET.fromstring(xml_blob)
        batch = root.findall("./bills/bill")
        if not batch:
            break
        items.extend(batch)
        offset += page
        time.sleep(REQUEST_DELAY)
    return items

def download_bill(bill_elem, out_dir):
    bill_number = bill_elem.findtext("./number")
    bill_url    = bill_elem.findtext("./url").strip()
    if not bill_number or not bill_url:
        return False
    try:
        full_xml = fetch(bill_url)
    except requests.HTTPError:
        return False
    save_bytes(out_dir / f"{bill_elem.findtext('./type')}_{bill_number}.xml", full_xml)
    broot  = ET.fromstring(full_xml)
    best_text_url = None
    best_ext      = None
    priority = ["Formatted Text", "Formatted XML", "PDF"]
    for v in broot.findall(".//textVersions/item"):
        for fmt in v.findall("./formats/item"):
            ftype = (fmt.findtext("./type") or "").strip()
            furl  = (fmt.findtext("./url")  or "").strip()
            if ftype in priority and furl:
                best_text_url = furl
                best_ext      = ".html" if "Text" in ftype else ".xml" if "XML" in ftype else ".pdf"
                break
        if best_text_url:
            break
    if best_text_url:
        try:
            blob = fetch(best_text_url, format=None)
            save_bytes(out_dir / f"{bill_elem.findtext('./type')}_{bill_number}_FullText{best_ext}", blob)
        except requests.HTTPError:
            pass
    return True

def harvest_bills():
    print("\n📦 Downloading sponsored bills + full text…")
    total = 0
    for rep_key, (bio, _) in TARGET_REPS.items():
        print(f" • {rep_key} …", end="", flush=True)
        rep_dir = BILL_DIR / rep_key
        bill_items = bills_for_rep(bio)
        saved = 0
        for b in bill_items:
            if download_bill(b, rep_dir):
                saved += 1
            time.sleep(REQUEST_DELAY)
        total += saved
        print(f"  {saved} saved")
    print(f"bills saved: {total}")
if __name__ == "__main__":
    for what in ("hearings", "bills"):
        print("\n" + "="*60 + f"\n{what.upper()}\n" + "="*60)
        try:
            if what == "hearings":
                harvest_hearings()
            else:
                harvest_bills()
        except KeyboardInterrupt:
            print("\n⏹ Interrupted by user")
            break
