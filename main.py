import sys, os, re, json, shutil
from datetime import datetime

# ==================== Utilities ====================

def normalize_ws(s: str) -> str:
    """Collapse internal whitespace to single spaces, strip edges."""
    return re.sub(r"\s+", " ", (s or "")).strip()

def has_signal(text: str) -> bool:
    """Basic 'is there text here' check to decide OCR fallback."""
    return len(re.findall(r"[A-Za-z0-9]", text or "")) > 50

def which(cmd):
    return shutil.which(cmd)

# ---------- Poppler / Tesseract helpers ----------

def get_poppler_path():
    env = os.getenv("POPPLER_PATH")
    if env and os.path.isdir(env):
        return env
    for c in [
        "/opt/homebrew/opt/poppler/bin", "/usr/local/opt/poppler/bin",
        "/usr/local/bin", "/usr/bin",
        r"C:\Program Files\poppler\bin",
        r"C:\Program Files\poppler-24.07.0\Library\bin",
        r"C:\Program Files\poppler-24.02.0\Library\bin",
        r"C:\Program Files\poppler-23.11.0\Library\bin",
    ]:
        if os.path.isdir(c):
            return c
    return None

def set_tesseract_cmd_if_needed():
    try:
        import pytesseract as pt
    except Exception:
        return
    cmd_env = os.getenv("TESSERACT_CMD")
    if cmd_env and os.path.exists(cmd_env):
        pt.pytesseract.tesseract_cmd = cmd_env
        return
    if os.name == "nt":
        for c in [r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                  r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"]:
            if os.path.exists(c):
                pt.pytesseract.tesseract_cmd = c
                return

# ==================== Extraction ====================

def extract_text_pymupdf(pdf_path: str) -> str:
    try:
        import fitz
    except Exception:
        return ""
    try:
        out = []
        with fitz.open(pdf_path) as doc:
            for page in doc:
                d = page.get_text("dict")
                for b in d.get("blocks", []):
                    for l in b.get("lines", []):
                        text = " ".join(s.get("text", "") for s in l.get("spans", [])).strip()
                        if text:
                            out.append(text)
        return "\n".join(out)
    except Exception:
        return ""

def extract_text_pdfminer(pdf_path: str) -> str:
    try:
        from pdfminer.high_level import extract_text
        return extract_text(pdf_path) or ""
    except Exception:
        return ""

def extract_text_pypdf(pdf_path: str) -> str:
    try:
        import PyPDF2
        out = []
        with open(pdf_path, "rb") as f:
            r = PyPDF2.PdfReader(f)
            for p in r.pages:
                out.append(p.extract_text() or "")
        return "\n".join(out)
    except Exception:
        return ""

def extract_text_ocr(pdf_path: str, dpi: int = 300) -> str:
    poppler_path = get_poppler_path()
    set_tesseract_cmd_if_needed()
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except Exception:
        return ""
    try:
        kwargs = {"dpi": dpi}
        if poppler_path:
            kwargs["poppler_path"] = poppler_path
        pages = convert_from_path(pdf_path, **kwargs)
        texts = []
        for img in pages:
            gray = img.convert("L")
            # Optional binarization for faint scans:
            # gray = gray.point(lambda x: 0 if x < 200 else 255, '1')
            txt = pytesseract.image_to_string(gray, lang="eng", config="--oem 3 --psm 6") or ""
            texts.append(txt)
        return "\n".join(texts)
    except Exception:
        return ""

def extract_text(pdf_path: str) -> str:
    for fn in (extract_text_pymupdf, extract_text_pdfminer, extract_text_pypdf, extract_text_ocr):
        t = fn(pdf_path)
        if has_signal(t):
            return t
    # best-effort return
    return extract_text_pymupdf(pdf_path) or extract_text_pdfminer(pdf_path) or extract_text_pypdf(pdf_path) or extract_text_ocr(pdf_path)

# ==================== Title / Type ====================

def guess_title(full_text: str) -> str:
    lines = [normalize_ws(x) for x in (full_text or "").splitlines() if normalize_ws(x)]
    sec_idx = next((i for i, ln in enumerate(lines) if is_section_header(ln)), None)
    search_space = lines[:sec_idx] if sec_idx is not None else lines[:30]
    for ln in search_space:
        if re.search(r"\b(Agreement|Contract|Terms|Master|License)\b", ln, re.I):
            return ln
    for ln in search_space:
        if len(ln) <= 120 and ln.isupper() and len(ln.split()) >= 2:
            return ln
    return lines[0] if lines else "Untitled"

def guess_contract_type(title: str, full_text: str) -> str:
    candidates = [
        "Master Services Agreement","Services Agreement","Software License Agreement","Subscription Agreement",
        "Non-Disclosure Agreement","Confidentiality Agreement","Employment Agreement","Consulting Agreement",
        "Data Processing Agreement","Purchase Agreement","Lease Agreement","Reseller Agreement","Partner Agreement",
        "SaaS Agreement","License Agreement","Statement of Work","Order Form","Addendum","Amendment","Agreement","Contract","Terms"
    ]
    hay = (title + "\n" + (full_text[:2000] or "")).lower()
    for c in candidates:
        if c.lower() in hay:
            return c
    return ""

# ==================== Effective date ====================

MONTHS = {m.lower(): i for i, m in enumerate(
    ["January","February","March","April","May","June","July","August","September","October","November","December"], start=1
)}
def _date_month_day_year(month, day, year):
    try:
        return datetime(int(year), MONTHS[month.lower()], int(day)).strftime("%Y-%m-%d")
    except Exception:
        return None
def _date_iso(y, m, d):
    try:
        return datetime(int(y), int(m), int(d)).strftime("%Y-%m-%d")
    except Exception:
        return None

def extract_effective_date(text: str):
    t = (" " + (text or "") + " ").replace("\u2013","-").replace("\u2014","-")
    m = re.search(r"(effective\s+(date|as of|on)\s*:?\s*)?"
                  r"(?P<month>January|February|March|April|May|June|July|August|September|October|November|December)"
                  r"\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?\s*,\s*(?P<year>\d{4})", t, re.I)
    if m:
        d = _date_month_day_year(m.group("month"), m.group("day"), m.group("year"))
        if d: return d
    m = re.search(r"(effective\s+(date|as of|on)\s*:?\s*)?"
                  r"(?:the\s+)?(?P<day>\d{1,2})(?:st|nd|rd|th)?\s+day\s+of\s+"
                  r"(?P<month>January|February|March|April|May|June|July|August|September|October|November|December)"
                  r"\s*,?\s*(?P<year>\d{4})", t, re.I)
    if m:
        d = _date_month_day_year(m.group("month"), m.group("day"), m.group("year"))
        if d: return d
    m = re.search(r"effective[^.]{0,80}?(\d{4})-(\d{2})-(\d{2})", t, re.I)
    if m:
        d = _date_iso(m.group(1), m.group(2), m.group(3))
        if d: return d
    m = re.search(r"(?P<month>January|February|March|April|May|June|July|August|September|October|November|December)"
                  r"\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?\s*,\s*(?P<year>\d{4})", t, re.I)
    if m:
        d = _date_month_day_year(m.group("month"), m.group("day"), m.group("year"))
        if d: return d
    return None

# ==================== Section / Clause detection ====================

SECTION_PATTERNS = [
    # "1", "1.1", "I", "II", or "Section 1 Title"
    re.compile(r"^\s*(Section\s+)?(?P<num>(\d+(\.\d+)*|[IVXLCM]+))[\.\):\-]?\s+(?P<title>.+)$", re.I),
    # ALL CAPS header (e.g., DEFINITIONS)
    re.compile(r"^\s*(?P<title>[A-Z0-9][A-Z0-9\s,'&\-/]{3,})\s*$"),
]

# Labels:
# - At paragraph START: allow numeric/alpha/roman (e.g., "1.2", "(a)", "(i)", "A.", "(A)")
LABEL_START = re.compile(
    r"^\s*(?P<label>(?:\d+(?:\.\d+){0,3}|\([a-z]\)|\([ivx]+\)|[A-Z]\.|\([A-Z]\)))\s+",
    re.IGNORECASE
)
# - INLINE: only bracketed letters/romans or A./(A) — NOT bare numerals, to avoid splitting "clause 1"
LABEL_INLINE = re.compile(
    r"(?:(?<=\s)|(?<=^))(?P<label>(?:\([a-z]\)|\([ivx]+\)|[A-Z]\.|\([A-Z]\)))\s+",
    re.IGNORECASE
)

def is_section_header(line: str) -> bool:
    line = line.strip()
    for pat in SECTION_PATTERNS:
        m = pat.match(line)
        if m:
            if pat is SECTION_PATTERNS[1]:
                if len(line.split()) > 10 or not line.isupper():
                    continue
            return True
    return False

def extract_section_header(line: str):
    line = line.strip()
    for pat in SECTION_PATTERNS:
        m = pat.match(line)
        if not m:
            continue
        if "num" in m.groupdict() and m.group("num"):
            number = normalize_ws(m.group("num"))
            title = normalize_ws(m.group("title") or "")
            return (number if number else None), (title if title else "Section")
        else:
            title = normalize_ws(m.group("title"))
            return None, title
    return None, normalize_ws(line)

# ---- Preamble filter: drop if only an Effective Date line ----

def preamble_is_only_effective_date(body_text: str) -> bool:
    """
    Return True if the preamble body looks like ONLY an effective date line
    (e.g., 'Effective Date: March 15, 2024'), with nothing else substantive.
    """
    body = normalize_ws(body_text or "")
    if not body:
        return False
    # Short guard: a single short line is typical for title pages
    if len(body) > 120:
        return False
    # Allow patterns: 'Effective Date: Month Day, Year' OR 'effective as of Month Day, Year'
    month_names = r"(January|February|March|April|May|June|July|August|September|October|November|December)"
    patterns = [
        rf"^effective\s*date\s*:\s*{month_names}\s+\d{{1,2}}(?:st|nd|rd|th)?\s*,\s*\d{{4}}$",
        rf"^effective\s+as\s+of\s+{month_names}\s+\d{{1,2}}(?:st|nd|rd|th)?\s*,\s*\d{{4}}$",
        rf"^{month_names}\s+\d{{1,2}}(?:st|nd|rd|th)?\s*,\s*\d{{4}}$",
        rf"^effective\s+on\s+{month_names}\s+\d{{1,2}}(?:st|nd|rd|th)?\s*,\s*\d{{4}}$",
    ]
    low = body.lower()
    for pat in patterns:
        if re.fullmatch(pat, low, flags=re.I):
            return True
    return False

def split_into_clauses(body_text: str):
    """
    Split into paragraph-like chunks using BLANK LINES only.
    Within each paragraph:
      • If a label appears at the very start (LABEL_START), treat it as the first clause's label.
      • Then split further on INLINE labels (LABEL_INLINE) such as (a), (b), (c)...
      • Bare numerals are NOT treated as inline labels to avoid splitting "clause 1" or "Section 1".
    """
    paragraphs = [p for p in re.split(r"\n\s*\n", (body_text or "").strip()) if p.strip()]
    clauses = []

    for p in paragraphs:
        p_flat = normalize_ws(re.sub(r"\n+", " ", p))

        # Detect label at paragraph start
        start_label = None
        start_match = LABEL_START.match(p_flat)
        content_start = 0
        if start_match:
            start_label = normalize_ws(start_match.group("label"))
            content_start = start_match.end()

        # INLINE labels (letters/romans/A.)
        rest = p_flat[content_start:]
        matches = list(LABEL_INLINE.finditer(rest))

        if not start_label and not matches:
            clauses.append({"text": p_flat, "label": ""})
            continue

        # Text before first label (unlabeled preface in the paragraph)
        first_pos = matches[0].start() if matches else len(rest)
        lead_text = normalize_ws(rest[:first_pos]) if content_start else normalize_ws(
            p_flat[:content_start] if start_label else p_flat[:first_pos])
        if lead_text:
            clauses.append({"text": lead_text, "label": ""})

        # First labeled chunk (if start label exists)
        cursor = 0
        if start_label:
            end = matches[0].start() if matches else len(rest)
            first_text = normalize_ws(rest[:end])
            if first_text:
                clauses.append({"text": first_text, "label": start_label})
            cursor = end

        # Subsequent inline labeled chunks
        for i, m in enumerate(matches):
            label = normalize_ws(m.group("label"))
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(rest)
            text = normalize_ws(rest[start:end])
            if text:
                clauses.append({"text": text, "label": label})

    return clauses

def parse_sections(full_text: str, doc_title: str = None):
    lines = [ln for ln in (full_text or "").splitlines()]
    sections, cur = [], None

    def flush(sec):
        if not sec:
            return
        body = "\n".join(sec["body"]).strip()

        # NEW: Drop a leading Preamble that only contains an Effective Date line
        if not sections and sec["title"] == "Preamble":
            if preamble_is_only_effective_date(body):
                return

        clauses = split_into_clauses(body)
        for i, cl in enumerate(clauses):
            cl["index"] = i
        sections.append({"title": sec["title"], "number": sec["number"], "clauses": clauses})

    for ln in lines:
        n = normalize_ws(ln)
        if not n:
            continue
        # Ignore the very top title as a section header
        if doc_title and n == doc_title and not sections and cur is None:
            continue
        if is_section_header(n):
            flush(cur)
            num, htitle = extract_section_header(n)
            cur = {"title": htitle, "number": num, "body": []}
        else:
            if cur is None:
                cur = {"title": "Preamble", "number": None, "body": []}
            cur["body"].append(n)

    flush(cur)

    if not sections:
        txt = normalize_ws(full_text)
        clauses = split_into_clauses(txt) if txt else []
        for i, cl in enumerate(clauses):
            cl["index"] = i
        sections = [{"title": "Body", "number": None, "clauses": clauses}]
    return sections

# ==================== Flat-line fixer ====================

def inject_linebreaks_if_flat(text: str) -> str:
    if (text or "").count("\n") >= 3:
        return text
    # Newline before typical headers: "Section 1", Roman numerals, 1., 1.1, etc.
    text = re.sub(
        r"\s+(?=(Section\s+\d+|[IVXLCM]+|(?:\d+(?:\.\d+){0,3}))\s*[\.\):\-]?\s+[A-Z])",
        "\n",
        text or ""
    )
    # Newline before ALL CAPS titles (DEFINITIONS, CONFIDENTIALITY, etc.)
    text = re.sub(r"\s+(?=([A-Z]{3,}(?:\s+[A-Z]{3,}){0,2})\s*(?:\n|$))", "\n", text or "")
    return text

# ==================== Assembly ====================

def build_output(full_text: str):
    # Preserve line breaks
    full_text = (full_text or "").replace("\r", "\n").replace("\x00", " ")
    full_text = inject_linebreaks_if_flat(full_text)

    title = guess_title(full_text)
    contract_type = guess_contract_type(title, full_text)
    effective_date = extract_effective_date(full_text)
    sections = parse_sections(full_text, doc_title=title)

    # Final normalization & determinism
    for s in sections:
        for i, cl in enumerate(s["clauses"]):
            cl["index"] = i
            cl["text"] = normalize_ws(cl["text"])
            if cl.get("label") is None:
                cl["label"] = ""
    return {
        "title": title,
        "contract_type": contract_type,
        "effective_date": effective_date if effective_date else None,
        "sections": sections
    }

# ==================== CLI ====================

def main():
    if len(sys.argv) != 3:
        print("Usage: python first-name_last-name.py <input.pdf> <output.json>", file=sys.stderr)
        return 2

    in_path, out_path = sys.argv[1], sys.argv[2]
    if not os.path.isfile(in_path):
        print(f"Error: input file not found: {in_path}", file=sys.stderr)
        return 1

    text = extract_text(in_path)
    if not has_signal(text):
        text = ""  # graceful degradation

    result = build_output(text)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return 0

if __name__ == "__main__":
    sys.exit(main())
