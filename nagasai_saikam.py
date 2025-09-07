
import sys, os, re, json, shutil
from datetime import datetime
from multiprocessing import Process, Queue
from typing import Any, Dict

# ==================== CONFIG ====================
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
AI_API_TIMEOUT = 60      # per API request timeout
AI_HARD_WALL_SECS = 40   # total budget before switching to local parser

# ==================== AI availability ====================
_HAVE_OPENAI = False
try:
    from openai import OpenAI
    _HAVE_OPENAI = True
except Exception:
    _HAVE_OPENAI = False
_HAVE_OPENAI = _HAVE_OPENAI or ("openai" in sys.modules)

# ==================== Prompts ====================
AI_SYSTEM_PROMPT = """You are a contract parsing engine. Return ONLY valid UTF-8 JSON matching this schema exactly:

{
  "title": "Contract Title",
  "contract_type": "Agreement Type",
  "effective_date": "YYYY-MM-DD or null",
  "sections": [
    {
      "title": "Section Title",
      "number": "Section Number or null",
      "clauses": [
        { "text": "Clause text", "label": "Label or \"\"", "index": 0 }
      ]
    }
  ]
}

Rules:
- Do not include extra keys.
- Strings vs null: sections[n].number -> string or null; clauses[n].label -> string or "".
- effective_date must be ISO YYYY-MM-DD or null.
- Clause index is 0-based within each section.
- Preserve section/paragraph reading order; normalize internal whitespace to single spaces; strip leading/trailing whitespace.
- For inline clauses like "(a) ... (b) ...", split into separate clauses with labels.
- If the document starts with a title and only an effective date line before Section 1, DO NOT create a 'Preamble' section — capture the date in effective_date only.
- Output must be valid JSON (UTF-8), no comments, no markdown.
"""

AI_USER_PROMPT = """Parse the ATTACHED PDF contract and return EXACTLY the required JSON schema. 
Use the file as your sole source of truth. If any field is unknown, follow the no-ambiguity rules."""

# ==================== Utilities ====================
def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def has_signal(text: str) -> bool:
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

# ==================== Text extraction (local) ====================
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
    return extract_text_pymupdf(pdf_path) or extract_text_pdfminer(pdf_path) or extract_text_pypdf(pdf_path) or extract_text_ocr(pdf_path)

# ==================== Heuristic local parser ====================
SECTION_PATTERNS = [
    re.compile(r"^\s*(Section\s+)?(?P<num>(\d+(\.\d+)*|[IVXLCM]+))[\.\):\-]?\s+(?P<title>.+)$", re.I),
    re.compile(r"^\s*(?P<title>[A-Z0-9][A-Z0-9\s,'&\-/]{3,})\s*$"),
]
LABEL_START = re.compile(
    r"^\s*(?P<label>(?:\d+(?:\.\d+){0,3}|\([a-z]\)|\([ivx]+\)|[A-Z]\.|\([A-Z]\)))\s+",
    re.IGNORECASE
)
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

def preamble_is_only_effective_date(body_text: str) -> bool:
    body = normalize_ws(body_text or "")
    if not body:
        return False
    if len(body) > 120:
        return False
    month_names = r"(January|February|March|April|May|June|July|August|September|October|November|December)"
    patterns = [
        rf"^effective\s*date\s*:\s*{month_names}\s+\d{{1,2}}(?:st|nd|rd|th)?\s*,\s*\d{{4}}$",
        rf"^effective\s+as\s+of\s+{month_names}\s+\d{{1,2}}(?:st|nd|rd|th)?\s*,\s*\d{{4}}$",
        rf"^{month_names}\s+\d{{1,2}}(?:st|nd|rd|th)?\s*,\s*\d{{4}}$",
        rf"^effective\s+on\s+{month_names}\s+\d{{1,2}}(?:st|nd|rd|th)?\s*,\s*\d{{4}}$",
    ]
    return any(re.fullmatch(p, body, flags=re.I) for p in patterns)

def split_into_clauses(body_text: str):
    paragraphs = [p for p in re.split(r"\n\s*\n", (body_text or "").strip()) if p.strip()]
    clauses = []
    for p in paragraphs:
        p_flat = normalize_ws(re.sub(r"\n+", " ", p))
        start_label = None
        start_match = LABEL_START.match(p_flat)
        content_start = 0
        if start_match:
            start_label = normalize_ws(start_match.group("label"))
            content_start = start_match.end()
        rest = p_flat[content_start:]
        matches = list(LABEL_INLINE.finditer(rest))
        if not start_label and not matches:
            clauses.append({"text": p_flat, "label": ""})
            continue
        first_pos = matches[0].start() if matches else len(rest)
        lead_text = normalize_ws(rest[:first_pos]) if content_start else normalize_ws(
            p_flat[:content_start] if start_label else p_flat[:first_pos])
        if lead_text:
            clauses.append({"text": lead_text, "label": ""})
        if start_label:
            end = matches[0].start() if matches else len(rest)
            first_text = normalize_ws(rest[:end])
            if first_text:
                clauses.append({"text": first_text, "label": start_label})
        for i, m in enumerate(matches):
            label = normalize_ws(m.group("label"))
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(rest)
            text = normalize_ws(rest[start:end])
            if text:
                clauses.append({"text": text, "label": label})
    return clauses

def inject_linebreaks_if_flat(text: str) -> str:
    if (text or "").count("\n") >= 3:
        return text
    text = re.sub(
        r"\s+(?=(Section\s+\d+|[IVXLCM]+|(?:\d+(?:\.\d+){0,3}))\s*[\.\):\-]?\s+[A-Z])",
        "\n",
        text or ""
    )
    text = re.sub(r"\s+(?=([A-Z]{3,}(?:\s+[A-Z]{3,}){0,2})\s*(?:\n|$))", "\n", text or "")
    return text

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

def extract_effective_date(text: str):
    t = (" " + (text or "") + " ").replace("\u2013","-").replace("\u2014","-")
    m = re.search(r"(effective\s+(date|as of|on)\s*:?\s*)?"
                  r"(?P<month>January|February|March|April|May|June|July|August|September|October|November|December)"
                  r"\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?\s*,\s*(?P<year>\d{4})", t, re.I)
    if m:
        try:
            return datetime(int(m.group("year")), 
                            ["january","february","march","april","may","june","july","august","september","october","november","december"].index(m.group("month").lower())+1,
                            int(m.group("day"))).strftime("%Y-%m-%d")
        except Exception:
            pass
    m = re.search(r"(effective\s+(date|as of|on)\s*:?\s*)?"
                  r"(?:the\s+)?(?P<day>\d{1,2})(?:st|nd|rd|th)?\s+day\s+of\s+"
                  r"(?P<month>January|February|March|April|May|June|July|August|September|October|November|December)"
                  r"\s*,?\s*(?P<year>\d{4})", t, re.I)
    if m:
        try:
            return datetime(int(m.group("year")),
                            ["january","february","march","april","may","june","july","august","september","october","november","december"].index(m.group("month").lower())+1,
                            int(m.group("day"))).strftime("%Y-%m-%d")
        except Exception:
            pass
    m = re.search(r"effective[^.]{0,80}?(\d{4})-(\d{2})-(\d{2})", t, re.I)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))).strftime("%Y-%m-%d")
        except Exception:
            pass
    m = re.search(r"(?P<month>January|February|March|April|May|June|July|August|September|October|November|December)"
                  r"\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?\s*,\s*(?P<year>\d{4})", t, re.I)
    if m:
        try:
            return datetime(int(m.group("year")),
                            ["january","february","march","april","may","june","july","august","september","october","november","december"].index(m.group("month").lower())+1,
                            int(m.group("day"))).strftime("%Y-%m-%d")
        except Exception:
            pass
    return None

def parse_sections(full_text: str, doc_title: str = None):
    lines = [ln for ln in (full_text or "").splitlines()]
    sections, cur = [], None
    def flush(sec):
        if not sec:
            return
        body = "\n".join(sec["body"]).strip()
        if not sections and sec["title"] == "Preamble" and preamble_is_only_effective_date(body):
            return
        clauses = split_into_clauses(body)
        for i, cl in enumerate(clauses):
            cl["index"] = i
        sections.append({"title": sec["title"], "number": sec["number"], "clauses": clauses})
    for ln in lines:
        n = normalize_ws(ln)
        if not n:
            continue
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

def build_output_heuristic(full_text: str):
    full_text = (full_text or "").replace("\r", "\n").replace("\x00", " ")
    full_text = inject_linebreaks_if_flat(full_text)
    title = guess_title(full_text)
    contract_type = guess_contract_type(title, full_text)
    effective_date = extract_effective_date(full_text)
    sections = parse_sections(full_text, doc_title=title)
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

# ==================== AI worker (separate process) ====================
def _ai_worker(pdf_path: str, q: Queue):
    try:
        if not _HAVE_OPENAI or not os.getenv("OPENAI_API_KEY"):
            q.put({"ok": False, "err": "AI unavailable"})
            return
        client = OpenAI()
        uploaded = client.files.create(
            file=open(pdf_path, "rb"),
            purpose="assistants"
        )
        resp = client.responses.create(
            model=OPENAI_MODEL,
            timeout=AI_API_TIMEOUT,
            input=[
                {"role": "system", "content": AI_SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "input_text", "text": AI_USER_PROMPT},
                    {"type": "input_file", "file_id": uploaded.id}
                ]}
            ]
        )
        output_text = ""
        if getattr(resp, "output", None):
            for piece in resp.output[0].content:
                if piece.type == "output_text":
                    output_text += piece.text
        if not output_text:
            output_text = getattr(resp, "output_text", "") or ""
        output_text = output_text.strip()
        try:
            parsed = json.loads(output_text)
        except Exception:
            parsed = {"title": "Untitled", "contract_type": "", "effective_date": None, "sections": []}
        for s in parsed.get("sections", []):
            if s.get("number", "") == "":
                s["number"] = None
            elif s.get("number") is not None:
                s["number"] = str(s["number"])
            for i, cl in enumerate(s.get("clauses", [])):
                cl["index"] = i
                if cl.get("label") is None:
                    cl["label"] = ""
                cl["text"] = normalize_ws((cl.get("text","") or ""))
        q.put({"ok": True, "data": parsed})
    except Exception as e:
        q.put({"ok": False, "err": f"{type(e).__name__}: {e}"})

# ==================== CLI ====================
def main():
    if len(sys.argv) != 3:
        print("Usage: python first-name_last-name.py <input.pdf> <output.json>", file=sys.stderr)
        return 2

    in_path, out_path = sys.argv[1], sys.argv[2]
    if not os.path.isfile(in_path):
        print(f"Error: input file not found: {in_path}", file=sys.stderr)
        return 1

    result: Dict[str, Any] = None
    used_ai = False

    # Try AI in a separate process with a hard timeout.
    if _HAVE_OPENAI and os.getenv("OPENAI_API_KEY"):
        q = Queue()
        proc = Process(target=_ai_worker, args=(in_path, q), daemon=True)
        proc.start()
        proc.join(AI_HARD_WALL_SECS)
        if proc.is_alive():
            proc.terminate()
            proc.join(2)
            print(f"[info] AI parsing exceeded {AI_HARD_WALL_SECS}s — switching to local library parser.", file=sys.stderr)
        else:
            msg = None
            try:
                msg = q.get_nowait()
            except Exception:
                pass
            if msg and msg.get("ok"):
                result = msg.get("data")
                used_ai = True
            else:
                err = (msg or {}).get("err", "no result from AI process")
                print(f"[info] AI parsing failed ({err}) — switching to local library parser.", file=sys.stderr)

    # Local fallback
    if not used_ai or result is None:
        text = extract_text(in_path)
        if not has_signal(text):
            text = ""
        result = build_output_heuristic(text)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return 0

if __name__ == "__main__":
    sys.exit(main())
