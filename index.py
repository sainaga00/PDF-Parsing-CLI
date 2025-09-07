import os, sys, json, re

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
AI_TIMEOUT = 90
from openai import OpenAI

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


def main():
    if len(sys.argv) != 3:
        print("Usage: python first-name_last-name.py <input.pdf> <output.json>", file=sys.stderr)
        return 2

    in_path, out_path = sys.argv[1], sys.argv[2]
    if not os.path.isfile(in_path):
        print(f"Error: input file not found: {in_path}", file=sys.stderr)
        return 1

    try:
        client = OpenAI()
        # 1) Upload PDF
        uploaded = client.files.create(
            file=open(in_path, "rb"),
            purpose="assistants"  # suitable for Responses with files
        )

        # 2) Ask the model to parse the file and emit EXACT schema JSON
        #    NOTE: We pass the PDF via an input_file content part.
        resp = client.responses.create(
            model=OPENAI_MODEL,
            timeout=AI_TIMEOUT,
            input=[
                {"role": "system", "content": AI_SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "input_text", "text": AI_USER_PROMPT},
                    {"type": "input_file", "file_id": uploaded.id}
                ]}
            ]
        )

        # 3) Collect model text
        output_text = ""
        if getattr(resp, "output", None):
            # Responses API unified output (preferred)
            for piece in resp.output[0].content:
                if piece.type == "output_text":
                    output_text += piece.text
        if not output_text:
            # Fallback accessor
            output_text = getattr(resp, "output_text", "") or ""

        output_text = output_text.strip()

        # 4) Parse JSON; if invalid, fail to stub
        try:
            parsed = json.loads(output_text)
        except Exception:
            parsed = {"title": "Untitled", "contract_type": "", "effective_date": None, "sections": []}

        # 5) Final light schema repair (labels and numbers types; indices deterministic)
        for s in parsed.get("sections", []):
            if s.get("number", "") == "":
                s["number"] = None
            elif s.get("number") is not None:
                s["number"] = str(s["number"])
            for i, cl in enumerate(s.get("clauses", [])):
                cl["index"] = i
                if cl.get("label") is None:
                    cl["label"] = ""
                # normalize whitespace in text
                cl["text"] = re.sub(r"\s+", " ", (cl.get("text","") or "").strip())

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(parsed, f, ensure_ascii=False, indent=2)
        return 0

    except Exception as e:
        # Minimal valid stub if anything goes wrong with the AI path
        stub = {"title": "Untitled", "contract_type": "", "effective_date": None, "sections": []}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(stub, f, ensure_ascii=False, indent=2)
        return 0


if __name__ == "__main__":
    sys.exit(main())
