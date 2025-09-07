# PDF-Parsing-CLI
A Python tool to parse PDF contracts into a strict JSON schema.
Supports two modes:

AI Mode (if OPENAI_API_KEY is set) → Uses OpenAI for parsing.

Local Mode (fallback) → Extracts and structures contract text using PyMuPDF, pdfminer, PyPDF2, and OCR.

Output Schema
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

Installation
Python & Libraries

Python 3.9+

Install required packages:

pip install openai PyMuPDF pdfminer.six PyPDF2 pdf2image pytesseract

System Dependencies

For OCR-based fallback:

Poppler

macOS: brew install poppler

Ubuntu/Debian: sudo apt-get install poppler-utils

Windows: Install Poppler and set POPPLER_PATH

Tesseract OCR

macOS: brew install tesseract

Ubuntu/Debian: sudo apt-get install tesseract-ocr

Windows: Install from Tesseract project
 and set TESSERACT_CMD

Environment Variables
Variable	Purpose	Default
OPENAI_API_KEY	Enables AI parsing	unset
OPENAI_MODEL	OpenAI model	gpt-4o-mini
POPPLER_PATH	Path to Poppler binaries (Windows only)	auto-detected
TESSERACT_CMD	Path to Tesseract binary (Windows only)	auto-detected
Usage
python first-name_last-name.py <input.pdf> <output.json>


Example:

python nagasai_saikam.py input.pdf output.json


If AI is enabled, runs AI parsing with timeout → else falls back to local parser.

Always outputs valid JSON to <output.json>.

Example Output
{
  "title": "MASTER SERVICES AGREEMENT",
  "contract_type": "Master Services Agreement",
  "effective_date": "2023-05-01",
  "sections": [
    {
      "title": "Definitions",
      "number": "1",
      "clauses": [
        { "text": "Affiliate means …", "label": "", "index": 0 }
      ]
    }
  ]
}
