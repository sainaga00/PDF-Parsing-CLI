Parses contracts from PDF into a strict JSON schema using OpenAI (if API key set) or local parsing (PyMuPDF, pdfminer, PyPDF2, OCR).

Install
pip install openai PyMuPDF pdfminer.six PyPDF2 pdf2image pytesseract


System deps for OCR:

macOS: brew install poppler tesseract

Ubuntu: sudo apt-get install poppler-utils tesseract-ocr

Windows: install Poppler + Tesseract, set POPPLER_PATH / TESSERACT_CMD.

Usage
python first-name_last-name.py <input.pdf> <output.json>

Env Vars

OPENAI_API_KEY – enable AI parsing

OPENAI_MODEL – default gpt-4o-mini

Output Example
{
  "title": "MASTER SERVICES AGREEMENT",
  "contract_type": "Master Services Agreement",
  "effective_date": "2023-05-01",
  "sections": [...]
}
