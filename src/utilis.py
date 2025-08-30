import pdfplumber
import docx2txt
import re

def extract_text_pdf(path):
    texts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            texts.append(page.extract_text() or "")
    return "\n".join(texts)

def extract_text_docx(path):
    return docx2txt.process(path)

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
