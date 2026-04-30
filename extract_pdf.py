import pypdf

with open("Proyecto Análisis de Algoritmos - 2026-1.pdf", "rb") as f:
    reader = pypdf.PdfReader(f)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"

with open("pdf_text.txt", "w", encoding="utf-8") as f:
    f.write(text)
