import pymupdf
def load_pdf(pdf_path):
    doc = pymupdf.open(pdf_path)
    pages = ""
    
    for page in doc:
        text = page.get_text().encode("utf-8")
        # print(text)
        pages += str(text)
    return pages