import os
import pymupdf
from langchain.schema import Document

def load_pdfs(pdf_path_or_dir):
    documents = []
    if os.path.isdir(pdf_path_or_dir):
        for filename in os.listdir(pdf_path_or_dir):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(pdf_path_or_dir, filename)
                doc = pymupdf.open(pdf_path)
                pages = ""
                for page in doc:
                    text = page.get_text()
                    pages += str(text)
                documents.append(Document(
                    page_content=pages,
                    metadata={"source": filename, "type": "pdf"}
                ))
                doc.close()
    elif os.path.isfile(pdf_path_or_dir) and pdf_path_or_dir.endswith(".pdf"):
        doc = pymupdf.open(pdf_path_or_dir)
        pages = ""
        for page in doc:
            text = page.get_text()
            pages += str(text)
        documents.append(Document(
            page_content=pages,
            metadata={"source": os.path.basename(pdf_path_or_dir), "type": "pdf"}
        ))
        doc.close()
    return documents
