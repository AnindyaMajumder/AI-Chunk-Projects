import pandas as pd
import os
import pymupdf
from langchain.schema import Document

def load_pdfs(pdf_dir):
    documents = []
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, filename)
            doc = pymupdf.open(pdf_path)
            pages = ""
            
            for page in doc:
                text = page.get_text()
                pages += str(text)
            
            # Create a Document object with metadata
            documents.append(Document(
                page_content=pages,
                metadata={"source": filename, "type": "pdf"}
            ))
            doc.close()
    
    print(len(documents))
    return documents

def load_training_phrases(csv_path):
    documents = []
    if os.path.isdir(csv_path):
        for filename in os.listdir(csv_path):
            if filename.endswith(".csv"):
                print(f"Loading training phrases")
                df = pd.read_csv(os.path.join(csv_path, filename), encoding="utf-8")
                for index, row in df.iterrows():
                    content = " ".join(str(value) for value in row.values if pd.notna(value))
                    # Create a Document object with metadata
                    documents.append(Document(
                        page_content=content,
                        metadata={"source": filename, "type": "training_phrase", "row": index}
                    ))
    print(len(documents))
    return documents
    