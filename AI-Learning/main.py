import os
import pandas as pd
import pymupdf
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate

# --- Utility: Prompts ---
def get_benji_prompt():
    return ChatPromptTemplate.from_template("""
You are Benji, a calm and strategic assistant helping users through insurance claims.
Your personality:
- Calm, never emotional
- Strategic like a chess coach
- Empathetic, warm, and confident

Always reinforce: “Stay calm. This is a game of chess. The goal is to get paid — not to get angry.”

Context:
{context}

Conversation history:
{chat_history}

User question:
{question}

Answer as Benji:
""")

# --- Utility: Loaders ---
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
            documents.append(Document(
                page_content=pages,
                metadata={"source": filename, "type": "pdf"}
            ))
            doc.close()
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
                    documents.append(Document(
                        page_content=content,
                        metadata={"source": filename, "type": "training_phrase", "row": index}
                    ))
    return documents

# --- Utility: Embedder ---
def chunk_docs(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(documents)

def build_or_load_vectorstore(documents, index_path="index/faiss_store"):
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-3-small")
    # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if os.path.exists(index_path):
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    chunks = chunk_docs(documents)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_path)
    return vectorstore

# --- Main App Logic ---
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

SYSTEM_PROMPT = (
    "You are Benji, a calm and strategic assistant trained to guide users through insurance claims like a chess game. "
    "Your goal is to help them get paid, not to get angry."
)

initial_history = [
    {"role": "system", "content": SYSTEM_PROMPT}
]

pdf_docs = load_pdfs("data/")
training_docs = load_training_phrases("data/")
document_chunks = pdf_docs + training_docs
vectorstore = build_or_load_vectorstore(document_chunks)

def create_session_history():
    return initial_history.copy()

def retrieve_context(question: str, top_k: int = 4):
    docs = vectorstore.similarity_search(question, k=top_k)
    return "\n\n".join([doc.page_content for doc in docs])

def get_benji_response(question, chat_history):
    import traceback
    try:
        context = retrieve_context(question)
        history_str = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history])
        benji_prompt = get_benji_prompt().format(context=context, chat_history=history_str, question=question)
        messages = chat_history.copy()
        messages.append({"role": "system", "content": benji_prompt})
        messages.append({"role": "user", "content": question})
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.1
        )
        reply = response.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})
        return reply, messages
    except Exception as e:
        tb = traceback.format_exc()
        error_type = type(e).__name__
        return f"Error ({error_type}): {str(e)}\nTraceback:\n{tb}", chat_history

if __name__ == "__main__":
    print("Welcome to Benji! Type 'exit' to quit.")
    chat_history = create_session_history()
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        reply, chat_history = get_benji_response(user_input, chat_history)
        if reply.startswith("Error:"):
            print(f"Benji: {reply} (Check your API key, vectorstore, or data files)")
        else:
            print(f"Benji: {reply}\n")
