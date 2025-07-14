import os
from dotenv import load_dotenv
from openai import OpenAI
from utils.loaders import load_pdfs, load_training_phrases
from utils.embedder import build_or_load_vectorstore
from utils.prompts import get_benji_prompt

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# Global system prompt
SYSTEM_PROMPT = (
    "You are Benji, a calm and strategic assistant trained to guide users through insurance claims like a chess game. "
    "Your goal is to help them get paid, not to get angry."
)

# Shared history starter
initial_history = [
    {"role": "system", "content": SYSTEM_PROMPT}
]

# Vectorstore init
pdf_docs = load_pdfs("data/")
training_docs = load_training_phrases("data/")
document_chunks = pdf_docs + training_docs
vectorstore = build_or_load_vectorstore(document_chunks)  # Should return FAISS index with retriever


def create_session_history():
    return initial_history.copy()


def retrieve_context(question: str, top_k: int = 4):
    """
    Use FAISS retriever to get relevant document chunks
    """
    docs = vectorstore.similarity_search(question, k=top_k)
    return "\n\n".join([doc.page_content for doc in docs])


def get_benji_response(question, chat_history):
    import traceback
    try:
        # Retrieve relevant knowledge
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
