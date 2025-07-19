import os
from dotenv import load_dotenv
from openai import OpenAI

from utils.loaders import load_pdfs
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


# --- CSV advice logic (import, format, but do NOT add to vectorstore) ---
import pandas as pd
from collections import defaultdict

def load_training_phrases_and_advices(csv_path):
    advices_by_category = defaultdict(list)
    if os.path.isdir(csv_path):
        for filename in os.listdir(csv_path):
            if filename.endswith(".csv"):
                df = pd.read_csv(os.path.join(csv_path, filename), encoding="utf-8")
                for _, row in df.iterrows():
                    category = row.get('Category', row.get('category', 'General'))
                    advice = row.get('Advice', row.get('advice', ''))
                    if pd.notna(category) and pd.notna(advice):
                        advices_by_category[str(category).strip()].append(str(advice).strip())
    return advices_by_category

def format_advices_for_prompt(advices_by_category):
    lines = ["Reference Advice (imported from CSV):"]
    for category, advices in advices_by_category.items():
        lines.append(f"{category}:")
        for advice in advices:
            lines.append(f"  - {advice}")
    return "\n".join(lines)

# Load PDFs (only PDFs go into vectorstore)
pdf_docs = load_pdfs("data/")

# Load and format CSV advice for prompt only
advices_by_category = load_training_phrases_and_advices("data/")
csv_advice_reference = format_advices_for_prompt(advices_by_category)

# Only PDF docs are stored in the vectorstore
document_chunks = pdf_docs
vectorstore = build_or_load_vectorstore(document_chunks)  # Should return FAISS index with retriever


def create_session_history():
    return initial_history.copy()



def retrieve_context(question: str, top_k: int = 4):
    """
    Use FAISS retriever to get relevant document chunks (PDFs only).
    """
    docs = vectorstore.similarity_search(question, k=4)
    formatted_chunks = [doc.page_content for doc in docs]
    return "\n\n".join(formatted_chunks)


def get_benji_response(question, chat_history):
    import traceback
    try:
        # Retrieve relevant knowledge (PDFs only)
        docs = vectorstore.similarity_search(question, k=4)
        context_chunks = [doc.page_content for doc in docs]

        def estimate_tokens(text):
            # Rough estimate: 1 token ≈ 4 characters
            return len(text) // 4

        max_tokens = 8000  # Safe threshold for GPT-4 (adjust as needed)
        history = chat_history.copy()
        # Try to keep as much history and context as possible
        while True:
            history_str = "\n".join([f"{m['role']}: {m['content']}" for m in history])
            context = "\n\n".join(context_chunks)
            benji_prompt = get_benji_prompt(csv_advice_reference).format(context=context, chat_history=history_str, question=question)
            total_text = benji_prompt + question + history_str + context
            total_tokens = estimate_tokens(total_text)
            if total_tokens < max_tokens:
                break
            # First, trim history (after system prompt)
            if len(history) > 1:
                history.pop(1)
            # If history is minimal, trim context chunks
            elif len(context_chunks) > 1:
                context_chunks.pop(0)
            else:
                break
        messages = history.copy()
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
