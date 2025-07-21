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

# Only PDF docs are stored in the global vectorstore
document_chunks = pdf_docs
global_vectorstore = build_or_load_vectorstore(document_chunks)  # Should return FAISS index with retriever


def create_session_history():
    return initial_history.copy()

def retrieve_context(question: str, top_k: int = 4):
    """
    Use FAISS retriever to get relevant document chunks (PDFs only).
    """
    docs = vectorstore.similarity_search(question, k=4)
    formatted_chunks = [doc.page_content for doc in docs]
    return "\n\n".join(formatted_chunks)



# --- Benji Chat Logic (matching main.py) ---
def estimate_token_count(text):
    # Rough estimate: 1 token ≈ 4 characters (for English)
    return len(text) // 4

def trim_chat_history(history_list, max_tokens=2048):
    trimmed = []
    total_tokens = 0
    # Start from the most recent messages
    for msg in reversed(history_list):
        msg_text = f"User: {msg['human']}\nBenji: {msg['ai']}" if 'human' in msg and 'ai' in msg else f"{msg['role']}: {msg['content']}"
        msg_tokens = estimate_token_count(msg_text)
        if total_tokens + msg_tokens > max_tokens:
            break
        trimmed.insert(0, msg)  # Insert at the beginning to maintain order
        total_tokens += msg_tokens
    return trimmed

def get_history_text(history_list, max_tokens=2048):
    trimmed_history = trim_chat_history(history_list, max_tokens)
    return "\n".join([
        f"User: {msg['human']}\nBenji: {msg['ai']}" if 'human' in msg and 'ai' in msg else f"{msg['role']}: {msg['content']}" for msg in trimmed_history
    ])

def get_benji_response(claim_no, name, phone, email, user_question, chat_history_list=None, local_folder_name="local_knowledge", local_pdf_path_or_folder="upload/"):
    import traceback
    try:
        if chat_history_list is None:
            chat_history_list = []
        # --- Local knowledge support ---
        # Build local vectorstore if local_pdf_path_or_folder exists and has PDFs
        local_store_path = os.path.join("index", local_folder_name, "faiss_store")
        local_docs = load_pdfs(local_pdf_path_or_folder)
        if local_docs:
            local_vectorstore = build_or_load_vectorstore(local_docs, local_store_path)
            local_context_docs = local_vectorstore.similarity_search(user_question, k=7)
            local_context = "\n\n".join([doc.page_content for doc in local_context_docs])
        else:
            local_context = ""
        # --- Global knowledge ---
        global_context_docs = global_vectorstore.similarity_search(user_question, k=4)
        global_context = "\n\n".join([doc.page_content for doc in global_context_docs])
        # --- Combine context ---
        combined_context = f"[Local knowledge: {local_folder_name}]\n" + local_context + "\n\n[Global knowledge]\n" + global_context

        # Build advice text from CSV
        advice_text = csv_advice_reference

        # System prompt instructions for Benji (from main.py)
        system_message = (
            "You are Benji, a calm and strategic assistant helping users through insurance claims.\n"
            "Your personality:\n"
            "- Calm, never emotional\n"
            "- Strategic like a chess coach\n"
            "- Empathetic, warm, and confident\n"
            "Include editable templates when useful. Avoid robotic responses.\n"
            "Give the template only when the user asks for it, otherwise provide a direct answer.\n"
            "You strictly only answer questions related to insurance claims or claim processes."
            "If the user greets you (e.g., 'hi', 'hello', 'good morning', 'bye') respond politely as a normal chatbot would, but remind them you can only assist with insurance-related issues. For any non-insurance topic, say: 'Sorry, I can only help with insurance claim related questions.\n"
            "Keep responses concise and focused on the user's claim. If user asked for his informations, provide it precisely. If any information is missing, say that information is missing\n"
            "If the user asks for summary of the conversation, provide a summary of the chat history.\n"
            "\nBest practices and advice for insurance claims:\n" + advice_text + "\n"
        )
        # User prompt template
        user_template = (
            "Context:\n{context}\n\n"
            "Conversation history:\n{chat_history}\n\n"
            "User question:\n{question}\n\n"
            "CLAIM DETAILS:\n"
            "- Claim Number: {claim_no}\n"
            "- Claimant Name: {name}\n"
            "- Contact Phone: {phone}\n"
            "- Contact Email: {email}\n\n"
            "Answer as Benji:"
        )

        # Prepare chat history text
        history_text = get_history_text(chat_history_list, max_tokens=2048)

        # Compose the prompt
        prompt = system_message + user_template.format(
            context=combined_context,
            chat_history=history_text,
            question=user_question,
            claim_no=claim_no,
            name=name,
            phone=phone,
            email=email
        )

        # Call OpenAI
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_template.format(
                    context=combined_context,
                    chat_history=history_text,
                    question=user_question,
                    claim_no=claim_no,
                    name=name,
                    phone=phone,
                    email=email
                )}
            ],
            temperature=0.3,
            max_tokens=2048
        )
        reply = response.choices[0].message.content
        chat_history_list.append({"human": user_question, "ai": reply})
        return reply, chat_history_list
    except Exception as e:
        tb = traceback.format_exc()
        error_type = type(e).__name__
        return f"Error ({error_type}): {str(e)}\nTraceback:\n{tb}", chat_history_list
