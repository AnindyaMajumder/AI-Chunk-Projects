from dotenv import load_dotenv
load_dotenv()
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
import os
import pymupdf
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
import json

# Load PDF documents from a directory
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

# Load training phrases from CSV files
def load_training_phrases(csv_path):
    documents = []
    if os.path.isdir(csv_path):
        for filename in os.listdir(csv_path):
            if filename.endswith(".csv"):
                # print(f"Loading training phrases")
                df = pd.read_csv(os.path.join(csv_path, filename), encoding="utf-8")
                for index, row in df.iterrows():
                    content = " ".join(str(value) for value in row.values if pd.notna(value))
                    # Create a Document object with metadata
                    documents.append(Document(
                        page_content=content,
                        metadata={"source": filename, "type": "training_phrase", "row": index}
                    ))
    return documents

# Embedder utility functions
def chunk_docs(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]  # Adjust separators as needed
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

def prompt(insurance_company: str, policy_number: str, policy_report_number: str, adjuster_name: str, adjuster_phone: str, claim_number: str, adjuster_email: str):
    # Load advice from Training Phrases.csv
    advice_docs = load_training_phrases("data/")
    advice_text = "\n".join([f"- {doc.page_content}" for doc in advice_docs])

    # System prompt instructions for Benji
    system_message = (
        "You are Benji, a calm and strategic assistant helping users through insurance claims.\n"
        "Your personality:\n"
        "- Calm, never emotional\n"
        "- Strategic like a chess coach\n"
        "- Empathetic, warm, and confident\n"
        "Include editable templates when useful. Avoid robotic responses.\n"
        "Give the template only when the user asks for it, otherwise provide a direct answer. Include the user's claim details in your template responses.\n"
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
        "- Insurance Company Name: {insurance_company}\n"
        "- Policy Number: {policy_number}\n"
        "- Policy Report Number: {policy_report_number}\n"
        "- Adjuster Name: {adjuster_name}\n"
        "- Adjuster Phone Number: {adjuster_phone}\n"
        "- Claim Number: {claim_number}\n"
        "- Adjuster Email: {adjuster_email}\n\n"
        "Answer as Benji:"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("user", user_template)
    ])
    return prompt

def model_init():
    load_dotenv()
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.3,
        max_tokens=2048,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    return llm

def chaining(insurance_company: str, policy_number: str, policy_report_number: str, adjuster_name: str, adjuster_phone: str, claim_number: str, adjuster_email: str, global_knowledge="data/", local_knowledge="upload/", local_folder_name="local_knowledge"):
    load_dotenv()
    global_store = "index"
    local_store = os.path.join(global_store, local_folder_name)
    global_docs = load_pdfs(global_knowledge)
    global_vectorstore = build_or_load_vectorstore(global_docs, os.path.join(global_store, "faiss_store"))
    local_docs = load_pdfs(local_knowledge)
    local_vectorstore = build_or_load_vectorstore(local_docs, os.path.join(local_store, "faiss_store"))
    llm = model_init()
    prompt_template = prompt(insurance_company, policy_number, policy_report_number, adjuster_name, adjuster_phone, claim_number, adjuster_email)
    def format_inputs(inputs):
        local_docs = local_vectorstore.as_retriever(search_kwargs={"k": 7}).invoke(inputs["question"])
        global_docs = global_vectorstore.as_retriever(search_kwargs={"k": 4}).invoke(inputs["question"])
        local_context = "\n\n".join([doc.page_content for doc in local_docs])
        global_context = "\n\n".join([doc.page_content for doc in global_docs])
        combined_context = (
            f"[Local knowledge: {local_folder_name}]\n" + local_context + "\n\n[Global knowledge]\n" + global_context
        )
        return {
            "context": combined_context,
            "chat_history": inputs.get("chat_history", ""),
            "question": inputs["question"],
            "insurance_company": inputs["insurance_company"],
            "policy_number": inputs["policy_number"],
            "policy_report_number": inputs["policy_report_number"],
            "adjuster_name": inputs["adjuster_name"],
            "adjuster_phone": inputs["adjuster_phone"],
            "claim_number": inputs["claim_number"],
            "adjuster_email": inputs["adjuster_email"]
        }
    chain = RunnableLambda(format_inputs) | prompt_template | llm | StrOutputParser()
    return chain

# Function to manage and return chat history as a list of dictionaries
# Accepts either a file path (str) or a list of dicts directly
def get_chat_history(history_source="chat_history.json"):
    if isinstance(history_source, list):
        return history_source
    elif isinstance(history_source, str):
        if os.path.exists(history_source):
            with open(history_source, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            return []
    else:
        return []

def estimate_token_count(text):
    # Rough estimate: 1 token ≈ 4 characters (for English)
    return len(text) // 4

def trim_chat_history(history_list, max_tokens=2048):
    trimmed = []
    total_tokens = 0
    # Start from the most recent messages
    for msg in reversed(history_list):
        msg_text = f"User: {msg['human']}\nBenji: {msg['ai']}"
        msg_tokens = estimate_token_count(msg_text)
        if total_tokens + msg_tokens > max_tokens:
            break
        trimmed.insert(0, msg)  # Insert at the beginning to maintain order
        total_tokens += msg_tokens
    return trimmed

def get_history_text(history_list, max_tokens=2048):
    trimmed_history = trim_chat_history(history_list, max_tokens)
    return "\n".join([
        f"User: {msg['human']}\nBenji: {msg['ai']}" for msg in trimmed_history
    ])

def run_benji_chat(insurance_company, policy_number, policy_report_number, adjuster_name, adjuster_phone, claim_number, adjuster_email, user_question, chat_history_list=None, local_folder_name="custom_local_knowledge", local_pdf_path_or_folder="upload/"):
    if chat_history_list is None:
        chat_history_list = []
    chain = chaining(insurance_company, policy_number, policy_report_number, adjuster_name, adjuster_phone, claim_number, adjuster_email, local_knowledge=local_pdf_path_or_folder, local_folder_name=local_folder_name)
    inputs = {
        "insurance_company": insurance_company,
        "policy_number": policy_number,
        "policy_report_number": policy_report_number,
        "adjuster_name": adjuster_name,
        "adjuster_phone": adjuster_phone,
        "claim_number": claim_number,
        "adjuster_email": adjuster_email,
        "question": user_question,
        "chat_history": chat_history_list
    }
    response = chain.invoke({
        **inputs,
        "chat_history": get_history_text(chat_history_list, max_tokens=2048)
    })
    chat_history_list.append({"human": user_question, "ai": response})
    return response, chat_history_list

# Example usage for local testing only
if __name__ == "__main__":
    def run_benji_chat(insurance_company, policy_number, policy_report_number, adjuster_name, adjuster_phone, claim_number, adjuster_email, user_question, chat_history_list=None, local_folder_name="custom_local_knowledge", local_pdf_path_or_folder="upload/"):
        if chat_history_list is None:
            chat_history_list = []
        def get_history_text(history_list):
            return "\n".join([
                f"User: {msg['human']}\nBenji: {msg['ai']}" for msg in history_list
            ])
        chain = chaining(insurance_company, policy_number, policy_report_number, adjuster_name, adjuster_phone, claim_number, adjuster_email, local_knowledge=local_pdf_path_or_folder, local_folder_name=local_folder_name)
        inputs = {
            "insurance_company": insurance_company,
            "policy_number": policy_number,
            "policy_report_number": policy_report_number,
            "adjuster_name": adjuster_name,
            "adjuster_phone": adjuster_phone,
            "claim_number": claim_number,
            "adjuster_email": adjuster_email,
            "question": user_question,
            "chat_history": chat_history_list
        }
        response = chain.invoke({
            **inputs,
            "chat_history": get_history_text(chat_history_list)
        })
        chat_history_list.append({"human": user_question, "ai": response})
        return response, chat_history_list
    
    # Example interaction
    insurance_company = "Acme Insurance"
    policy_number = "POL123456"
    policy_report_number = "REP7890"
    adjuster_name = "Jane Smith"
    adjuster_phone = "555-123-4567"
    claim_number = "CLM987654"
    adjuster_email = "jane.smith@acme.com"
    local_folder_name = "local_knowledge"
    local_pdf_path = "upload/policy.pdf"  # Change this to your specific PDF path
    chat_history_list = []
    print("Welcome to Benji Insurance Chatbot!")
    print("Type 'exit' to end the chat.")
    while True:
        user_question = input("You: ")
        if user_question.strip().lower() == "exit":
            print("Goodbye!")
            break
        response, chat_history_list = run_benji_chat(
            insurance_company, policy_number, policy_report_number, adjuster_name, adjuster_phone, claim_number, adjuster_email, user_question, chat_history_list, local_folder_name, local_pdf_path
        )
        print(f"Benji: {response}\n")