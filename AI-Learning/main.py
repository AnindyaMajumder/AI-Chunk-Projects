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
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
import json

# Load PDF documents from a directory
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
    
    return documents

# Load training phrases from CSV files
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

def prompt(claim_no: int, name: str, phone: str, email: str):
    # System prompt instructions for Benji
    system_message = (
        "You are Benji, a calm and strategic assistant helping users through insurance claims.\n"
        "Your personality:\n"
        "- Calm, never emotional\n"
        "- Strategic like a chess coach\n"
        "- Empathetic, warm, and confident\n"
        "Always reinforce: 'Stay calm. This is a game of chess. The goal is to get paid — not to get angry.'\n"
        "Include editable templates when useful. Avoid robotic responses."
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

def chaining(claim_no: int, name: str, phone: str, email: str, global_knowledge = "data/", local_knowledge = "upload/", global_store="index/", local_store="locals/"):
    load_dotenv()
    
    # Load global knowledge base
    global_docs = load_pdfs(global_knowledge)
    global_vectorstore = build_or_load_vectorstore(global_docs, os.path.join(global_store, "faiss_store"))
    
    # Load local knowledge base
    local_docs = load_pdfs(local_knowledge)
    local_vectorstore = build_or_load_vectorstore(local_docs, os.path.join(local_store, "faiss_store"))
    
    # Initialize the model
    llm = model_init()
    
    # Create the prompt
    prompt_template = prompt(claim_no, name, phone, email)
    
    def format_inputs(inputs):
        # Retrieve relevant documents
        global_docs = global_vectorstore.as_retriever().invoke(inputs["question"])
        local_docs = local_vectorstore.as_retriever().invoke(inputs["question"])
        # Combine contexts
        combined_context = "\n\n".join([doc.page_content for doc in global_docs + local_docs])
        # Return dict for prompt
        return {
            "context": combined_context,
            "chat_history": inputs.get("chat_history", ""),
            "question": inputs["question"],
            "claim_no": inputs["claim_no"],
            "name": inputs["name"],
            "phone": inputs["phone"],
            "email": inputs["email"]
        }
    # Chain using LangChain's RunnableLambda and RunnablePassthrough
    
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

def main():
    claim_no = 123456
    name = "John Doe"
    phone = "123-456-7890"
    email = "johndoe@gmail.com"

    chain = chaining(claim_no, name, phone, email)
    print("Welcome to Benji Insurance Chatbot!")
    print("Type 'exit' to end the chat.")
    history_file = "chat_history.json"
    chat_history_list = get_chat_history(history_file)

    def get_history_text(history_list):
        # Convert history list to text for prompt
        return "\n".join([
            f"User: {msg['human']}\nBenji: {msg['ai']}" for msg in history_list
        ])

    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == "exit":
            print("Goodbye!")
            break
        # Build the input dict
        inputs = {
            "claim_no": claim_no,
            "name": name,
            "phone": phone,
            "email": email,
            "question": user_input,
            # For backend, send chat_history as a list of dicts
            "chat_history": chat_history_list
        }
        response = chain.invoke({
            **inputs,
            # For prompt formatting, still use text
            "chat_history": get_history_text(chat_history_list)
        })
        print(f"Benji: {response}\n")
        # Update chat history list
        chat_history_list.append({"human": user_input, "ai": response})
        
        # If you want to persist history locally, uncomment below:
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(chat_history_list, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()