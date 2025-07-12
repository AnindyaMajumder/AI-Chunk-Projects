import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
# from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from utils.loaders import load_pdfs, load_training_phrases
from utils.embedder import build_or_load_vectorstore, chunk_docs
from utils.prompts import get_benji_prompt

load_dotenv()

def initialize_benji_chain():
    """
    Initialize and return the Benji AI chain for Django backend integration.
    
    Returns:
        ConversationalRetrievalChain: Configured chain ready for use
    """
    try:
        # Load and combine docs
        pdf_docs = load_pdfs("data/")
        training_docs = load_training_phrases("data/")
        all_docs = pdf_docs + training_docs

        # Build or load FAISS vectorstore
        vectorstore = build_or_load_vectorstore(all_docs)

        # Set up memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Create model and chain
        llm = ChatOpenAI(temperature=0.1)

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_type="similarity", k=4),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": get_benji_prompt()}
        )
        
        return qa_chain
        
    except Exception as e:
        print(f"Error initializing Benji chain: {str(e)}")
        raise

# Initialize the chain for immediate use
qa_chain = initialize_benji_chain()

def get_benji_response(question, session_memory=None):
    """
    Get a response from Benji AI for Django integration.
    
    Args:
        question (str): User's question
        session_memory (ConversationBufferMemory, optional): Session-specific memory
    
    Returns:
        str: Benji's response
    """
    try:
        if session_memory:
            # Use session-specific memory for this request
            temp_chain = ConversationalRetrievalChain.from_llm(
                llm=ChatOpenAI(temperature=0.1),
                retriever=qa_chain.retriever,
                memory=session_memory,
                combine_docs_chain_kwargs={"prompt": get_benji_prompt()}
            )
            response = temp_chain({"question": question})
        else:
            # Use global chain
            response = qa_chain({"question": question})
            
        return response.get("answer", "I'm sorry, I couldn't process that request.")
        
    except Exception as e:
        return f"Error: {str(e)}"

def create_session_memory():
    """
    Create a new conversation memory for a Django session.
    
    Returns:
        ConversationBufferMemory: New memory instance for session
    """
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
