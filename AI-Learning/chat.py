#!/usr/bin/env python3
"""
Live Terminal Chat Interface for Benji AI Assistant
Run this script to chat with Benji in real-time through the terminal.

Usage:
    python chat.py

Requirements:
    - .env file with OPENAI_API_KEY
    - Documents in data/ directory
    - All dependencies installed (pip install -r requirements.txt)
"""

import os
import sys
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from utils.loaders import load_pdfs, load_training_phrases
from utils.embedder import build_or_load_vectorstore
from utils.prompts import get_benji_prompt

def initialize_chat_system():
    """Initialize the chat system with all necessary components."""
    print("🔄 Initializing Benji AI Assistant...")
    
    # Load environment variables
    load_dotenv()
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment variables.")
        print("Please set your OpenAI API key in a .env file or environment variable.")
        print("Example: Create a .env file with: OPENAI_API_KEY=your_key_here")
        sys.exit(1)
    
    try:
        # Load and combine documents
        print("📚 Loading documents...")
        pdf_docs = load_pdfs("data/")
        training_docs = load_training_phrases("data/")
        all_docs = pdf_docs + training_docs
        print(f"✅ Loaded {len(all_docs)} documents")
        
        # Validate documents were loaded
        if len(all_docs) == 0:
            print("⚠️  Warning: No documents found in data/ directory")
            print("Please ensure PDF files and Training Phrases.csv exist in the data/ folder")
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                sys.exit(1)
        
        # Build or load FAISS vectorstore
        print("🔍 Building/loading vector store...")
        vectorstore = build_or_load_vectorstore(all_docs)
        print("✅ Vector store ready")
        
        # Set up memory for conversation
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create the language model
        llm = ChatOpenAI(temperature=0.1)
        
        # Create the conversational chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_type="similarity", k=4),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": get_benji_prompt()}
        )
        
        print("✅ Benji AI Assistant is ready!")
        return qa_chain
        
    except Exception as e:
        print(f"❌ Error initializing chat system: {str(e)}")
        sys.exit(1)

def print_welcome_message():
    """Print a welcome message for the user."""
    print("\n" + "="*60)
    print("🤖 Welcome to Benji AI Assistant - Live Chat")
    print("="*60)
    print("Your calm and strategic insurance claims assistant is ready!")
    print("\nCommands:")
    print("  - Type your question and press Enter to chat")
    print("  - Type 'quit', 'exit', or 'bye' to end the session")
    print("  - Type 'clear' to clear the conversation history")
    print("  - Type 'help' to see this message again")
    print("\n" + "-"*60 + "\n")

def print_help():
    """Print help information."""
    print("\n📋 Help - Available Commands:")
    print("  • quit/exit/bye - End the chat session")
    print("  • clear - Clear conversation history")
    print("  • help - Show this help message")
    print("  • Just type your insurance-related question to get help from Benji!")
    print()

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def format_response(response):
    """Format the AI response for better readability."""
    # Add some visual separation
    print("\n🤖 Benji:")
    print("-" * 50)
    print(response)
    print("-" * 50)

def main():
    """Main chat loop."""
    try:
        # Initialize the chat system
        qa_chain = initialize_chat_system()
        
        # Print welcome message
        print_welcome_message()
        
        # Main chat loop
        while True:
            try:
                # Get user input
                user_input = input("💬 You: ").strip()
                
                # Handle empty input
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n👋 Thanks for chatting with Benji! Stay calm and strategic!")
                    break
                elif user_input.lower() == 'clear':
                    # Clear conversation memory
                    qa_chain.memory.clear()
                    clear_screen()
                    print("🧹 Conversation history cleared!")
                    print_welcome_message()
                    continue
                elif user_input.lower() == 'help':
                    print_help()
                    continue
                
                # Process the question with Benji
                print("\n🤔 Benji is thinking...")
                try:
                    response = qa_chain({"question": user_input})
                    answer = response.get("answer", "I'm sorry, I couldn't process that request.")
                    format_response(answer)
                except Exception as e:
                    print(f"\n❌ Error processing your question: {str(e)}")
                
                print()  # Add spacing for next input
                
            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted. Thanks for using Benji AI Assistant!")
                break
            except EOFError:
                print("\n\n👋 Thanks for chatting with Benji!")
                break
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
