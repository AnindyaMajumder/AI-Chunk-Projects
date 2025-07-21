import os
from utils.prompts import get_benji_prompt
from app import create_session_history, get_benji_response

def main():
    print("Welcome to Benji! Type 'exit' to quit.")
    chat_history = []
    # Dummy claim info
    claim_no = 123456
    name = "John Doe"
    phone = "123-456-7890"
    email = "john.doe@example.com"
    local_folder_name = "local_knowledge"
    local_pdf_path_or_folder = "upload/"
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        reply, chat_history = get_benji_response(
            claim_no, name, phone, email, user_input, chat_history, local_folder_name, local_pdf_path_or_folder
        )
        if reply.startswith("Error"):
            print(f"Benji: {reply} (Check your API key, vectorstore, or data files)")
        else:
            print(f"Benji: {reply}\n")

if __name__ == "__main__":
    main()
