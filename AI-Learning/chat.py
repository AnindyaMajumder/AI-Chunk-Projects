import os
from utils.prompts import get_benji_prompt
from app import create_session_history, get_benji_response

def main():
    print("Welcome to Benji! Type 'exit' to quit.")
    chat_history = create_session_history()
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        reply, chat_history = get_benji_response(user_input, chat_history)
        if reply.startswith("Error"):
            print(f"Benji: {reply} (Check your API key, vectorstore, or data files)")
        else:
            print(f"Benji: {reply}\n")

if __name__ == "__main__":
    main()
