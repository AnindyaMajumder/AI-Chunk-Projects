import openai
import json
from dotenv import load_dotenv
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import faiss
import pickle

load_dotenv()

# === Configuration ===
openai.api_key = os.getenv("OPENAI_API_KEY")

# === Product Data ===
def load_products():
    try:
        with open("products.json", "r", encoding="utf-8") as f:
            products = json.load(f)
        return products
    except FileNotFoundError:
        print("Error: products.json file not found.")
        return []

# === Embedding ===
def embed_product_descriptions(products):
    EMBEDDING_PATH = "index/product_embeddings.faiss"
    PICKLE_PATH = "index/product_embeddings.pkl"
    descriptions = [product["description"] for product in products]
    # Check if embedding exists
    if os.path.exists(EMBEDDING_PATH) and os.path.exists(PICKLE_PATH):
        index = faiss.read_index(EMBEDDING_PATH)
        with open(PICKLE_PATH, "rb") as f:
            embeddings = pickle.load(f)
        return embeddings, index
    # If not, create embeddings
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=descriptions
    )
    embeddings = np.array([item.embedding for item in response.data], dtype="float32")
    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    # Save index and embeddings
    faiss.write_index(index, EMBEDDING_PATH)
    with open(PICKLE_PATH, "wb") as f:
        pickle.dump(embeddings, f)
        
    return embeddings, index

# === Semantic Search for products ===
def semantic_search(query, products, index=None, embeddings=None):
    # Always generate query embedding
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    )
    query_embedding = np.array(response.data[0].embedding, dtype="float32")
    # If index and embeddings are available, use FAISS
    if index is not None and embeddings is not None:
        D, I = index.search(np.array([query_embedding]).astype("float32"), k=5)
        return [products[i] for i in I[0]]
    # Otherwise, use cosine similarity
    scores = []
    for product in products:
        score = cosine_similarity(
            np.array(query_embedding).reshape(1, -1),
            np.array(product["embedding"]).reshape(1, -1)
        )[0][0]
        scores.append((product, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [product for product, score in scores[:5]]

Messages = []
Messages.append({"role": "system", 
                 "content": """You are a helpful gift recommendation assistant. Always respond in a chill tone like best friend and also humorous.\n
                 You will recommend gifts based on user inputs and generate a personal story and memory-making activity related to the gift.\n
                 You will ask 3-4 questions to gather user preferences and then recommend a gift from the provided list. Ask at max one question at a time.\n
                 The gift should be relevant to the occasion and budget specified by the user.\n
                 At the end, you must strictly output only the query text for semantic search, with absolutely no other text, letters, or characters before it. The output must start with "Query:" and nothing else. The format is: "Query: Find a gift for [occasion] for [relationship] within [budget] and [user preferences]." Only this formatted output will be returned, with no additional explanation, greeting, or text before or after.\n
                 It only answers questions related to gift recommendations and does not provide any other information or assistance. Suggest to contact with the specialized one. \n
                """
                })
# === AI Chatbot ===
def chat_with_gpt(prompt):
    Messages.append({"role": "user", "content": prompt})
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=Messages,
        temperature=0.7
    )
    answer = response.choices[0].message.content
    Messages.append({"role": "assistant", "content": answer})
    return answer

def chat_history(occasion, price_str, answers):
    return Messages

# === Query Extraction and Semantic Search ===
def handle_query_response(response, products, index, embeddings):
    if "Query:" in response:
        semantic_search_query = response.split("Query:", 1)[1].strip()
        search_results = semantic_search(semantic_search_query, products, index, embeddings)
        
        Messages.append({"role": "assistant", "content": f"Suggested products:\n {semantic_search_query}"})
        return search_results
    return None

# === Main Functionality ===
if __name__ == "__main__":
    products = load_products()
    embeddings, index = embed_product_descriptions(products)
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        response = chat_with_gpt(user_input)
        search_results = handle_query_response(response, products, index, embeddings)
        if search_results is not None:
            print("Search Results:", json.dumps(search_results, indent=2))
            break
        else:
            print("Assistant:", response)