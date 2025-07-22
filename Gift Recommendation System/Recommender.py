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
    response = openai.Embedding.create(
        api_key=openai.api_key,
        model="text-embedding-3-small",
        input=descriptions
    )
    embeddings = np.array([d["embedding"] for d in response["data"]], dtype="float32")
    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    # Save index and embeddings
    faiss.write_index(index, EMBEDDING_PATH)
    with open(PICKLE_PATH, "wb") as f:
        pickle.dump(embeddings, f)
        
    return embeddings, index

# === Semantic Search for products ===
def semantic_search(query, products):
    query_embedding = openai.Embedding.create(
        api_key=openai.api_key,
        model="text-embedding-3-small",
        input=query
    )["data"][0]["embedding"]
    
    # Calculate similarity scores
    scores = []
    for product in products:
        score = cosine_similarity(
            np.array(query_embedding).reshape(1, -1),
            np.array(product["embedding"]).reshape(1, -1)
        )[0][0]
        scores.append((product, score))
    
    # Sort by score
    scores.sort(key=lambda x: x[1], reverse=True)
    return [product for product, score in scores[:5]]

Messages = []
Messages.append({
    "role": "system", 
    "content": """You are a helpful gift recommendation assistant. 
                You will help users find the perfect gift based on their preferences and the product descriptions provided. 
                You will ask 3-5 questions to understand the user's needs and preferences, and then provide personalized recommendations.
                """
    })
# === AI Chatbot ===
def chat_with_ai(user_message):
    Messages.append({"role": "user", "content": user_message})
    # Here you would typically call your AI model with the Messages context
    # For now, let's just echo the user message
    ai_response = f"AI response to: {user_message}"
    Messages.append({"role": "assistant", "content": ai_response})
    return ai_response

# === Get Chat History ===
def get_chat_history():
    return Messages

# === Main Function ===
if __name__ == "__main__":
    products = load_products()
    if not products:
        print("No products available for recommendation.")
    else:
        embeddings, index = embed_product_descriptions(products)
        print("Product embeddings loaded and indexed.")
        
        # Example usage
        user_query = "I am looking for a gift for my friend's birthday."
        recommendations = semantic_search(user_query, products)
        print("Recommended Products:", recommendations)
        
        # Chat with AI
        user_message = "What gift do you recommend for a tech enthusiast?"
        ai_response = chat_with_ai(user_message)
        print("AI Response:", ai_response)
        
        # Get chat history
        chat_history = get_chat_history()
        print("Chat History:", chat_history)