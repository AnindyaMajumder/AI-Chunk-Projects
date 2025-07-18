import openai
import json
from dotenv import load_dotenv
import os 

load_dotenv()

# === Configuration ===
openai.api_key = os.getenv("OPENAI_API_KEY")

# === Product Data ===
products = [
    {
        "name": "Personalized Tumbler + Plant Set",
        "description": "Thoughtful and practical with a custom tumbler and plant.",
        "image": "https://cdn.shopify.com/s/files/1/0946/1625/6825/files/for.png?v=1750446667",
        "price": "75.00",
        "url": "https://farawaytogether.com/products/tumbler-30oz-plant"
    },
    {
        "name": "Personalized Plant Label Gift Set",
        "description": "Customized planter stakes for a personal touch.",
        "image": "https://cdn.shopify.com/s/files/1/0946/1625/6825/files/Untitled_design_-_2025-06-24T114938.168.png?v=1750956200",
        "price": "5.00",
        "url": "https://farawaytogether.com/products/personalized-plant"
    },
    {
        "name": "Personalized 30 oz. Stanley Tumbler",
        "description": "Custom engraving adds a personal touch, making it a thoughtful gift.",
        "image": "https://cdn.shopify.com/s/files/1/0946/1625/6825/files/Untitled_design_-_2025-06-19T140337.665.png?v=1750368378",
        "price": "70.00",
        "url": "https://farawaytogether.com/products/tumbler-30oz"
    },
    {
        "name": "Personalized 14 oz. Stanley Tumbler",
        "description": "Great size for kids with a custom engraved name.",
        "image": "https://cdn.shopify.com/s/files/1/0946/1625/6825/files/ChatGPTImageJun9_2025_01_42_09PM.png?v=1750307444",
        "price": "14.50",
        "url": "https://farawaytogether.com/products/faraway-families-childrens-picture-book-english-version-test"
    }
]

# === GPT Integration ===
def recommend_with_gpt(occasion, price_str, answers):
    prompt = f"""
You are a helpful gift assistant. Your job is to:
1. Recommend a gift from the list below.
2. Generate a short personal story related to the user's answers and the gift.
3. Propose a meaningful memory-making activity for the user to do with or around the gift.

Occasion: {occasion}
Budget: {price_str}
User’s answers: {answers}

Here is the list of available products:
{json.dumps(products, indent=2)}

Return a JSON object with:
{{
  "gift": {{
    "name": "...",
    "reason": "...",
    "url": "...",
    "image": "..."
  }},
  "story": "...",
  "memory_maker": "..."
}}
"""
    res = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return json.loads(res.choices[0].message["content"])

# === Main Runner ===
if __name__ == "__main__":
    # Example structured input
    json_input = '''{
      "user_name": "Rahim",
      "occasion": "Birthday",
      "price_range": "$50 – $100",
      "response_form_data": [
        {"question": "When choosing a gift, what matters most to you?", "response_text": "Creating excitement and surprise"},
        {"question": "Ideally, what emotional reaction do you hope your gift sparks?", "response_text": "Joy and delight"},
        {"question": "What guides your gift choices most often?", "response_text": "Popular trends or universally appealing gifts"},
        {"question": "If your gift isn't received enthusiastically, you tend to feel:", "response_text": "Uncertain about your relationship with the recipient"},
        {"question": "Why do you most often choose to give gifts?", "response_text": "To strengthen relationships"},
        {"question": "What type of gifts do you feel proudest to give?", "response_text": "Impressive or luxurious gifts that reflect your status"},
        {"question": "When selecting a gift, you most frequently feel:", "response_text": "Anxious about making the perfect choice"},
        {"question": "What meaning do you typically associate with your gifts?", "response_text": "Proof of your generosity or thoughtfulness"},
        {"question": "How do you want recipients to think of you after receiving your gift?", "response_text": "They truly understand me."},
        {"question": "Ultimately, what would make your gift-giving feel more meaningful?", "response_text": "Creating more lasting emotional memories"}
      ]
    }'''

    # Parse JSON
    user_data = json.loads(json_input)
    occasion = user_data.get("occasion", "Birthday")
    price_range = user_data.get("price_range", "$50 – $100")
    answers = [entry["response_text"] for entry in user_data.get("response_form_data", [])]

    # Call GPT
    print("\n⌛ Generating personalized recommendation via GPT...")
    result = recommend_with_gpt(occasion, price_range, answers)

    # Display results
    gift = result["gift"]   
    print("\n🎯 GPT Recommended Gift:")
    print(f"- Name: {gift['name']}")
    print(f"- Reason: {gift['reason']}")
    print(f"- URL: {gift['url']}")
    print(f"- Image: {gift['image']}")

    print("\n📖 Personalized Story:")
    print(result["story"])

    print("\n🌟 Memory Maker Idea:")
    print(result["memory_maker"])