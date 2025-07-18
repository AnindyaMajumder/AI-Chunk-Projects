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
    },
        {
        "name": "Personalized Garden Tool Set",
        "description": "Custom-engraved handles for the plant lover in your life.",
        "image": "https://cdn.shopify.com/s/files/1/0946/1625/6825/files/garden_tool_set_custom.png?v=1751205100",
        "price": "38.00",
        "url": "https://farawaytogether.com/products/personalized-garden-tool-set"
    },
    {
        "name": "Custom Glass Water Bottle with Sleeve",
        "description": "Stay hydrated in style with a name-engraved bottle.",
        "image": "https://cdn.shopify.com/s/files/1/0946/1625/6825/files/glass_bottle_custom_sleeve.png?v=1751205555",
        "price": "28.00",
        "url": "https://farawaytogether.com/products/custom-glass-water-bottle"
    },
    {
        "name": "Herb Garden Starter Kit",
        "description": "All-in-one kit for growing fresh herbs at home.",
        "image": "https://cdn.shopify.com/s/files/1/0946/1625/6825/files/herb_garden_kit.png?v=1751206000",
        "price": "45.00",
        "url": "https://farawaytogether.com/products/herb-garden-starter-kit"
    },
    {
        "name": "Custom Engraved Wooden Gift Box",
        "description": "A reusable keepsake box engraved with any message.",
        "image": "https://cdn.shopify.com/s/files/1/0946/1625/6825/files/custom_wooden_gift_box.png?v=1751206344",
        "price": "20.00",
        "url": "https://farawaytogether.com/products/custom-wooden-box"
    },
    {
        "name": "Personalized Eco Tote Bag",
        "description": "Reusable canvas tote printed with a name or phrase.",
        "image": "https://cdn.shopify.com/s/files/1/0946/1625/6825/files/custom_eco_tote.png?v=1751206682",
        "price": "18.00",
        "url": "https://farawaytogether.com/products/personalized-eco-tote"
    }
]

# === GPT Integration ===
def recommend_with_gpt(occasion, price_str, answers):
    prompt = f"""
You are a helpful gift assistant. Your job is to:
1. Recommend more than one gift (at least 2) from the list below that fit the occasion and budget. The number of gifts is up to your judgment, but always suggest more than one.
2. For each gift, briefly explain the reason for your recommendation.

Occasion: {occasion}
Budget: {price_str}

Here is the list of available products:
{json.dumps(products, indent=2)}

Return a JSON object with:
{{
  "gifts": [
    {{
      "name": "...",
      "reason": "...",
      "url": "...",
      "image": "..."
    }}
    // ... more gifts ...
  ]
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
    # Get user input for occasion and price range
    occasion = input("Enter the occasion (e.g., Birthday, Anniversary): ").strip()
    price_range = input("Enter your price range (e.g., $20-$50): ").strip()

    # No questions/answers, just pass empty list for answers
    print("\n⌛ Generating personalized recommendation via GPT...")
    try:
        result = recommend_with_gpt(occasion, price_range, [])
    except json.JSONDecodeError:
        print("\n❌ Error: The AI response was not valid JSON. Please try again or check your OpenAI API usage.")
        exit(1)

    # Display results
    gifts = result.get("gifts", [])
    if not gifts:
        print("\nNo gift recommendations found.")
    else:
        print("\n🎯 GPT Recommended Gifts:")
        for idx, gift in enumerate(gifts, 1):
            print(f"\nGift {idx}:")
            print(f"- Name: {gift.get('name','')}")
            print(f"- Reason: {gift.get('reason','')}")
            print(f"- URL: {gift.get('url','')}")
            print(f"- Image: {gift.get('image','')}")