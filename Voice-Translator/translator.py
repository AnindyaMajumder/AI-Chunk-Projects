from openai import OpenAI
client = OpenAI()

def translator(text, target_language):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a translation assistant. Only translate the text, do not include any additional information."},
            {"role": "user", "content": f"Translate the following text to {target_language}: {text}"}
        ]
    )
    translated_text = response.choices[0].message.content.strip()
    print(f"Translated text: {translated_text}\n")
    return translated_text