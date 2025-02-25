import requests
from config import AppConfig

config = AppConfig()

# Your OpenRouter API key
API_TOKEN = config.OPENROUTER_API_KEY # Your actual key
OPENROUTER_MODEL = config.OPENROUTER_MODEL
# OpenRouter API endpoint
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Headers for authentication
headers = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}

def get_response(user_input):
    # Payload with system prompt
    payload = {
        "model": f"{OPENROUTER_MODEL}",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Always respond clearly and concisely."},
            {"role": "user", "content": user_input}
        ],
        "max_tokens": 50,      # Keep responses short
        "temperature": 0.7,    # Balanced creativity
        "top_p": 0.9           # Filter unlikely tokens
    }

    # Send request to OpenRouter
    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        result = response.json()
        raw_output = result["choices"][0]["message"]["content"]
        print("Raw output:", raw_output)  # Debug line
        return raw_output.strip() if raw_output.strip() else "Hi! How can I help you?"
    else:
        return f"Error {response.status_code}: {response.text}"

# Simple chat loop
while True:
    message = input("You: ")
    if message.lower() == "quit":
        break
    reply = get_response(message)
    print("Bot:", reply)