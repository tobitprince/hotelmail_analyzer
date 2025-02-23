from google import genai
from config import AppConfig

config = AppConfig()
API_TOKEN = config.GEMINI_API_KEY

client = genai.Client(api_key= API_TOKEN)
response = client.models.generate_content(
    model="gemini-2.0-flash", contents="Explain how AI works"
)
print(response.text)