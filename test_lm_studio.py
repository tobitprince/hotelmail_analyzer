# import requests
# import os
# import sys
# import json
# from dotenv import load_dotenv
# import logging

# logging.basicConfig(
#     level=logging.DEBUG,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# def test_lm_studio_connection():
#     if not load_dotenv():
#         logger.error("Could not load .env file")
#         return False

#     url = os.getenv('LM_STUDIO_API_URL')
#     model = os.getenv('LM_STUDIO_MODEL')

#     if not url or not model:
#         logger.error("Missing required environment variables")
#         return False

#     logger.info(f"Testing connection to: {url}")
#     logger.info(f"Using model: {model}")

#     try:
#         # Test with actual completion request
#         headers = {
#             "Content-Type": "application/json"
#         }

#         data = {
#             "model": model,
#             "prompt": "Hello, this is a test.",
#             "max_tokens": 10,
#             "temperature": 0.7
#         }

#         logger.debug(f"Sending request with data: {json.dumps(data, indent=2)}")

#         response = requests.post(url, headers=headers, json=data, timeout=10)
#         response.raise_for_status()

#         logger.info(f"Status code: {response.status_code}")
#         logger.info(f"Response: {response.text}")

#         if response.status_code == 200:
#             result = response.json()
#             if 'error' in result:
#                 logger.error(f"API Error: {result['error']}")
#                 return False
#             return True

#     except requests.exceptions.ConnectionError as e:
#         logger.error(f"Connection failed. Is LM Studio running? Error: {e}")
#     except requests.exceptions.Timeout:
#         logger.error("Connection timed out")
#     except requests.exceptions.RequestException as e:
#         logger.error(f"Request failed: {e}")
#     except Exception as e:
#         logger.error(f"Unexpected error: {e}")

#     return False

# if __name__ == "__main__":
#     success = test_lm_studio_connection()
#     sys.exit(0 if success else 1)


import requests
import logging
from config import AppConfig

def test_huggingface_connection():
    """Test connection to Hugging Face API"""
    config = AppConfig()
    url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-v0.1"
    headers = {"Authorization": f"Bearer {config.HUGGINGFACE_API_TOKEN}"}
    
    try:
        response = requests.post(url, headers=headers, json={"inputs": "test"}, timeout=10)
        response.raise_for_status()
        print(f"API Connection Successful: Status {response.status_code}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"API Connection Failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_huggingface_connection()