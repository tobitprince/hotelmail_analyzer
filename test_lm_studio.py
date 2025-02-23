import requests
import logging
from config import AppConfig

def test_huggingface_connection():
    """Test connection to Hugging Face API"""
    config = AppConfig()
    url = config.HUGGINGFACE_API_URL
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
