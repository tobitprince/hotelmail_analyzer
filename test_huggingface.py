import requests
import logging
from config import AppConfig
from log_config import setup_logging

# Initialize configuration and logging
logger = setup_logging()
config = AppConfig()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
HUGGINGFACE_API_TOKEN = config.HUGGINGFACE_API_TOKEN  # Correct variable name

# List of models to test
MODELS_TO_TEST = [
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "google/flan-t5-base",
    "facebook/bart-large",
    "EleutherAI/gpt-neo-1.3B"
]

def test_huggingface_connection(model):
    """Test connection to a specific Hugging Face model."""
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}", "Content-Type": "application/json"}
    payload = {"inputs": "test"}

    try:
        logger.info(f"Sending request to Hugging Face API for model: {model}")
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        logger.info(f"API Response for {model}: {result}")
        print(f"API Connection Successful for {model}: Status {response.status_code}")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"API Connection Failed for {model}: {str(e)}")
        print(f"API Connection Failed for {model}: {str(e)}")
        return False

def test_all_models():
    """Test connectivity for all specified models."""
    logger.info("Starting to test connectivity for all models")
    results = {}
    for model in MODELS_TO_TEST:
        success = test_huggingface_connection(model)
        results[model] = "Success" if success else "Failed"
        logger.info(f"Test result for {model}: {'Success' if success else 'Failed'}")
        print(f"Test result for {model}: {'Success' if success else 'Failed'}\n")

    # Summary
    logger.info("Test Summary:")
    print("Test Summary:")
    for model, status in results.items():
        logger.info(f"{model}: {status}")
        print(f"{model}: {status}")

if __name__ == "__main__":
    test_all_models()