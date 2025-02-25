from dataclasses import dataclass
from dotenv import load_dotenv, find_dotenv
import os
import logging

logger = logging.getLogger(__name__)

@dataclass
class AppConfig:
    def __init__(self):
        # Force load from .env file
        dotenv_path = find_dotenv()
        load_dotenv(dotenv_path, override=True)

        # Load configuration values
        self.GOOGLE_PLACES_API_KEY: str = os.getenv("GOOGLE_PLACES_API_KEY")
        self.MBOX_FILE: str = os.getenv("MBOX_FILE", "path/to/your_downloaded_file.mbox")
        self.OUTPUT_CSV: str = os.getenv("OUTPUT_CSV", "tipjar_customers.csv")
        self.GEMINI_OUTPUT_CSV: str = os.getenv("GEMINI_OUTPUT_CSV", "gemini_tipjar_customers.csv")
        self.OPENROUTER_OUTPUT_CSV: str = os.getenv("OPENROUTER_OUTPUT_CSV", "openrouter_tipjar_customers.csv")
        self.LMSTUDIO_OUTPUT_CSV: str = os.getenv("LMSTUDIO_OUTPUT_CSV", "lmstudio_tipjar_customers.csv")
        self.HUGGINGFACE_OUTPUT_CSV: str = os.getenv("HUGGINGFACE_OUTPUT_CSV", "huggingface_tipjar_customers.csv")
        self.CHUNK_SIZE_MB: int = 100
        self.BATCH_SIZE: int = 1
        self.GPU_MEMORY: int = 4096
        self.LM_STUDIO_API_URL: str = os.getenv("LM_STUDIO_API_URL", "http://127.0.0.1:1234/v1/completions")
        self.LM_STUDIO_MODEL: str = os.getenv("LM_STUDIO_MODEL", "mistral-7b-instruct-v0.2")
        self.LM_STUDIO_TIMEOUT: int = int(os.getenv("LM_STUDIO_TIMEOUT", "30"))
        self.HUGGINGFACE_API_URL: str = os.getenv("HUGGINGFACE_API_URL", "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-v0.1")
        self.HUGGINGFACE_API_TOKEN: str = os.getenv("HUGGINGFACE_API_TOKEN")
        self.HUGGINGFACE_MODEL: str = os.getenv("HUGGINGFACE_MODEL", "gpt2")
        self.OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY")
        self.OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "gpt2")
        self.GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")

        # Log critical configuration values
        if not self.GOOGLE_PLACES_API_KEY:
            logger.warning("GOOGLE_PLACES_API_KEY not found in configuration")