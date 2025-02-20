from dataclasses import dataclass
from decouple import config

@dataclass
class AppConfig:
    CHUNK_SIZE_MB: int = 100
    BATCH_SIZE: int = 1
    GPU_MEMORY: int = 4096
    MBOX_FILE: str = config("MBOX_FILE", default="path/to/your_downloaded_file.mbox")
    OUTPUT_CSV: str = config("OUTPUT_CSV", default="tipjar_customers.csv")
    GOOGLE_PLACES_API_KEY: str = config("GOOGLE_PLACES_API_KEY")
    LM_STUDIO_API_URL: str = config("LM_STUDIO_API_URL", default="http://127.0.0.1:1234/v1/completions")
    LM_STUDIO_MODEL: str = config("LM_STUDIO_MODEL", default="mistral-7b-instruct-v0.2")
    LM_STUDIO_TIMEOUT: int = config("LM_STUDIO_TIMEOUT", cast=int, default=30)