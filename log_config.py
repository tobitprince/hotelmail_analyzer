import logging
import sys
from logging.handlers import RotatingFileHandler
from config import AppConfig  # Replace with wherever your API key is stored

config = AppConfig()  # Adjust this to match how you access your GOOGLE_PLACES_API_KEY

# Custom filter to mask sensitive data
class SensitiveDataFilter(logging.Filter):
    def __init__(self, api_key):
        super().__init__()
        self.api_key = api_key

    def filter(self, record):
        if self.api_key in record.msg:
            record.msg = record.msg.replace(self.api_key, 'API_KEY_REDACTED')
        return True

def setup_logging():
    """Configure logging to both file and console."""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')

    # File handler
    file_handler = RotatingFileHandler(
        'email_processor.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    # Add filter to mask the API key
    api_key = config.GOOGLE_PLACES_API_KEY  # Replace with your actual API key variable
    sensitive_filter = SensitiveDataFilter(api_key)
    file_handler.addFilter(sensitive_filter)
    console_handler.addFilter(sensitive_filter)

    # Remove any existing handlers
    logger.handlers = []

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger