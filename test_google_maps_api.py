import logging
from openrouter_email import get_hotel_details_from_google
from config import AppConfig

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_api():
    config = AppConfig()
    logger.debug(f"Loaded GOOGLE_PLACES_API_KEY: {config.GOOGLE_PLACES_API_KEY[:5]}...")  # Masked for security

    hotels = [
        "Sarova Stanley",
        "Fairmont The Norfolk",
        "Mara Serena Safari Lodge"
    ]

    for hotel in hotels:
        logger.info(f"\nTesting hotel: {hotel}")
        result = get_hotel_details_from_google(hotel)
        logger.info(f"Results for {hotel}: {result}")

if __name__ == "__main__":
    test_api()