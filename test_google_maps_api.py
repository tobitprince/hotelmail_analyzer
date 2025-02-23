import logging
from chatgpt_email import get_hotel_details_from_google
from config import AppConfig

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_api():
    #config = AppConfig()
    #print(f"Config loaded API key: {config.GOOGLE_PLACES_API_KEY}")  # Debug line
    
    hotels = [
        "Sarova Stanley",
        "Fairmont The Norfolk",
        "Mara Serena Safari Lodge"
    ]

    for hotel in hotels:
        print(f"\nTesting hotel: {hotel}")
        result = get_hotel_details_from_google(hotel)
        print(f"Results: {result}")

if __name__ == "__main__":
    test_api()