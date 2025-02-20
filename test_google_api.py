import logging
from chatgpt_email import get_hotel_details_from_google

logging.basicConfig(level=logging.DEBUG)

def test_api():
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