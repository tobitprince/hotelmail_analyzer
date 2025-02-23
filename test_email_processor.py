import os
import csv
import unittest
from unittest.mock import patch, MagicMock

# Import the functions to test from your modules
from chatgpt_email import classify_email, extract_hotel_name, get_hotel_details_from_google, write_csv
from config import AppConfig

# Dummy functions for demonstration if they don't exist.
# In your project these functions already exist.
# def classify_email(subject): ...
# def extract_hotel_name(response_str): ...
# def get_hotel_details_from_google(hotel_name): ...
# def save_contact_to_csv(contact, filename='tipjar_customers1.csv'): ...

class TestEmailProcessor(unittest.TestCase):
  def setUp(self):
    # Use a test configuration file or dummy API keys
    self.config = AppConfig()
    self.test_csv = "test_contacts.csv"
    # Remove the file if it exists from previous tests.
    if os.path.exists(self.test_csv):
      os.remove(self.test_csv)

  @patch("chatgpt_email.send_request_to_openrouter")
  def test_classify_email_hotel(self, mock_send):
    # Simulate OpenRouter response for a hotel classification.
    mock_send.return_value = '{"classification": "Hotel"}'
    subject = "Booking Confirmation - Sarova Stanley, Nairobi"
    classification = classify_email(subject)
    self.assertEqual(classification, "Hotel")

  @patch("chatgpt_email.send_request_to_openrouter")
  def test_classify_email_other(self, mock_send):
    # Simulate response for a non-hotel email.
    mock_send.return_value = '{"classification": "Other"}'
    subject = "Your Flight Itinerary"
    classification = classify_email(subject)
    self.assertEqual(classification, "Other")

  def test_extract_hotel_name(self):
    # Provide a sample raw response string from OpenRouter for hotel name extraction.
    raw_response = '{"hotel_name": "The Sarova Stanley"}'
    hotel_name = extract_hotel_name(raw_response)
    self.assertEqual(hotel_name, "The Sarova Stanley")

  @patch("chatgpt_email.requests.get")
  def test_google_places_details(self, mock_get):
    # Simulate successful Google Places API responses.
    # First call returns textsearch; second call returns place details
    textsearch_response = MagicMock()
    textsearch_response.json.return_value = {
      'status': 'OK',
      'results': [{
        'formatted_address': 'Harry Thuku Rd, Nairobi, Kenya',
        'geometry': {'location': {'lat': -1.278375, 'lng': 36.8163346}},
        'photos': [{'height': 854, 'width': 1280}],
        'place_id': 'PLACEID123'
      }]
    }
    detail_response = MagicMock()
    detail_response.json.return_value = {
      'status': 'OK',
      'result': {
        'name': 'Fairmont The Norfolk',
        'website': 'https://www.fairmont.com/norfolk-hotel-nairobi/',
        'formatted_address': 'Harry Thuku Rd, Nairobi, Kenya',
        'formatted_phone_number': '020 2265000'
      }
    }
    mock_get.side_effect = [textsearch_response, detail_response]
    details = get_hotel_details_from_google("Fairmont The Norfolk")
    self.assertIsInstance(details, dict)
    self.assertIn("Hotel Name", details)
    self.assertEqual(details.get("Hotel Name"), "Fairmont The Norfolk")

  def test_csv_write_success(self):
    # Simulate saving a contact to CSV.
    contact = {
      "Hotel Name": "Test Hotel",
      "Website": "http://testhotel.com",
      "Address": "Test Address",
      "Coordinates": "0,0",
      "Contact": "000"
    }
    # This function should write the data without exception.
    save_contact_to_csv(contact, filename=self.test_csv)
    # Now read back and check.
    with open(self.test_csv, "r", newline="") as csvfile:
      reader = csv.DictReader(csvfile)
      rows = list(reader)
      self.assertEqual(len(rows), 1)
      self.assertEqual(rows[0]["Hotel Name"], "Test Hotel")

  def test_csv_write_permission_error(self):
    # Simulate permission error when writing CSV.
    contact = {
      "Hotel Name": "Test Hotel 2",
      "Website": "http://testhotel2.com",
      "Address": "Test Address 2",
      "Coordinates": "1,1",
      "Contact": "111"
    }
    # Create a file and open it in read-only mode.
    with open(self.test_csv, "w") as f:
      f.write("dummy")
    os.chmod(self.test_csv, 0o400)  # read-only
    with self.assertRaises(PermissionError):
      save_contact_to_csv(contact, filename=self.test_csv)
    # Cleanup: set back permission
    os.chmod(self.test_csv, 0o600)
    os.remove(self.test_csv)

if __name__ == "__main__":
  unittest.main()