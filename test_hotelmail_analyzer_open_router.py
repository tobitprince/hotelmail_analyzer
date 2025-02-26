"""
Tests for openrouter_email.py using OpenRouter API.

This test suite verifies the email classification and hotel name extraction
functionality using mocked OpenRouter API responses.
"""

import pytest
from unittest.mock import patch
import logging
from openrouter_email import classify_email, extract_hotel_name  # Ensure correct import
from config import AppConfig

# Setup logging
logging.basicConfig(level=logging.DEBUG)  # Match your script's verbosity
logger = logging.getLogger(__name__)

# Define a fixture for config
@pytest.fixture
def config():
    """Fixture to provide a mock AppConfig instance."""
    class MockAppConfig:
        OPENROUTER_API_KEY = "mock_key"
        OPENROUTER_MODEL = "cognitivecomputations/dolphin-mixtral-8x7b"
        GOOGLE_PLACES_API_KEY = "mock_google_key"
        MBOX_FILE = "test_emails.mbox"
        OPENROUTER_OUTPUT_CSV = "openrouter_tipjar_customers.csv"
        BATCH_SIZE = 25
    return MockAppConfig()

# Test classification
@pytest.mark.parametrize("subject,body,mock_response,expected", [
    (
        "Booking Confirmation - Sarova Stanley",
        "Your room is confirmed for 2 nights...",
        '{"classification": "Hotel"}',
        "Hotel"
    ),
    (
        "Your Flight Itinerary",
        "Kenya Airways booking details...",
        '{"classification": "Other"}',
        "Other"
    ),
    (
        "Payment Receipt",
        "Your payment of $100 has been processed",
        '{"classification": "Other"}',
        "Other"
    ),
    (
        "Welcome to Mara Serena Safari Lodge",
        "Thank you for choosing our lodge...",
        '{"classification": "Hotel"}',
        "Hotel"
    ),
    (
        "Jumia Weekly Deals",
        "Check out our latest offers...",
        '{"classification": "Other"}',
        "Other"
    ),
    (
        "Safari Package with Accommodation",
        "Includes stays at luxury camps...",
        '{"classification": "Hotel"}',
        "Hotel"
    ),
    (
        "Conference Room Booking",
        "Your meeting room at Hilton is confirmed",
        '{"classification": "Hotel"}',
        "Hotel"
    ),
    # New cases from your recent MBOX
    (
        "Exclusive Conference Packages Await at Sarit Expo Centre! 💼",
        "Book your event space at Sarit Expo Centre, accommodation options available...",
        '{"classification": "Hotel"}',
        "Hotel"
    ),
    (
        "Updated invitation: Mombasa Trip, KATA & Sarit Expo @ Thu Jan 30, 2025",
        "Join us for a travel expo with stays at coastal hotels...",
        '{"classification": "Hotel"}',
        "Hotel"
    ),
    (
        "SHUKRAN AT SCARLET VACATIONS",
        "Enjoy your stay at Scarlet Resort...",
        '{"classification": "Hotel"}',
        "Hotel"
    )
])
def test_classify_email(subject, body, mock_response, config, expected):
    """Test email classification with OpenRouter responses."""
    with patch("openrouter_email.query_openrouter_batch") as mock_query:
        mock_query.return_value = [{"generated_text": mock_response}]
        result = classify_email(subject, body, config)
        logger.info(f"Test: '{subject}' - Expected: {expected}, Got: {result}")
        assert result == expected, f"Expected: {expected}, Got: {result}"

# Test hotel name extraction
@pytest.mark.parametrize("subject,body,mock_response,expected", [
    (
        "Booking Confirmation - Sarova Stanley",
        "Your room at Sarova Stanley is confirmed...",
        '{"hotel_name": "Sarova Stanley"}',
        "Sarova Stanley"
    ),
    (
        "Your Flight Itinerary",
        "Kenya Airways booking details...",
        '{"hotel_name": ""}',
        ""
    ),
    (
        "Welcome to Mara Serena Safari Lodge",
        "Thank you for choosing Mara Serena...",
        '{"hotel_name": "Mara Serena Safari Lodge"}',
        "Mara Serena Safari Lodge"
    ),
    (
        "General Email",
        "No hotel mentioned here...",
        '{"hotel_name": ""}',
        ""
    ),
    (
        "Safari Package with Accommodation",
        "Stay at Amboseli Serena Safari Lodge...",
        '{"hotel_name": "Amboseli Serena Safari Lodge"}',
        "Amboseli Serena Safari Lodge"
    ),
    # New cases from your recent MBOX
    (
        "Exclusive Conference Packages Await at Sarit Expo Centre! 💼",
        "Event at Sarit Expo Centre, stay at nearby Hilton Nairobi...",
        '{"hotel_name": "Hilton Nairobi"}',
        "Hilton Nairobi"
    ),
    (
        "SHUKRAN AT SCARLET VACATIONS",
        "Enjoy your stay at Scarlet Resort...",
        '{"hotel_name": "Scarlet Resort"}',
        "Scarlet Resort"
    )
])
def test_extract_hotel_name(subject, body, mock_response, config, expected):
    """Test hotel name extraction with OpenRouter responses."""
    with patch("openrouter_email.query_openrouter_batch") as mock_query:
        mock_query.return_value = [{"generated_text": mock_response}]
        result = extract_hotel_name(subject, body, config)
        logger.info(f"Test: '{subject}' - Expected: {expected}, Got: {result}")
        assert result == expected, f"Expected: {expected}, Got: {result}"

# Test error handling for classification
@pytest.mark.parametrize("error_scenario,expected_result", [
    ("api_failure", "Other"),
    ("empty_response", "Other"),
    ("invalid_json", "Other"),
    ("network_timeout", "Other"),
    ("payment_required", "Other")  # Added from your recent 402 error
])
def test_classify_email_error_handling(error_scenario, expected_result, config):
    """Test error handling in classify_email."""
    with patch("openrouter_email.query_openrouter_batch") as mock_query:
        if error_scenario == "api_failure":
            mock_query.return_value = None
        elif error_scenario == "empty_response":
            mock_query.return_value = []
        elif error_scenario == "invalid_json":
            mock_query.return_value = [{"generated_text": "not_json"}]
        elif error_scenario == "network_timeout":
            mock_query.side_effect = TimeoutError("Connection timeout")
        elif error_scenario == "payment_required":
            from requests.exceptions import HTTPError
            mock_query.side_effect = HTTPError("402 Client Error: Payment Required")
        
        result = classify_email("Test Subject", "Test Body", config)
        assert result == expected_result, f"Error handling failed for {error_scenario}: Got {result}"

# Test error handling for hotel name extraction
@pytest.mark.parametrize("error_scenario,expected_result", [
    ("api_failure", ""),
    ("empty_response", ""),
    ("invalid_json", ""),
    ("network_timeout", ""),
    ("payment_required", "")  # Added from your recent 402 error
])
def test_extract_hotel_name_error_handling(error_scenario, expected_result, config):
    """Test error handling in extract_hotel_name."""
    with patch("openrouter_email.query_openrouter_batch") as mock_query:
        if error_scenario == "api_failure":
            mock_query.return_value = None
        elif error_scenario == "empty_response":
            mock_query.return_value = []
        elif error_scenario == "invalid_json":
            mock_query.return_value = [{"generated_text": "not_json"}]
        elif error_scenario == "network_timeout":
            mock_query.side_effect = TimeoutError("Connection timeout")
        elif error_scenario == "payment_required":
            from requests.exceptions import HTTPError
            mock_query.side_effect = HTTPError("402 Client Error: Payment Required")
        
        result = extract_hotel_name("Test Subject", "Test Body", config)
        assert result == expected_result, f"Error handling failed for {error_scenario}: Got {result}"

# Test edge cases for classification
def test_classify_email_edge_cases(config):
    """Test edge cases for classify_email."""
    test_cases = [
        ("", "", '{"classification": "Other"}', "Other"),
        (None, None, '{"classification": "Other"}', "Other"),
        ("🏨 Hotel", "Test", '{"classification": "Hotel"}', "Hotel"),
        ("   Hotel   ", "   Test   ", '{"classification": "Hotel"}', "Hotel"),
        ("Invalid UTF-8 \ud800", "Test", '{"classification": "Other"}', "Other")  # Invalid Unicode
    ]
    with patch("openrouter_email.query_openrouter_batch") as mock_query:
        for subject, body, mock_response, expected in test_cases:
            mock_query.return_value = [{"generated_text": mock_response}]
            result = classify_email(subject or "", body or "", config)
            assert result == expected, f"Edge case failed: ({subject}, {body}) - Expected: {expected}, Got: {result}"

# Test edge cases for hotel name extraction
def test_extract_hotel_name_edge_cases(config):
    """Test edge cases for extract_hotel_name."""
    test_cases = [
        ("", "", '{"hotel_name": ""}', ""),
        (None, None, '{"hotel_name": ""}', ""),
        ("🏨 Hotel", "Test", '{"hotel_name": "Test Hotel"}', "Test Hotel"),
        ("   ", "   Hotel XYZ   ", '{"hotel_name": "Hotel XYZ"}', "Hotel XYZ"),
        ("Invalid UTF-8 \ud800", "Test", '{"hotel_name": "Test Hotel"}', "Test Hotel")  # Invalid Unicode
    ]
    with patch("openrouter_email.query_openrouter_batch") as mock_query:
        for subject, body, mock_response, expected in test_cases:
            mock_query.return_value = [{"generated_text": mock_response}]
            result = extract_hotel_name(subject or "", body or "", config)
            assert result == expected, f"Edge case failed: ({subject}, {body}) - Expected: {expected}, Got: {result}"

if __name__ == "__main__":
    pytest.main(["-v", __file__])