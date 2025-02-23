"""
Tests for hotelmail_analyzer.py using OpenRouter API.

This test suite verifies the email classification and hotel name extraction
functionality using mocked OpenRouter API responses.
"""

import pytest
from unittest.mock import patch
import logging
from chatgpt_email import classify_email, extract_hotel_name  # Adjust import based on your file name

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    )
])
def test_classify_email(subject, body, mock_response, expected):
    """Test email classification with OpenRouter responses."""
    with patch("chatgpt_email.query_openrouter_batch") as mock_query:
        mock_query.return_value = [{"generated_text": mock_response}]
        result = classify_email(subject, body)
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
    )
])
def test_extract_hotel_name(subject, body, mock_response, expected):
    """Test hotel name extraction with OpenRouter responses."""
    with patch("chatgpt_email.query_openrouter_batch") as mock_query:
        mock_query.return_value = [{"generated_text": mock_response}]
        result = extract_hotel_name(subject, body)
        logger.info(f"Test: '{subject}' - Expected: {expected}, Got: {result}")
        assert result == expected, f"Expected: {expected}, Got: {result}"

# Test error handling for classification
@pytest.mark.parametrize("error_scenario,expected_result", [
    ("api_failure", "Other"),
    ("empty_response", "Other"),
    ("invalid_json", "Other"),
    ("network_timeout", "Other")
])
def test_classify_email_error_handling(error_scenario, expected_result):
    """Test error handling in classify_email."""
    with patch("chatgpt_email.query_openrouter_batch") as mock_query:
        if error_scenario == "api_failure":
            mock_query.return_value = None
        elif error_scenario == "empty_response":
            mock_query.return_value = []
        elif error_scenario == "invalid_json":
            mock_query.return_value = [{"generated_text": "not_json"}]
        else:  # network_timeout
            mock_query.side_effect = TimeoutError("Connection timeout")

        result = classify_email("Test Subject", "Test Body")
        assert result == expected_result, f"Error handling failed for {error_scenario}: Got {result}"

# Test error handling for hotel name extraction
@pytest.mark.parametrize("error_scenario,expected_result", [
    ("api_failure", ""),
    ("empty_response", ""),
    ("invalid_json", ""),
    ("network_timeout", "")
])
def test_extract_hotel_name_error_handling(error_scenario, expected_result):
    """Test error handling in extract_hotel_name."""
    with patch("chatgpt_email.query_openrouter_batch") as mock_query:
        if error_scenario == "api_failure":
            mock_query.return_value = None
        elif error_scenario == "empty_response":
            mock_query.return_value = []
        elif error_scenario == "invalid_json":
            mock_query.return_value = [{"generated_text": "not_json"}]
        else:  # network_timeout
            mock_query.side_effect = TimeoutError("Connection timeout")

        result = extract_hotel_name("Test Subject", "Test Body")
        assert result == expected_result, f"Error handling failed for {error_scenario}: Got {result}"

# Test edge cases for classification
def test_classify_email_edge_cases():
    """Test edge cases for classify_email."""
    test_cases = [
        ("", "", '{"classification": "Other"}', "Other"),
        (None, None, '{"classification": "Other"}', "Other"),
        ("🏨 Hotel", "Test", '{"classification": "Hotel"}', "Hotel"),
        ("   Hotel   ", "   Test   ", '{"classification": "Hotel"}', "Hotel")
    ]
    with patch("chatgpt_email.query_openrouter_batch") as mock_query:
        for subject, body, mock_response, expected in test_cases:
            mock_query.return_value = [{"generated_text": mock_response}]
            result = classify_email(subject, body)
            assert result == expected, f"Edge case failed: ({subject}, {body}) - Expected: {expected}, Got: {result}"

# Test edge cases for hotel name extraction
def test_extract_hotel_name_edge_cases():
    """Test edge cases for extract_hotel_name."""
    test_cases = [
        ("", "", '{"hotel_name": ""}', ""),
        (None, None, '{"hotel_name": ""}', ""),
        ("🏨 Hotel", "Test", '{"hotel_name": "Test Hotel"}', "Test Hotel"),
        ("   ", "   Hotel XYZ   ", '{"hotel_name": "Hotel XYZ"}', "Hotel XYZ")
    ]
    with patch("chatgpt_email.query_openrouter_batch") as mock_query:
        for subject, body, mock_response, expected in test_cases:
            mock_query.return_value = [{"generated_text": mock_response}]
            result = extract_hotel_name(subject, body)
            assert result == expected, f"Edge case failed: ({subject}, {body}) - Expected: {expected}, Got: {result}"

if __name__ == "__main__":
    pytest.main(["-v", __file__])