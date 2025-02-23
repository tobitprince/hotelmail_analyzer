import pytest
from unittest.mock import patch
from chatgpt_email import classify_email
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.mark.parametrize("subject,body,mock_response,expected", [
    (
        "Booking Confirmation - Sarova Stanley",
        "Your room is confirmed for 2 nights...",
        "HOTEL",
        "HOTEL"  # Changed from "Hotel"
    ),
    (
        "Your Flight Itinerary",
        "Kenya Airways booking details...",
        "NOT_HOTEL",
        "NOT_HOTEL"  # Changed from "Other"
    ),
    (
        "Payment Receipt",
        "Your payment of $100 has been processed",
        "NOT_HOTEL",
        "NOT_HOTEL"  # Changed from "Other"
    ),
    (
        "Welcome to Mara Serena Safari Lodge",
        "Thank you for choosing our lodge...",
        "HOTEL",
        "HOTEL"  # Changed from "Hotel"
    ),
    (
        "Jumia Weekly Deals",
        "Check out our latest offers...",
        "NOT_HOTEL",
        "NOT_HOTEL"  # Changed from "Other"
    ),
    (
        "Safari Package with Accommodation",
        "Includes stays at luxury camps...",
        "HOTEL",
        "HOTEL"  # Changed from "Hotel"
    ),
    (
        "Conference Room Booking",
        "Your meeting room at Hilton is confirmed",
        "HOTEL",
        "HOTEL"  # Changed from "Hotel"
    )
])
def test_classify_email(subject, body, mock_response, expected):
    """Test email classification with proper response format."""
    with patch("chatgpt_email.query_huggingface_batch") as mock_query:
        # Configure mock with correct response format
        mock_query.return_value = [{"generated_text": mock_response}]
        
        # Call classify_email and log results
        result = classify_email(subject, body)
        logger.info(f"Test: '{subject}' - Expected: {expected}, Got: {result}")
        
        # Assert with proper expected format
        assert result == expected, f"Expected: {expected}, Got: {result}"

@pytest.mark.parametrize("error_scenario,expected_result", [
    ("api_failure", "NOT_HOTEL"),
    ("empty_response", "NOT_HOTEL"),
    ("invalid_response", "NOT_HOTEL"),
    ("network_timeout", "NOT_HOTEL")
])
def test_error_handling(error_scenario, expected_result):
    """Test error handling with correct response format."""
    with patch("chatgpt_email.query_huggingface_batch") as mock_query:
        # Configure error scenarios
        if error_scenario == "api_failure":
            mock_query.return_value = None
        elif error_scenario == "empty_response":
            mock_query.return_value = []
        elif error_scenario == "invalid_response":
            mock_query.return_value = [{"generated_text": "INVALID"}]
        else:  # network_timeout
            mock_query.side_effect = TimeoutError("Connection timeout")

        result = classify_email("Test Subject", "Test Body")
        assert result == expected_result, f"Error handling failed for {error_scenario}"

def test_edge_cases():
    """Test edge cases with correct response format."""
    test_cases = [
        ("", "", "NOT_HOTEL"),
        (None, None, "NOT_HOTEL"),
        ("🏨 Hotel", "Test", "HOTEL"),
        ("   Hotel   ", "   Test   ", "HOTEL")
    ]

    with patch("chatgpt_email.query_huggingface_batch") as mock_query:
        for subject, body, expected in test_cases:
            # Configure mock with proper response format
            mock_query.return_value = [{"generated_text": expected}]
            result = classify_email(subject, body)
            assert result == expected, f"Edge case failed: ({subject}, {body})"

if __name__ == "__main__":
    pytest.main(["-v", __file__])