import pytest
from chatgpt_email import classify_email

@pytest.mark.parametrize("subject,body,expected", [
    (
        "Booking Confirmation - Sarova Stanley",
        "Your room is confirmed for...",
        True
    ),
    (
        "Your Flight Itinerary",
        "Kenya Airways booking details...",
        False
    ),
    (
        "Payment Receipt",
        "Your payment of $100 has been processed",
        False
    ),
    (
        "Welcome to Mara Serena Safari Lodge",
        "Thank you for choosing our lodge...",
        True
    ),
    (
        "Jumia Weekly Deals",
        "Check out our latest offers...",
        False
    ),
])
def test_classify_email(subject, body, expected):
    assert classify_email(subject, body) == expected