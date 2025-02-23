"""
Email Processor for Hotel-Related Communications using Hugging Face API

This script processes MBOX email files to identify hotel-related emails and extract hotel names
using the Hugging Face Inference API (Mistral-7B model). It also fetches hotel details from the
Google Places API, stores results in a SQLite database, and exports them to a CSV file.

Dependencies:
- requests, sqlalchemy, python-decouple, tqdm
- Hugging Face API token
- Google Places API key

Environment Variables:
- MBOX_FILE: Path to the MBOX file
- OUTPUT_CSV: Path for the output CSV file
- GOOGLE_PLACES_API_KEY: Google Places API key
- HUGGINGFACE_API_TOKEN: Hugging Face API token
"""

import os
import mailbox
import re
import csv
import json
import requests
import logging
from datetime import datetime
from email.parser import BytesParser
from email import policy
from decouple import config
from tqdm import tqdm
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from config import AppConfig
from log_config import setup_logging

# Initialize configuration and logging
logger = setup_logging()
config = AppConfig()


# Hugging Face API setup
HUGGINGFACE_API_URL = config.HUGGINGFACE_API_URL
HEADERS = {"Authorization": f"Bearer {config.HUGGINGFACE_API_TOKEN}", "Content-Type": "application/json"}

if not config.HUGGINGFACE_API_TOKEN:
    raise ValueError("HUGGINGFACE_API_TOKEN is missing from environment variables")

# Database setup
Base = declarative_base()
engine = create_engine('sqlite:///huggingface_email_processing.db')
Session = sessionmaker(bind=engine)

class ProcessedEmail(Base):
    __tablename__ = 'processed_emails'
    id = Column(Integer, primary_key=True)
    message_id = Column(String, unique=True)
    date_processed = Column(DateTime, default=datetime.utcnow)
    is_hotel_related = Column(Boolean, default=False)
    error = Column(Text, nullable=True)

class Contact(Base):
    __tablename__ = 'contacts'
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True)
    display_name = Column(String)
    hotel_name = Column(String)
    website = Column(String)
    address = Column(String)
    coordinates = Column(String)
    contact = Column(String)
    subjects = Column(Text)  # JSON array
    dates = Column(Text)     # JSON array

Base.metadata.create_all(engine)

### Utility Functions

def test_huggingface_connection():
    """Test connection to Hugging Face API"""
    try:
        response = requests.post(
            HUGGINGFACE_API_URL,
            headers=HEADERS,
            json={"inputs": "test"},
            timeout=10
        )
        response.raise_for_status()
        logger.info(f"✓ Hugging Face API is running: Status {response.status_code}")
        print(f"API Connection Successful: Status {response.status_code}")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"× Hugging Face API connection error: {str(e)}")
        print(f"API Connection Failed: {str(e)}")
        return False

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def query_huggingface(prompt):
    """Query Hugging Face API with a prompt"""
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 100, "temperature": 0.7}
    }
    response = requests.post(HUGGINGFACE_API_URL, headers=HEADERS, json=payload, timeout=10)
    response.raise_for_status()
    result = response.json()
    return result[0].get("generated_text", "").strip()

def classify_email(subject, body):
    """Classify an email as hotel-related using Hugging Face"""
    prompt = (
        "Classify this email as 'Hotel' if it involves hotel bookings, confirmations, offers, or stays; "
        "otherwise 'Other'. Return ONLY a JSON object like {\"classification\": \"Hotel\"}. "
        f"Subject: {subject}\nBody: {body}"
    )
    try:
        result = query_huggingface(prompt)
        cleaned_result = re.sub(r'```json\s*|\s*```', '', result).strip()
        output = json.loads(cleaned_result)
        return output.get("classification", "Other")
    except Exception as e:
        logger.error(f"Error classifying email: {e}")
        return "Other"

def extract_hotel_name(subject, body):
    """Extract hotel name from email using Hugging Face"""
    prompt = (
        "Extract the hotel name from this email. Return ONLY a JSON object like {\"hotel_name\": \"Sarova Stanley\"} or {\"hotel_name\": \"\"}. "
        f"Subject: {subject}\nBody: {body}"
    )
    try:
        result = query_huggingface(prompt)
        cleaned_result = re.sub(r'```json\s*|\s*```', '', result).strip()
        output = json.loads(cleaned_result)
        return output.get("hotel_name", "")
    except Exception as e:
        logger.error(f"Error extracting hotel name: {e}")
        return ""

def get_hotel_details_from_google(hotel_name):
    """Fetch hotel details from Google Places API"""
    if not hotel_name or not config.GOOGLE_PLACES_API_KEY:
        return {}
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {"query": f"{hotel_name} hotel", "key": config.GOOGLE_PLACES_API_KEY}
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        results = response.json().get("results", [])
        if not results:
            return {}
        place = results[0]
        return {
            "Hotel Name": place.get("name", hotel_name),
            "Address": place.get("formatted_address", ""),
            "Coordinates": f"{place['geometry']['location']['lat']},{place['geometry']['location']['lng']}"
        }
    except Exception as e:
        logger.error(f"Error fetching Google Places data: {e}")
        return {}

def process_email(message):
    """Process a single email"""
    with Session() as session:
        message_id = message.get("Message-ID", "")
        if not message_id or session.query(ProcessedEmail).filter_by(message_id=message_id).first():
            return

        subject = message.get("subject", "")
        body = message.get_payload(decode=True).decode("utf-8", errors="ignore") if message.get_payload() else ""
        
        classification = classify_email(subject, body)
        is_hotel = classification.lower() == "hotel"

        if is_hotel:
            hotel_name = extract_hotel_name(subject, body)
            hotel_details = get_hotel_details_from_google(hotel_name)
            contact_data = {
                "Email": message.get("From", "").split()[-1].strip("<>"),
                "Display Name": message.get("From", "").split("<")[0].strip(),
                "Hotel Name": hotel_name,
                "Address": hotel_details.get("Address", ""),
                "Coordinates": hotel_details.get("Coordinates", ""),
                "Subjects": json.dumps([subject]),
                "Dates": json.dumps([message.get("Date", "")])
            }
            contact = session.query(Contact).filter_by(email=contact_data["Email"]).first()
            if not contact:
                session.add(Contact(**contact_data))
            else:
                contact.subjects = json.dumps(json.loads(contact.subjects) + [subject])
                contact.dates = json.dumps(json.loads(contact.dates) + [message.get("Date", "")])

        session.add(ProcessedEmail(message_id=message_id, is_hotel_related=is_hotel))
        session.commit()

def process_mbox(mbox_path):
    """Process emails from an MBOX file"""
    mbox = mailbox.mbox(mbox_path, factory=lambda f: BytesParser(policy=policy.default).parse(f))
    total = sum(1 for _ in mbox)
    mbox.seek(0)
    with tqdm(total=total, desc="Processing emails") as pbar:
        for message in mbox:
            process_email(message)
            pbar.update(1)
    mbox.close()

def write_csv(output_file):
    """Export contacts to CSV"""
    with Session() as session:
        contacts = session.query(Contact).all()
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["Email", "Display Name", "Hotel Name", "Address", "Coordinates", "Subjects", "Dates"])
            writer.writeheader()
            for contact in contacts:
                writer.writerow({
                    "Email": contact.email,
                    "Display Name": contact.display_name,
                    "Hotel Name": contact.hotel_name,
                    "Address": contact.address,
                    "Coordinates": contact.coordinates,
                    "Subjects": contact.subjects,
                    "Dates": contact.dates
                })

def main():
    """Main function"""
    if not test_huggingface_connection():
        logger.error("Cannot connect to Hugging Face API. Exiting.")
        return
    process_mbox(config.MBOX_FILE)
    write_csv(config.HUGGINGFACE_OUTPUT_CSV)
    logger.info("Processing complete!")

if __name__ == "__main__":
    main()