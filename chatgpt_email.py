"""
Email Processor for Hotel-Related Communications

This module processes MBOX email files to identify and extract information about hotel-related
communications. It uses OpenRouter with Dolphin 3.0 R1 Mistral 24B for email classification and
the Google Places API for fetching additional hotel details.

Key Features:
- Processes large MBOX files in manageable chunks
- Uses AI for intelligent email classification
- Extracts hotel names and contact information
- Integrates with Google Places API
- Maintains a SQLite database
- Exports results to CSV
- Supports batch processing for GPU efficiency

Dependencies:
- OpenRouter API key (free tier)
- Google Places API key
- Python packages: sqlalchemy, requests, python-decouple, tqdm, etc.

Environment Variables:
- MBOX_FILE: Path to the MBOX file
- OUTPUT_CSV: Path for the output CSV file
- GOOGLE_PLACES_API_KEY: Google Places API key
- OPENROUTER_API_KEY: OpenRouter API key
"""

import os
import mailbox
import re
import csv
import json
import requests
import logging
from datetime import datetime
from email import policy
import time
import tempfile
from email.parser import BytesParser
from email.utils import getaddresses
from decouple import config
from tqdm import tqdm
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session
from sqlalchemy.exc import SQLAlchemyError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager

from config import AppConfig
from log_config import setup_logging

# Initialize configuration and logging
setup_logging()
logger = logging.getLogger(__name__)
config = AppConfig()

# OpenRouter API setup
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = config.OPENROUTER_API_KEY  # Add this to your .env file
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY is missing from environment variables")

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json"
}

# Database setup
Base = declarative_base()
engine = create_engine('sqlite:///email_processing2.db',
                      poolclass=QueuePool,
                      pool_size=5,
                      max_overflow=10,
                      pool_timeout=30)
Session = sessionmaker(bind=engine)

@contextmanager
def get_db_session():
    """Context manager for database sessions."""
    session = Session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

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
    subjects = Column(Text)  # Stored as JSON array
    dates = Column(Text)     # Stored as JSON array

Base.metadata.create_all(engine)

# Existing functions (unchanged): get_total_emails, is_email_processed, save_processed_email, save_contact
def get_total_emails(mbox_path):
    try:
        if not os.path.exists(mbox_path):
            print(f"Error: Mbox file not found at {mbox_path}")
            return 0
        file_size = os.path.getsize(mbox_path)
        if file_size == 0:
            print("Error: Mbox file is empty")
            return 0
        print(f"Loading mbox file ({file_size/1024/1024:.2f} MB)...")
        count = 0
        mbox = mailbox.mbox(mbox_path)
        print("Counting messages...")
        for _ in mbox:
            count += 1
            if count % 100 == 0:
                print(f"Counted {count} messages...")
        mbox.close()
        print(f"Found {count} total messages")
        return count
    except Exception as e:
        print(f"Error counting emails: {str(e)}")
        return 0

def is_email_processed(session, message_id):
    if not message_id:
        return False
    message_id = message_id.strip()
    try:
        return session.query(ProcessedEmail).filter(ProcessedEmail.message_id == message_id).first() is not None
    except SQLAlchemyError as e:
        logging.error(f"Database error checking if email is processed: {e}")
        return False

def save_processed_email(session, message_id, is_hotel=False, error=None):
    try:
        existing = session.query(ProcessedEmail).filter(ProcessedEmail.message_id == message_id.strip()).first()
        if existing:
            existing.date_processed = datetime.now()
            existing.is_hotel_related = is_hotel
            existing.error = error
        else:
            new_email = ProcessedEmail(
                message_id=message_id.strip(),
                date_processed=datetime.now(),
                is_hotel_related=is_hotel,
                error=error
            )
            session.add(new_email)
        session.commit()
    except SQLAlchemyError as e:
        logger.error(f"Database error saving processed email: {e}")
        session.rollback()

def save_contact(session, contact_data):
    try:
        email = contact_data.get("Email")
        if not email:
            logger.error("Attempted to save contact without email address")
            return None
        subjects = list(contact_data.get("Subjects", set()))
        dates = list(contact_data.get("Dates", set()))
        contact = session.query(Contact).filter_by(email=email).first()
        if not contact:
            contact = Contact(
                email=email,
                display_name=contact_data.get("Display Name", ""),
                hotel_name=contact_data.get("Hotel Name", ""),
                website=contact_data.get("Website", ""),
                address=contact_data.get("Address", ""),
                coordinates=contact_data.get("Coordinates", ""),
                contact=contact_data.get("Contact", ""),
                subjects=json.dumps(subjects),
                dates=json.dumps(dates)
            )
            session.add(contact)
        else:
            existing_subjects = set(json.loads(contact.subjects)) if contact.subjects else set()
            existing_dates = set(json.loads(contact.dates)) if contact.dates else set()
            updated_subjects = list(existing_subjects | set(subjects))
            updated_dates = list(existing_dates | set(dates))
            contact.subjects = json.dumps(updated_subjects)
            contact.dates = json.dumps(updated_dates)
            if not contact.hotel_name and contact_data.get("Hotel Name"):
                contact.hotel_name = contact_data["Hotel Name"]
                contact.website = contact_data.get("Website", "")
                contact.address = contact_data.get("Address", "")
                contact.coordinates = contact_data.get("Coordinates", "")
                contact.contact = contact_data.get("Contact", "")
        session.commit()
        logger.info(f"Successfully saved/updated contact: {email}")
        return contact
    except Exception as e:
        logger.error(f"Error saving contact: {e}")
        session.rollback()
        return None

# OpenRouter batch query function
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(requests.exceptions.HTTPError)
)
def query_openrouter_batch(prompts):
    """Query OpenRouter API with batched prompts."""
    if not prompts:
        return []

    responses = []
    for prompt in prompts:
        payload = {
            "model": "cognitivecomputations/dolphin3.0-r1-mistral-24b:free",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Provide concise responses in JSON format."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 100,  # Adjusted for email tasks
            "temperature": 0.1,  # Low for structured output
            "top_p": 0.95
        }
        try:
            response = requests.post(OPENROUTER_API_URL, headers=HEADERS, json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            responses.append({"generated_text": content})
        except Exception as e:
            logger.error(f"Error querying OpenRouter: {str(e)}")
            responses.append(None)
    return responses

# Updated classify_email
def classify_email(subject, body):
    """Classify an email as hotel-related or not using OpenRouter."""
    prompt = (
        f"Analyze this email and determine if it's related to hotels, lodges, or camps.\n\n"
        f"Subject: {subject}\n"
        f"Body: {body}\n\n"
        "Return ONLY a JSON object with one key 'classification' and value either 'Hotel' or 'Other'."
    )
    try:
        result = query_openrouter_batch([prompt])
        if result and result[0]:
            generated_text = result[0].get("generated_text", "").strip()
            try:
                output = json.loads(generated_text)
                classification = output.get("classification", "Other").strip()
                return classification
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON from OpenRouter: {generated_text}")
                return "Other"
        else:
            logger.warning("No valid response from OpenRouter")
            return "Other"
    except Exception as e:
        logger.error(f"Error classifying email with OpenRouter: {e}")
        return "Other"

# Updated extract_hotel_name
def extract_hotel_name(subject, body):
    """Extract hotel names from email content using OpenRouter."""
    prompt = (
        f"Extract the name of any hotel, lodge, or camp mentioned in this email.\n\n"
        f"Subject: {subject}\n"
        f"Body: {body}\n\n"
        "Return ONLY a JSON object with one key 'hotel_name'. If none found, use empty string."
    )
    try:
        result = query_openrouter_batch([prompt])
        if result and result[0]:
            generated_text = result[0].get("generated_text", "").strip()
            try:
                output = json.loads(generated_text)
                hotel_name = output.get("hotel_name", "").strip()
                return hotel_name
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON from OpenRouter: {generated_text}")
                return ""
        else:
            logger.warning("No valid response from OpenRouter")
            return ""
    except Exception as e:
        logger.error(f"Error extracting hotel name with OpenRouter: {e}")
        return ""

# Existing Google Places function (unchanged)
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_hotel_details_from_google(hotel_name):
    if not hotel_name:
        logger.warning("Empty hotel name provided")
        return {}
    try:
        query = f"{hotel_name} hotel kenya" if "hotel" not in hotel_name.lower() else f"{hotel_name} kenya"
        logger.debug(f"Querying Google Places API with: {query}")
        url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
        params = {
            'query': query,
            'key': config.GOOGLE_PLACES_API_KEY
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        results = response.json()
        logger.debug(f"Google Places API response: {results}")

        # Check if the response contains results
        if results.get('status') != 'OK' or not results.get('results'):
            logger.warning(f"No results found for hotel: {hotel_name}")
            return {}

        # Log the first result
        place = results['results'][0]
        logger.debug(f"First result: {place}")

        # Get place details
        place_id = place['place_id']
        details_url = "https://maps.googleapis.com/maps/api/place/details/json"
        details_params = {
            'place_id': place_id,
            'fields': 'name,website,formatted_address,geometry,formatted_phone_number',
            'key': config.GOOGLE_PLACES_API_KEY
        }
        details_response = requests.get(details_url, params=details_params, timeout=10)
        details_response.raise_for_status()
        details = details_response.json()['result']
        logger.debug(f"Place details: {details}")

        return {
            "Hotel Name": details.get('name', hotel_name),
            "Website": details.get('website', ''),
            "Address": details.get('formatted_address', ''),
            "Coordinates": f"{details['geometry']['location']['lat']},{details['geometry']['location']['lng']}",
            "Contact": details.get('formatted_phone_number', '')
        }
    except Exception as e:
        logger.error(f"Error fetching hotel details: {e}")
        return {"Hotel Name": hotel_name, "Website": "", "Address": "", "Coordinates": "", "Contact": ""}

# Existing helper functions (unchanged)
def extract_addresses_and_names(header_value):
    addresses = []
    if header_value:
        parsed = getaddresses([header_value])
        for name, email in parsed:
            if email:
                addresses.append((email, name.strip()))
    return addresses

def process_emails_in_batch(emails):
    """Process a batch of emails to classify and extract hotel-related information."""
    if not emails:
        return

    prompts = []
    for message in emails:
        subject = message.get("subject", "")
        body = ""
        if message.is_multipart():
            for part in message.walk():
                if part.get_content_type() == "text/plain":
                    try:
                        body = part.get_content()
                        break
                    except UnicodeDecodeError:
                        continue
        else:
            try:
                body = message.get_content()
            except UnicodeDecodeError:
                pass

        prompt = (
            "Analyze this email and determine if it's related to hotels, lodges, or camps.\n\n"
            f"Subject: {subject}\n"
            f"Body: {body}\n\n"
            "Return ONLY a JSON object with one key 'classification' "
            "and value either 'Hotel' or 'Other'."
        )
        prompts.append(prompt)

    results = query_openrouter_batch(prompts)

    with Session() as session:
        for message, result in zip(emails, results):
            try:
                classification = "Other"
                if result:
                    generated_text = result.get("generated_text", "").strip()
                    try:
                        output = json.loads(generated_text)
                        classification = output.get("classification", "Other").strip()
                    except json.JSONDecodeError:
                        classification = "Other"

                is_hotel = classification.lower() == "hotel"
                if is_hotel:
                    process_hotel_email(message)

                msg_id = message.get("Message-ID", "")
                save_processed_email(
                    session,
                    msg_id,
                    is_hotel=is_hotel
                )

            except Exception as e:
                msg_id = message.get("Message-ID", "")
                error_msg = f"Error processing email: {str(e)}"
                logger.error(error_msg)
                save_processed_email(
                    session,
                    msg_id,
                    is_hotel=False,
                    error=error_msg
                )

def process_hotel_email(message):
    try:
        contacts = {}
        subject = message.get("subject", "")
        date = message.get("date", "")
        content = ""
        if message.is_multipart():
            for part in message.walk():
                if part.get_content_type() == "text/plain":
                    content = part.get_content()
                    break
        else:
            content = message.get_content()
        extracted_name = extract_hotel_name(subject, content)
        logger.info(f"Processing email with subject '{subject}', extracted hotel name: '{extracted_name}'")
        headers = ["From", "To", "Cc", "Bcc"]
        all_addresses = []
        for header in headers:
            header_value = message.get(header, "")
            if header_value:
                all_addresses.extend(extract_addresses_and_names(header_value))
        with get_db_session() as session:
            for email, display_name in all_addresses:
                contact_data = {
                    "Email": email.lower(),
                    "Display Name": display_name,
                    "Subjects": {subject} if subject else set(),
                    "Dates": {date} if date else set(),
                    "Hotel Name": extracted_name,
                    "Website": "",
                    "Address": "",
                    "Coordinates": "",
                    "Contact": ""
                }
                if extracted_name:
                    details = get_hotel_details_from_google(extracted_name)
                    if details:
                        contact_data.update({
                            "Website": details.get("Website", ""),
                            "Address": details.get("Address", ""),
                            "Coordinates": details.get("Coordinates", ""),
                            "Contact": details.get("Contact", "")
                        })
                saved_contact = save_contact(session, contact_data)
                if saved_contact:
                    contacts[email] = contact_data
        logger.info(f"Processed hotel email with {len(contacts)} contacts")
        return contacts
    except Exception as e:
        logger.error(f"Error in process_hotel_email: {e}")
        return {}

def process_mbox(mbox_path):
    contacts = {}
    total_emails = get_total_emails(mbox_path)
    mbox = mailbox.mbox(mbox_path, factory=lambda f: BytesParser(policy=policy.default).parse(f))
    logging.info("Starting to process %s emails", total_emails)
    current_batch = []
    with tqdm(total=total_emails, desc="Processing emails") as pbar:
        try:
            with Session() as session:
                for message in mbox:
                    message_id = message.get("Message-ID", "")
                    if not message_id:
                        logging.warning("Email without Message-ID encountered")
                        pbar.update(1)
                        continue
                    current_batch.append(message)
                    if len(current_batch) >= config.BATCH_SIZE:
                        process_emails_in_batch(current_batch)
                        pbar.update(len(current_batch))
                        current_batch = []
                if current_batch:
                    process_emails_in_batch(current_batch)
                    pbar.update(len(current_batch))
        except Exception as e:
            logging.error(f"Error processing mbox: {e}")
            raise
        finally:
            mbox.close()
    return contacts

def write_csv(session, output_file, append=False):
    try:
        contacts = session.query(Contact).all()
        if not contacts:
            logger.warning("No contacts found to write to CSV")
            return
        mode = 'a' if append else 'w'
        with open(output_file, mode, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'Email', 'Display Name', 'Hotel Name', 'Website',
                'Address', 'Coordinates', 'Contact', 'Subjects', 'Dates'
            ])
            if not append:
                writer.writeheader()
            query = session.query(Contact).yield_per(100)
            for contact in query:
                writer.writerow({
                    'Email': contact.email,
                    'Display Name': contact.display_name,
                    'Hotel Name': contact.hotel_name,
                    'Website': contact.website,
                    'Address': contact.address,
                    'Coordinates': contact.coordinates,
                    'Contact': contact.contact,
                    'Subjects': contact.subjects,
                    'Dates': contact.dates
                })
        logger.info(f"Successfully wrote {len(contacts)} contacts to {output_file}")
    except Exception as e:
        logger.error(f"Error writing to CSV: {e}")

def process_mbox_in_chunks(mbox_path, output_csv, chunk_size=25):
    import tempfile
    import shutil
    import atexit
    temp_dir = tempfile.mkdtemp()
    atexit.register(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
    try:
        mbox = mailbox.mbox(mbox_path)
        total_messages = sum(1 for _ in mbox)
        logger.info(f"Starting to process {total_messages} emails")
        mbox.close()  # Close initial count instance
        mbox = mailbox.mbox(mbox_path)
        current_chunk = []
        chunk_number = 0
        try:
            for i, message in enumerate(mbox):
                current_chunk.append(message)
                if len(current_chunk) >= chunk_size or i == total_messages - 1:
                    chunk_path = os.path.join(temp_dir, f"chunk_{chunk_number}.mbox")
                    temp_mbox = mailbox.mbox(chunk_path)
                    try:
                        for msg in current_chunk:
                            temp_mbox.add(msg)
                        temp_mbox.close()
                        process_mbox(chunk_path)
                    finally:
                        if os.path.exists(chunk_path):
                            os.remove(chunk_path)
                    current_chunk = []
                    chunk_number += 1
        finally:
            mbox.close()
    except Exception as e:
        logger.error(f"Error processing mbox: {e}")
        raise
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def check_services():
    """Verify OpenRouter API is accessible."""
    try:
        payload = {
            "model": "cognitivecomputations/dolphin3.0-r1-mistral-24b:free",
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 5
        }
        response = requests.post(OPENROUTER_API_URL, headers=HEADERS, json=payload, timeout=10)
        response.raise_for_status()
        logger.info("✓ OpenRouter API is running and accessible")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"× OpenRouter error: {str(e)}")
        logger.error("Check: 1. Internet connection, 2. OPENROUTER_API_KEY in .env")
        return False

def main():
    """Main entry point for the email processor application."""
    logging.info("Starting email processing job")
    try:
        if not check_services():
            logging.error("OpenRouter unavailable. Check configuration.")
            return
        process_mbox_in_chunks(config.MBOX_FILE, config.OUTPUT_CSV)
        with Session() as session:
            write_csv(session, config.OUTPUT_CSV)
        logging.info("Email processing job completed successfully")
    except Exception as e:
        logging.error(f"Fatal error in main: {e}")
        raise

if __name__ == "__main__":
    main()