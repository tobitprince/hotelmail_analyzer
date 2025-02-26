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
from decouple import config as decouple_config
from tqdm import tqdm
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
from bs4 import BeautifulSoup
from config import AppConfig
from log_config import setup_logging

# def setup_logging():
#     logging.basicConfig(level=logging.DEBUG)  # Enable DEBUG logging
#     return logging.getLogger(__name__)

logger = setup_logging()

# OpenRouter API setup
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {
    "Content-Type": "application/json",
    "X-Title": "Email Processor for Hotel Related Communications",
}

# Database setup
Base = declarative_base()
engine = create_engine('sqlite:///openrouter_email_processing.db',
                      poolclass=QueuePool,
                      pool_size=5,
                      max_overflow=10,
                      pool_timeout=30)
Session = sessionmaker(bind=engine)

@contextmanager
def get_db_session():
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
    subjects = Column(Text)
    dates = Column(Text)

Base.metadata.create_all(engine)

### Utility Functions
def extract_emails_from_body(body):
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.findall(email_pattern, body)

def determine_hotel_email(hotel_name, emails, website):
    hotel_words = [word.lower() for word in hotel_name.split() if word.lower() not in ['hotel', 'resort', 'inn', 'lodge']]
    for email in emails:
        domain = email.split('@')[1].lower()
        if any(word in domain for word in hotel_words):
            logger.info(f"Found hotel email in body: {email}")
            return email
    if website:
        try:
            response = requests.get(website, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            page_text = soup.get_text()
            page_emails = extract_emails_from_body(page_text)
            for email in page_emails:
                domain = email.split('@')[1].lower()
                if any(word in domain for word in hotel_words):
                    logger.info(f"Found hotel email on website: {email}")
                    return email
            if page_emails:
                logger.info(f"No matching email found; using first email from website: {page_emails[0]}")
                return page_emails[0]
        except Exception as e:
            logger.error(f"Error scraping website {website}: {e}")
    logger.warning(f"No hotel email found for {hotel_name}")
    return None

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
        logger.error(f"Database error checking if email is processed: {e}")
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
        email = contact_data.get("Email", "").lower().strip()
        if not email:
            logger.error("Attempted to save contact without email address")
            return None
        display_name = contact_data.get("Display Name", "").strip() or email.split('@')[0].replace('.', ' ').title()
        subjects = list(contact_data.get("Subjects", set()))
        dates = list(contact_data.get("Dates", set()))
        contact = session.query(Contact).filter_by(email=email).first()
        if not contact:
            contact = Contact(
                email=email,
                display_name=display_name,
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
            if contact_data.get("Hotel Name"):
                contact.hotel_name = contact_data["Hotel Name"]
                contact.website = contact_data.get("Website") or contact.website
                contact.address = contact_data.get("Address") or contact.address
                contact.coordinates = contact_data.get("Coordinates") or contact.coordinates
                contact.contact = contact_data.get("Contact") or contact.contact
            existing_subjects = set(json.loads(contact.subjects)) if contact.subjects else set()
            existing_dates = set(json.loads(contact.dates)) if contact.dates else set()
            contact.subjects = json.dumps(list(existing_subjects | set(subjects)))
            contact.dates = json.dumps(list(existing_dates | set(dates)))
        session.commit()
        logger.info(f"Successfully saved/updated contact: {email}")
        return contact
    except Exception as e:
        logger.error(f"Error saving contact: {e}")
        session.rollback()
        return None

### OpenRouter API Functions
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), retry=retry_if_exception_type(requests.exceptions.HTTPError))
def query_openrouter_batch(prompts, config):
    if not prompts:
        return []
    responses = []
    headers = HEADERS.copy()
    headers["Authorization"] = f"Bearer {config.OPENROUTER_API_KEY}"
    for prompt in prompts:
        payload = {
            "model": config.OPENROUTER_MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Provide concise responses in JSON format only, no extra text."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 300,
            "temperature": 0.5,
            "top_p": 0.95
        }
        try:
            response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()
            logger.debug(f"Full OpenRouter response: {result}")
            if "choices" in result and result["choices"]:
                content = result["choices"][0]["message"]["content"]
                responses.append({"generated_text": content})
            else:
                logger.error(f"Unexpected response structure: {result}")
                responses.append(None)
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {str(e)}")
            responses.append(None)
        except Exception as e:
            logger.error(f"Error processing response: {str(e)}")
            responses.append(None)
    return responses

def classify_email(subject, body, config):
    logger.info(f"Entering classify_email with subject: '{subject}'")
    prompt = (
        "Classify this email as 'Hotel' if it discusses hotel bookings, offers, events, "
        "or mentions hotels in a business context; otherwise, 'Other.' "
        "Return ONLY a JSON object like {\"classification\": \"Hotel\"} or {\"classification\": \"Other\"}. "
        "No extra text or formatting.\n\n"
        "Examples:\n"
        "- Subject: 'Booking Confirmation - Sarova Stanley', Body: 'Your room is confirmed...' → {\"classification\": \"Hotel\"}\n"
        "- Subject: 'Your Flight Itinerary', Body: 'Attached is your flight...' → {\"classification\": \"Other\"}\n"
        "- Subject: 'Shukran Invitation to Exhibit at the Third Regional Hospitality & Tourism Leadership Summit 2025', "
        "Body: 'This prestigious event will take place from 12th to 14th March 2025 at Hotel Sapphire...' → {\"classification\": \"Hotel\"}\n\n"
        f"Subject: {subject}\n"
        f"Body: {body}"
    )
    try:
        result = query_openrouter_batch([prompt], config)
        if result and result[0]:
            generated_text = result[0].get("generated_text", "").strip()
            logger.info(f"OpenRouter raw response for subject '{subject}': '{generated_text}'")
            cleaned_text = re.sub(r'```json\s*|\s*```', '', generated_text).strip()
            logger.info(f"Cleaned response for subject '{subject}': '{cleaned_text}'")
            if not cleaned_text:
                logger.warning(f"Empty response from OpenRouter for subject '{subject}'")
                return "Other"
            try:
                output = json.loads(cleaned_text)
                classification = output.get("classification", "Other").strip()
                logger.info(f"Parsed classification for subject '{subject}': '{classification}'")
                return classification
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON from OpenRouter for subject '{subject}': '{cleaned_text}'")
                return "Other"
        else:
            logger.warning(f"No valid response from OpenRouter for subject '{subject}'")
            return "Other"
    except Exception as e:
        logger.error(f"Error classifying email with OpenRouter for subject '{subject}': {e}")
        return "Other"

def extract_hotel_name(subject, body, config):
    prompt = (
        f"Extract the name of any hotel, lodge, or camp mentioned in this email. "
        "Return ONLY a JSON object with one key 'hotel_name', e.g., {\"hotel_name\": \"Sarova Stanley\"} or {\"hotel_name\": \"\"}. No extra text or formatting.\n\n"
        f"Subject: {subject}\n"
        f"Body: {body}"
    )
    try:
        result = query_openrouter_batch([prompt], config)
        if result and result[0]:
            generated_text = result[0].get("generated_text", "").strip()
            logger.info(f"OpenRouter raw response for hotel name extraction, subject '{subject}': '{generated_text}'")
            cleaned_text = re.sub(r'```json\s*|\s*```', '', generated_text).strip()
            logger.info(f"Cleaned response for hotel name extraction, subject '{subject}': '{cleaned_text}'")
            if not cleaned_text:
                logger.warning(f"Empty response from OpenRouter for hotel name extraction, subject '{subject}'")
                return ""
            try:
                output = json.loads(cleaned_text)
                hotel_name = output.get("hotel_name", "").strip()
                logger.info(f"Extracted hotel name for subject '{subject}': '{hotel_name}'")
                return hotel_name
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON from OpenRouter for subject '{subject}': '{cleaned_text}'")
                return ""
        else:
            logger.warning(f"No valid response from OpenRouter for subject '{subject}'")
            return ""
    except Exception as e:
        logger.error(f"Error extracting hotel name with OpenRouter for subject '{subject}': {e}")
        return ""

### Google Places API Function
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_hotel_details_from_google(hotel_name, config):
    if not hotel_name:
        logger.warning("Empty hotel name provided")
        return {}
    try:
        query = f"{hotel_name} hotel kenya" if "hotel" not in hotel_name.lower() else f"{hotel_name} kenya"
        logger.debug(f"Querying Google Places API with: {query}")
        url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
        api_key = config.GOOGLE_PLACES_API_KEY
        logger.debug(f"Using API key: {api_key[:5]}...")
        params = {'query': query, 'key': api_key}
        logger.debug(f"Sending request to {url} with params: {params}")
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        results = response.json()
        logger.debug(f"Google Places API response: {results}")
        if results.get('status') == 'REQUEST_DENIED':
            logger.error(f"Google Places API request denied: {results.get('error_message')}")
            return {}
        if results.get('status') != 'OK' or not results.get('results'):
            logger.warning(f"No results found for hotel: {hotel_name}, status: {results.get('status')}")
            return {}
        place = results['results'][0]
        if 'place_id' not in place:
            logger.info(f"No place_id found for {hotel_name}, using basic data")
            return {
                "Hotel Name": place.get('name', hotel_name),
                "Website": "",
                "Address": place.get('formatted_address', ''),
                "Coordinates": f"{place['geometry']['location']['lat']},{place['geometry']['location']['lng']}",
                "Contact": ""
            }
        place_id = place['place_id']
        details_url = "https://maps.googleapis.com/maps/api/place/details/json"
        details_params = {
            'place_id': place_id,
            'fields': 'name,website,formatted_address,geometry,formatted_phone_number',
            'key': api_key
        }
        logger.debug(f"Fetching details with place_id: {place_id}")
        details_response = requests.get(details_url, params=details_params, timeout=30)
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
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {str(e)}")
        return {"Hotel Name": hotel_name, "Website": "", "Address": "", "Coordinates": "", "Contact": ""}
    except Exception as e:
        logger.error(f"Unexpected error fetching hotel details: {e}")
        return {"Hotel Name": hotel_name, "Website": "", "Address": "", "Coordinates": "", "Contact": ""}

### Email Processing Functions
def process_emails_in_batch(emails, config):
    if not emails:
        return
    with Session() as session:
        for message in emails:
            try:
                subject = message.get("subject", "")
                body = ""
                if message.is_multipart():
                    for part in message.walk():
                        content_type = part.get_content_type()
                        if content_type == "text/plain":
                            try:
                                body += part.get_payload(decode=True).decode('utf-8', errors='ignore')
                            except UnicodeDecodeError:
                                continue
                        elif content_type == "text/html":
                            html = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                            soup = BeautifulSoup(html, 'html.parser')
                            body += soup.get_text()
                else:
                    try:
                        body = message.get_payload(decode=True).decode('utf-8', errors='ignore')
                    except UnicodeDecodeError:
                        pass
                classification = classify_email(subject, body, config)
                logger.info(f"Classified email - Subject: '{subject}', Classification: '{classification}'")
                is_hotel = classification.lower() == "hotel"
                if is_hotel:
                    process_hotel_email(message, config)
                msg_id = message.get("Message-ID", "")
                save_processed_email(session, msg_id, is_hotel=is_hotel)
            except Exception as e:
                msg_id = message.get("Message-ID", "")
                error_msg = f"Error processing email: {str(e)}"
                logger.error(error_msg)
                save_processed_email(session, msg_id, is_hotel=False, error=error_msg)

def process_hotel_email(message, config):
    try:
        subject = message.get("subject", "").strip()
        date = message.get("date", "").strip()
        content = ""
        if message.is_multipart():
            for part in message.walk():
                content_type = part.get_content_type()
                if content_type == "text/plain":
                    content += part.get_payload(decode=True).decode('utf-8', errors='ignore')
                elif content_type == "text/html":
                    html = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                    soup = BeautifulSoup(html, 'html.parser')
                    content += soup.get_text()
        else:
            content = message.get_payload(decode=True).decode('utf-8', errors='ignore')
        extracted_name = extract_hotel_name(subject, content, config)
        logger.info(f"Processing email with subject '{subject}', extracted hotel name: '{extracted_name}'")
        hotel_details = {}
        if extracted_name:
            logger.debug(f"Calling Google Places API for: {extracted_name}")
            hotel_details = get_hotel_details_from_google(extracted_name, config)
            logger.debug(f"Retrieved hotel details: {hotel_details}")
        body_emails = extract_emails_from_body(content)
        logger.info(f"Emails extracted from body: {body_emails}")
        hotel_email = determine_hotel_email(extracted_name, body_emails, hotel_details.get("Website"))
        if hotel_email:
            contact_data = {
                "Email": hotel_email.lower(),
                "Display Name": extracted_name,
                "Hotel Name": hotel_details.get("Hotel Name", extracted_name),
                "Website": hotel_details.get("Website", ""),
                "Address": hotel_details.get("Address", ""),
                "Coordinates": hotel_details.get("Coordinates", ""),
                "Contact": hotel_details.get("Contact", ""),
                "Subjects": {subject} if subject else set(),
                "Dates": {date} if date else set()
            }
            logger.debug(f"Contact data to save: {contact_data}")
            with get_db_session() as session:
                saved_contact = save_contact(session, contact_data)
                if saved_contact:
                    logger.info(f"Saved hotel contact: {hotel_email}")
                    return {hotel_email: contact_data}
        else:
            logger.warning(f"No hotel email found for {extracted_name}")
            return {}
    except Exception as e:
        logger.error(f"Error in process_hotel_email: {e}")
        return {}

def process_mbox(mbox_path, config):
    contacts = {}
    total_emails = get_total_emails(mbox_path)
    mbox = mailbox.mbox(mbox_path, factory=lambda f: BytesParser(policy=policy.default).parse(f))
    logger.info(f"Starting to process {total_emails} emails")
    current_batch = []
    with tqdm(total=total_emails, desc="Processing emails") as pbar:
        try:
            with Session() as session:
                for message in mbox:
                    message_id = message.get("Message-ID", "")
                    if not message_id:
                        logger.warning("Email without Message-ID encountered")
                        pbar.update(1)
                        continue
                    current_batch.append(message)
                    if len(current_batch) >= config.BATCH_SIZE:
                        process_emails_in_batch(current_batch, config)
                        pbar.update(len(current_batch))
                        current_batch = []
                if current_batch:
                    process_emails_in_batch(current_batch, config)
                    pbar.update(len(current_batch))
        except Exception as e:
            logger.error(f"Error processing mbox: {e}")
            raise
        finally:
            mbox.close()
    return contacts

def write_csv(session, output_file, append=False):
    try:
        contacts = session.query(Contact).all()
        if not contacts:
            logger.warning("No contacts to write to CSV")
            return
        fieldnames = ['Email', 'Display Name', 'Hotel Name', 'Website', 'Address', 'Coordinates', 'Contact', 'Subjects', 'Dates']
        mode = 'a' if append else 'w'
        with open(output_file, mode, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not append:
                writer.writeheader()
            for contact in contacts:
                row = {
                    'Email': contact.email,
                    'Display Name': contact.display_name,
                    'Hotel Name': contact.hotel_name,
                    'Website': contact.website or '',
                    'Address': contact.address or '',
                    'Coordinates': contact.coordinates or '',
                    'Contact': contact.contact or '',
                    'Subjects': contact.subjects,
                    'Dates': contact.dates
                }
                writer.writerow(row)
        logger.info(f"Successfully wrote {len(contacts)} contacts to {output_file}")
    except Exception as e:
        logger.error(f"Error writing to CSV: {e}")

def process_mbox_in_chunks(mbox_path, output_csv, config, chunk_size=25):
    import tempfile
    import shutil
    import atexit
    temp_dir = tempfile.mkdtemp()
    atexit.register(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
    try:
        mbox = mailbox.mbox(mbox_path)
        total_messages = sum(1 for _ in mbox)
        logger.info(f"Starting to process {total_messages} emails")
        mbox.close()
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
                        process_mbox(chunk_path, config)
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

### Service Check and Main Function
def check_services(config):
    try:
        payload = {"model": f"{config.OPENROUTER_MODEL}", "messages": [{"role": "user", "content": "test"}], "max_tokens": 5}
        headers = HEADERS.copy()
        headers["Authorization"] = f"Bearer {config.OPENROUTER_API_KEY}"
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        logger.info("✓ OpenRouter API is running and accessible")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"× OpenRouter error: {str(e)}")
        logger.error("Check: 1. Internet connection, 2. OPENROUTER_API_KEY in .env")
        return False

def main():
    logger.info("Starting email processing job")
    config = AppConfig()
    logger.debug(f"Loaded GOOGLE_PLACES_API_KEY in main: {config.GOOGLE_PLACES_API_KEY[:5] if config.GOOGLE_PLACES_API_KEY else 'None'}...")
    try:
        if not check_services(config):
            logger.error("OpenRouter unavailable. Check configuration.")
            return
        if not config.GOOGLE_PLACES_API_KEY:
            logger.error("GOOGLE_PLACES_API_KEY is missing.")
            return
        if not os.path.exists(config.MBOX_FILE):
            logger.error(f"MBOX_FILE not found at {config.MBOX_FILE}.")
            return
        process_mbox_in_chunks(config.MBOX_FILE, config.OPENROUTER_OUTPUT_CSV, config)
        with Session() as session:
            write_csv(session, config.OPENROUTER_OUTPUT_CSV)
        logger.info("Email processing job completed successfully")
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        raise

if __name__ == "__main__":
    main()