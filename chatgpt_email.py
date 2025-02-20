"""
Email Processor for Hotel-Related Communications

This module processes MBOX email files to identify and extract information about hotel-related
communications. It uses LM Studio with Mistral-7B for email classification and the Google Places
API for fetching additional hotel details.

Key Features:
- Processes large MBOX files in manageable chunks
- Uses AI for intelligent email classification
- Extracts hotel names and contact information
- Integrates with Google Places API
- Maintains a SQLite database
- Exports results to CSV
- Supports batch processing for GPU efficiency

Dependencies:
- LM Studio running locally with Mistral-7B model
- Google Places API key
- Python packages: sqlalchemy, requests, python-decouple, tqdm, etc.

Environment Variables:
- MBOX_FILE: Path to the MBOX file
- OUTPUT_CSV: Path for the output CSV file
- GOOGLE_PLACES_API_KEY: Google Places API key
- LM_STUDIO_API_URL: URL for LM Studio API
- LM_STUDIO_MODEL: Name of the LM Studio model to use
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
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, scoped_session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import insert
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager

from config import AppConfig
from log_config import setup_logging



# Initialize configuration and logging
setup_logging()
logger = logging.getLogger(__name__)
config = AppConfig()

# Database setup
Base = declarative_base()
engine = create_engine('sqlite:///email_processing.db',
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
    """
    SQLAlchemy model for tracking processed emails.

    Stores information about each processed email including its processing status
    and any errors encountered during processing.

    Attributes:
        id: Unique identifier for the record
        message_id: Email's Message-ID header (unique)
        date_processed: Timestamp of when the email was processed
        is_hotel_related: Whether the email was classified as hotel-related
        error: Any error message encountered during processing
    """
    __tablename__ = 'processed_emails'
    id = Column(Integer, primary_key=True)
    message_id = Column(String, unique=True)
    date_processed = Column(DateTime, default=datetime.utcnow)
    is_hotel_related = Column(Boolean, default=False)
    error = Column(Text, nullable=True)

class Contact(Base):
    """
    SQLAlchemy model for storing contact information extracted from emails.

    Maintains a record of contacts and their associated hotel information,
    including details fetched from Google Places API.

    Attributes:
        id: Unique identifier for the contact
        email: Contact's email address (unique)
        display_name: Contact's display name from email
        hotel_name: Associated hotel name
        website: Hotel's website
        address: Hotel's physical address
        coordinates: Hotel's geographical coordinates
        contact: Contact information (phone, etc.)
        subjects: JSON array of email subjects
        dates: JSON array of communication dates
    """
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

# Create database tables
Base.metadata.create_all(engine)




def get_total_emails(mbox_path):
    """
    Count the total number of emails in the mbox file.

    Args:
        mbox_path (str): Path to the MBOX file

    Returns:
        int: Total number of emails in the file, or 0 if file is invalid

    This function performs basic validation of the MBOX file and provides
    progress updates during counting for large files.
    """
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
        try:
            for key in mbox.keys():
                count += 1
                if count % 100 == 0:  # Progress update every 100 emails
                    print(f"Counted {count} messages...")
        finally:
            mbox.close()

        print(f"Found {count} total messages")
        return count
    except FileNotFoundError:
        print(f"Error: Could not open mbox file at {mbox_path}")
        return 0
    except Exception as e:
        print(f"Error counting emails: {str(e)}")
        return 0

def is_email_processed(session, message_id):
    """
    Check if an email has already been processed by looking up its Message-ID.

    Args:
        session: SQLAlchemy session
        message_id (str): Email's Message-ID header

    Returns:
        bool: True if email has been processed, False otherwise
    """
    if not message_id:
        return False
    message_id = message_id.strip()  # Remove leading/trailing spaces
    try:
        return session.query(ProcessedEmail).filter(ProcessedEmail.message_id == message_id).first() is not None
    except SQLAlchemyError as e:
        logging.error(f"Database error checking if email is processed: {e}")
        return False

def save_processed_email(session, message_id, is_hotel=False, error=None):
    """
    Record a processed email in the database with its classification and any errors.

    Args:
        session: SQLAlchemy session
        message_id (str): Email's Message-ID header
        is_hotel (bool): Whether email was classified as hotel-related
        error (str, optional): Error message if processing failed

    Handles database transaction and rollback on error.
    """
    try:
        # First try to update existing record
        existing = session.query(ProcessedEmail).filter(
            ProcessedEmail.message_id == message_id.strip()
        ).first()

        if existing:
            existing.date_processed = datetime.now()
            existing.is_hotel_related = is_hotel
            existing.error = error
        else:
            # Create new record if doesn't exist
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
    """
    Save or update contact information in the database.

    Args:
        session: SQLAlchemy session
        contact_data (dict): Dictionary containing contact information including:
            - Email: Contact's email address
            - Display Name: Contact's name
            - Hotel Name: Associated hotel (if any)
            - Website: Hotel website
            - Address: Physical address
            - Coordinates: Geographical coordinates
            - Contact: Contact information
            - Subjects: Set of email subjects
            - Dates: Set of communication dates

    Returns:
        Contact: The saved or updated Contact object
    """
    try:
        email = contact_data.get("Email")
        if not email:
            logger.error("Attempted to save contact without email address")
            return None

        # Convert sets to lists for JSON serialization
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
            # Update existing contact
            existing_subjects = set(json.loads(contact.subjects)) if contact.subjects else set()
            existing_dates = set(json.loads(contact.dates)) if contact.dates else set()

            # Update subjects and dates
            updated_subjects = list(existing_subjects | set(subjects))
            updated_dates = list(existing_dates | set(dates))

            contact.subjects = json.dumps(updated_subjects)
            contact.dates = json.dumps(updated_dates)

            # Update other fields if not already present
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

def check_services():
    """
    Verify that required services (LM Studio) are running and accessible.

    Tests LM Studio connectivity by attempting a simple completion request.
    Provides detailed error messages and setup instructions if services
    are not properly configured.

    Returns:
        bool: True if all services are running, False otherwise
    """
    try:
        response = requests.post(
            config.LM_STUDIO_API_URL,
            json={
                "model": config.LM_STUDIO_MODEL,
                "prompt": "test",
                "max_tokens": 5
            },
            timeout=10
        )
        response.raise_for_status()
        logging.info("LM Studio is running and accessible")
        return True
    except requests.exceptions.RequestException as e:
        logging.error(f"LM Studio error: {str(e)}")
        return False

def query_lm_studio_batch(prompts):
    """
    Query LM Studio with multiple prompts for GPU-efficient processing.

    Args:
        prompts (list): List of prompt strings to process

    Returns:
        list: List of response dictionaries, one for each prompt

    Features:
    - Batches prompts for optimal GPU utilization
    - Configurable batch size and GPU memory allocation
    - Handles timeouts and connection errors
    - Returns None for failed prompts while continuing with others
    """
    if not prompts:
        return []

    # Split prompts into batches
    batches = [prompts[i:i + config.BATCH_SIZE] for i in range(0, len(prompts), config.BATCH_SIZE)]
    all_results = []

    for batch in batches:
        payload = {
            "model": config.LM_STUDIO_MODEL,
            "prompt": batch,
            "max_tokens": 1500,
            "temperature": 0.0,
            "stop": ["\n", "\""],  # Stop at newlines or quotes for clean JSON
            "stream": False,
            "batch_size": len(batch),
            "gpu_memory": config.GPU_MEMORY
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(config.LM_STUDIO_API_URL, json=payload, timeout=config.LM_STUDIO_TIMEOUT)  # Increased timeout for batch
                response.raise_for_status()
                result = response.json()
                logging.info(f"Received response from LM Studio: {result}")  # Log the response

                if "choices" in result:
                    all_results.extend([{"text": choice["text"]} for choice in result["choices"]])
                else:
                    all_results.extend([None] * len(batch))
                    logging.error("Unexpected response format from LM Studio")
                break  # Exit retry loop if successful
            except requests.exceptions.RequestException as e:  # Catch broader exception
                logging.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    logging.error(f"Failed after {max_retries} attempts.")
                    all_results.extend([None] * len(batch))  # Mark as failed
                else:
                    time.sleep(2 ** attempt)  # Exponential backoff

    return all_results

# Improve retry mechanism
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((requests.exceptions.Timeout, requests.exceptions.ConnectionError))
)
def query_lm_studio(prompt):
    """
    Query LM Studio with a single prompt.

    Args:
        prompt (str): The prompt to send to LM Studio

    Returns:
        dict: Response from LM Studio, or None if query fails

    This is a convenience wrapper around query_lm_studio_batch for
    single prompt scenarios. Uses the same batched infrastructure
    for consistency.
    """
    results = query_lm_studio_batch([prompt])
    return results[0] if results else None


def classify_email(subject, body):
    """
    Classify an email as hotel-related or not using LM Studio.

    Args:
        subject (str): Email subject line
        body (str): Email body text

    Returns:
        str: "Hotel" if email is related to hotels/lodges/camps, "Other" otherwise

    Uses a carefully crafted prompt to instruct the LM Studio model to analyze
    the email content and determine if it's related to hospitality businesses.
    Handles JSON parsing of the model's response and provides a default
    classification of "Other" if analysis fails.
    """
    try:
        prompt = f"""[INST]Classify this email as hotel-related or not.

        Subject: {subject}
        Body: {body}

        STRICT CLASSIFICATION RULES:
        1. HOTEL if:
           - Direct hotel bookings or confirmations
           - Hotel marketing materials
           - Lodge or resort communications
           - Hotel service feedback

        2. NOT HOTEL if:
           - Airlines or flights
           - General business emails
           - Shopping or retail
           - Tech or software
           - Finance or banking
           - Event invitations (unless at hotels)
           - General marketing

        Examples:
        "Booking confirmation - Sarova Stanley" → HOTEL
        "Your flight itinerary" → NOT HOTEL
        "Safari package with accommodation" → HOTEL
        "Payment received" → NOT HOTEL

        Respond with EXACTLY one word - either "HOTEL" or "NOT_HOTEL"
        [/INST]"""

        response = query_lm_studio(prompt)
        if not response or 'choices' not in response:
            logger.warning("No valid response from LM Studio")
            return False

        # Get the classification text
        classification = response['choices'][0]['text'].strip().upper()
        # Log classification details
        logger.debug(f"""
        Classification Details:
        Subject: {subject[:100]}...
        Result: {classification}
        Raw Response: {response}
        """)

        # Track classification distribution
        if not hasattr(classify_email, 'stats'):
            classify_email.stats = {'HOTEL': 0, 'NOT_HOTEL': 0, 'INVALID': 0}

        if classification in ["HOTEL", "NOT_HOTEL"]:
            classify_email.stats[classification] += 1
        else:
            classify_email.stats['INVALID'] += 1

        # Alert if classification ratio is suspicious
        total = sum(classify_email.stats.values())
        if total > 100:  # Only check after processing enough emails
            hotel_ratio = classify_email.stats['HOTEL'] / total
            if hotel_ratio > 0.8:  # If more than 80% classified as hotels
                logger.warning(f"Suspicious classification ratio: {hotel_ratio:.2%} hotels")

        return classification == "HOTEL"

    except Exception as e:
        logger.error(f"Error classifying email: {e}")
        return False


def extract_hotel_name(subject, body):
    """
    Extract hotel names from email content using LM Studio.

    Args:
        subject (str): Email subject line
        body (str): Email body text

    Returns:
        str: Extracted hotel name, or empty string if none found

    Uses natural language processing to identify and extract hotel, lodge,
    or camp names from the email content. Handles various formats and
    contexts in which hotel names might appear.
    """
    try:
        # Clean and prepare text
        clean_subject = subject.strip() if subject else ""
        clean_body = body.strip() if body else ""

        prompt = f"""[INST]Extract the hotel, lodge, or camp name from this email.
        Subject: {clean_subject}
        Body: {clean_body}

        Rules:
        - Return ONLY the business name
        - Include full name (e.g., 'Sarova Stanley' not just 'Sarova')
        - If no hotel/lodge/camp found, return 'None'
        - Don't return partial matches
        - Don't include 'hotel', 'lodge', etc. unless part of official name

        Hotel name:[/INST]"""

        response = query_lm_studio(prompt)
        if not response or 'choices' not in response:
            logger.warning("No response from LM Studio")
            return ""

        hotel_name = response['choices'][0]['text'].strip()

        # Clean up response
        hotel_name = hotel_name.strip('"').strip("'").strip()

        # Validate response
        if hotel_name.lower() in ['none', 'not found', 'no hotel found', 'n/a', '']:
            logger.debug("No hotel name found")
            return ""

        logger.info(f"Extracted hotel name: '{hotel_name}' from subject: '{clean_subject}'")
        return hotel_name

    except Exception as e:
        logger.error(f"Error extracting hotel name: {e}")
        return ""

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_hotel_details_from_google(hotel_name):
    """
    Fetch detailed hotel information from Google Places API.
    """
    if not hotel_name:
        logger.warning("Empty hotel name provided")
        return {}

    try:
        # Format the search query - add "hotel" if not present
        query = f"{hotel_name} hotel kenya" if "hotel" not in hotel_name.lower() else f"{hotel_name} kenya"

        logger.debug(f"Querying Google Places API for: {query}")

        # Call Google Places API
        url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
        params = {
            'query': query,
            'key': config.GOOGLE_PLACES_API_KEY
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        results = response.json()

        # Log API response for debugging
        logger.debug(f"Google Places API response: {results}")

        if 'error_message' in results:
            logger.error(f"Google Places API error: {results['error_message']}")
            return {}

        if not results.get('results'):
            logger.warning(f"No results found for hotel: {hotel_name}")
            return {}

        place = results['results'][0]
        place_id = place['place_id']

        # Get detailed information
        details_url = "https://maps.googleapis.com/maps/api/place/details/json"
        details_params = {
            'place_id': place_id,
            'fields': 'name,website,formatted_address,geometry,formatted_phone_number',
            'key': config.GOOGLE_PLACES_API_KEY
        }

        details_response = requests.get(details_url, params=details_params, timeout=10)
        details_response.raise_for_status()
        details = details_response.json()

        if 'error_message' in details:
            logger.error(f"Google Places Details API error: {details['error_message']}")
            return {}

        if 'result' not in details:
            logger.warning(f"No details found for place_id: {place_id}")
            return {}

        details = details['result']

        hotel_details = {
            "Hotel Name": details.get('name', hotel_name),
            "Website": details.get('website', ''),
            "Address": details.get('formatted_address', ''),
            "Coordinates": f"{place['geometry']['location']['lat']},{place['geometry']['location']['lng']}",
            "Contact": details.get('formatted_phone_number', '')
        }

        logger.info(f"Successfully retrieved details for {hotel_name}: {hotel_details}")
        return hotel_details

    except requests.exceptions.RequestException as e:
        logger.error(f"Request error fetching hotel details for {hotel_name}: {e}")
    except KeyError as e:
        logger.error(f"Data parsing error for {hotel_name}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error fetching hotel details for {hotel_name}: {e}")

    return {
        "Hotel Name": hotel_name,
        "Website": "",
        "Address": "",
        "Coordinates": "",
        "Contact": ""
    }


def extract_addresses_and_names(header_value):
    """
    Parse email headers to extract addresses and display names.

    Args:
        header_value (str): Raw email header value (From, To, Cc, etc.)

    Returns:
        list: List of tuples (email, display_name)

    Uses email.utils.getaddresses to properly handle various email formats
    including quoted display names, multiple addresses, and special characters.
    """
    addresses = []
    if header_value:
        parsed = getaddresses([header_value])
        for name, email in parsed:
            if email:
                addresses.append((email, name.strip()))
    return addresses


def create_session():
    """
    Create a new thread-safe database session.

    Returns:
        scoped_session: A new SQLAlchemy scoped session

    Uses SQLAlchemy's scoped_session to ensure thread-safety when
    multiple parts of the application need database access.
    """
    return scoped_session(Session)

def process_emails_in_batch(emails):
    """
    Process multiple emails together for GPU-efficient classification.

    Args:
        emails (list): List of email message objects to process
        session: SQLAlchemy session for database operations

    Features:
    - Batches classification requests for GPU efficiency
    - Extracts email content from both multipart and simple messages
    - Handles errors individually per email
    - Records processing status in database
    - Processes hotel-related emails further for contact extraction
    """
    if not emails:
        return

    # Prepare batches of prompts for classification
    prompts = []
    for message in emails:
        subject = message.get("subject", "")

        # Extract body
        body = ""
        if message.is_multipart():
            for part in message.walk():
                if part.get_content_type() == "text/plain":
                    try:
                        body = part.get_content()
                        break
                    except Exception as e:
                        logging.error(f"Error extracting multipart content: {e}")
                        continue
        else:
            try:
                body = message.get_content()
            except Exception as e:
                logging.error(f"Error extracting content: {e}")

        prompt_text = (
            "<s>[INST]Analyze this email and determine if it's related to hotels, lodges, or camps.\n\n"
            f"Subject: {subject}\n"
            f"Body: {body}\n\n"
            "Return ONLY a JSON object with one key 'classification' and value either 'Hotel' or 'Other'.[/INST]"
            "{\n  \"classification\": \""
        )
        prompts.append(prompt_text)

    # Batch process with LM Studio
    results = query_lm_studio_batch(prompts)

    # Process results
    with Session() as session:
        for message, result in zip(emails, results):
            try:
                classification = "Other"
                if result:
                    # Complete the JSON from Mistral's response
                    text = result.get("text", "")
                    if text:
                        # Add closing quotes and brace
                        safe_text = text.replace('"', '\\"').strip()
                        json_str = '{"classification": "' + safe_text + '"}'
                        # Validate the JSON:
                        try:
                            output = json.loads(json_str)
                            classification = output.get("classification", "Other").strip()
                        except json.JSONDecodeError:
                            classification = "Other"

                is_hotel = classification.lower() == "hotel"
                if is_hotel:
                    process_hotel_email(message)

                save_processed_email(session, message.get("Message-ID", ""), is_hotel=is_hotel)
            except Exception as e:
                error_msg = f"Error processing email: {str(e)}"
                logging.error(error_msg)
                save_processed_email(session, message.get("Message-ID", ""), is_hotel=False, error=error_msg)

def process_hotel_email(message):
    """
    Extract and store information from a hotel-related email.

    Args:
        message: Email message object
        session: SQLAlchemy session

    Returns:
        dict: Dictionary of extracted contact information

    Processing steps:
    1. Extract all email addresses and names from headers
    2. Remove duplicate contacts
    3. Extract hotel name using LM Studio
    4. Fetch hotel details from Google Places API
    5. Save contact information to database
    """
    try:
        contacts = {}
        subject = message.get("subject", "")
        date = message.get("date", "")

        # Get message content
        content = ""
        if message.is_multipart():
            for part in message.walk():
                if part.get_content_type() == "text/plain":
                    content = part.get_content()
                    break
        else:
            content = message.get_content()

        # Extract hotel name first
        extracted_name = extract_hotel_name(subject, content)
        logger.info(f"Processing email with subject '{subject}', extracted hotel name: '{extracted_name}'")

        # Extract email addresses and display names from all headers
        headers = ["From", "To", "Cc", "Bcc"]
        all_addresses = []
        for header in headers:
            header_value = message.get(header, "")
            if header_value:
                all_addresses.extend(extract_addresses_and_names(header_value))

        # Process each unique contact
        with get_db_session() as session:
            for email, display_name in all_addresses:
                contact_data = {
                    "Email": email.lower(),
                    "Display Name": display_name,
                    "Subjects": {subject} if subject else set(),
                    "Dates": {date} if date else set(),
                    "Hotel Name": extracted_name,  # Set extracted name directly
                    "Website": "",
                    "Address": "",
                    "Coordinates": "",
                    "Contact": ""
                }

                # Only try Google Places if we have a hotel name
                if extracted_name:
                    try:
                        details = get_hotel_details_from_google(extracted_name)
                        if details:
                            contact_data.update({
                                "Website": details.get("Website", ""),
                                "Address": details.get("Address", ""),
                                "Coordinates": details.get("Coordinates", ""),
                                "Contact": details.get("Contact", "")
                            })
                    except Exception as e:
                        logger.error(f"Failed to get Google details for {extracted_name}: {e}")

                # Save contact data
                saved_contact = save_contact(session, contact_data)
                if saved_contact:
                    contacts[email] = contact_data

        logger.info(f"Processed hotel email with {len(contacts)} contacts")
        return contacts

    except Exception as e:
        logger.error(f"Error in process_hotel_email: {e}")
        return {}

def process_mbox(mbox_path):
    """
    Process an MBOX file to extract hotel-related information.

    Args:
        mbox_path (str): Path to the MBOX file
        session: SQLAlchemy session

    Returns:
        dict: Dictionary of processed contacts

    Features:
    - Counts total emails for progress tracking
    - Uses tqdm for progress display
    - Skips previously processed emails
    - Processes emails in batches for GPU efficiency
    - Handles errors gracefully and continues processing
    """
    contacts = {}
    total_emails = get_total_emails(mbox_path)
    mbox = mailbox.mbox(mbox_path, factory=lambda f: BytesParser(policy=policy.default).parse(f))

    logging.info("Starting to process %s emails", total_emails)

    # Process emails in batches
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

                    # if is_email_processed(session, message_id):
                    #     logging.info(f"Skipping already processed email: {message_id}")
                    #     pbar.update(1)
                    #     continue

                    # Add message to current batch
                    current_batch.append(message)

                    # Process batch when it reaches the batch size
                    if len(current_batch) >= config.BATCH_SIZE:
                        process_emails_in_batch(current_batch)
                        pbar.update(len(current_batch))
                        current_batch = []

                # Process remaining emails in the last batch
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
    """
    Export contact and hotel data to CSV format.

    Args:
        session: SQLAlchemy session
        output_file (str): Path to output CSV file
        append (bool): Whether to append to existing file

    CSV Structure:
    - Email: Contact's email address
    - Display Name: Contact's name
    - Hotel Name: Associated hotel
    - Website: Hotel website
    - Address: Physical address
    - Coordinates: Geographical coordinates
    - Contact: Contact information
    - Subjects: Semicolon-separated list of email subjects
    - Dates: Semicolon-separated list of communication dates

    Handles JSON decoding of stored sets and proper UTF-8 encoding.
    """
    try:
        # Query all contacts
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

            for contact in contacts:
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


def process_mbox_in_chunks(mbox_path, output_csv):
    """
    Process large MBOX files in memory-efficient chunks.

    Args:
        mbox_path (str): Path to the MBOX file
        output_csv (str): Path for the output CSV file

    Features:
    - Splits large files into manageable chunks
    - Processes each chunk independently
    - Maintains progress across chunks
    - Appends results to CSV incrementally
    - Cleans up temporary chunk files
    - Provides progress updates

    This is the main processing function that handles large MBOX files
    by breaking them into smaller pieces that can be processed without
    consuming too much memory.
    """
    try:
        chunk_size = config.CHUNK_SIZE_MB * 1024 * 1024  # Convert MB to bytes
        total_size = os.path.getsize(mbox_path)

        if total_size <= chunk_size:
            # Small enough to process directly
            session = Session()
            try:
                process_mbox(mbox_path)
                write_csv(session, output_csv)
            finally:
                session.close()
            return

        print(f"Processing {mbox_path} ({total_size/1024/1024:.2f} MB) in {config.CHUNK_SIZE_MB}MB chunks...")

        current_chunk = []
        current_size = 0
        chunk_number = 1

        with open(mbox_path, 'rb') as f:
            with tempfile.NamedTemporaryFile(mode='wb', delete=False) as chunk_file:  # Create temp file
                chunk_path = chunk_file.name
                for line in f:
                    if line.startswith(b'From ') and current_size > chunk_size:
                        # Process current chunk
                        print(f"\nProcessing chunk {chunk_number} ({current_size/1024/1024:.2f} MB)")
                        session = Session()
                        try:
                            process_mbox(chunk_path)
                            # Append results to main CSV
                            write_csv(session, output_csv, append=chunk_number > 1)
                        finally:
                            session.close()
                            os.remove(chunk_path)  # Clean up temp chunk

                        chunk_number += 1
                        current_size = 0
                        chunk_file = tempfile.NamedTemporaryFile(mode='wb', delete=False) # Create a new temp file
                        chunk_path = chunk_file.name

                    chunk_file.write(line)  # Write directly to the temp file
                    current_size += len(line)

                # Process final chunk if there's data
                if current_size > 0:
                    print(f"\nProcessing final chunk {chunk_number} ({current_size/1024/1024:.2f} MB)")
                    session = Session()
                    try:
                        with Session() as session:
                            process_mbox(chunk_path)
                            write_csv(session, output_csv, append=chunk_number > 1)
                    finally:
                        session.close()
                        os.remove(chunk_path)  # Clean up temp chunk

        print("\nProcessing complete!")
        print(f"Results written to: {output_csv}")

    except Exception as e:
        print(f"Error processing mbox: {str(e)}")
        logging.exception("Error processing mbox: %s", str(e))
        raise

def main():
    """
    Main entry point for the email processor application.

    Workflow:
    1. Set up logging
    2. Check required services (LM Studio)
    3. Process MBOX file in chunks
    4. Handle errors and cleanup

    The main function coordinates the overall processing flow and
    ensures proper error handling and resource cleanup.
    """
    logging.info("Starting email processing job")
    try:
        # Check if required services are running
        if not check_services():
            return

        # Process mbox file in chunks, writing results as we go
        process_mbox_in_chunks(config.MBOX_FILE, config.OUTPUT_CSV)

        logging.info("Email processing job completed successfully")
    except Exception as e:
        logging.error(f"Fatal error in main: {e}")
        raise


if __name__ == "__main__":
    main()
