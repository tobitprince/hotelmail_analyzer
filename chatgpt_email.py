import os
import mailbox
import re
import csv
import json
import requests
import logging
from datetime import datetime
from email import policy
from email.parser import BytesParser
from email.utils import getaddresses
from decouple import config
from tqdm import tqdm
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, scoped_session
from sqlalchemy.exc import SQLAlchemyError

# Constants for processing
CHUNK_SIZE_MB = 100  # Size of each chunk in MB
BATCH_SIZE = 4      # Number of emails to process in one GPU batch
GPU_MEMORY = 4096   # Memory to allocate for GPU in MB (4GB for Mistral)

# Set up logging
logging.basicConfig(
    filename='email_processor.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Database setup
Base = declarative_base()
engine = create_engine('sqlite:///email_processing.db')
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
    subjects = Column(Text)  # Stored as JSON
    dates = Column(Text)     # Stored as JSON

# Create database tables
Base.metadata.create_all(engine)

# Global constants loaded from the .env file using decouple
MBOX_FILE = config("MBOX_FILE", default="path/to/your_downloaded_file.mbox")
OUTPUT_CSV = config("OUTPUT_CSV", default="tipjar_customers.csv")
GOOGLE_PLACES_API_KEY = config("GOOGLE_PLACES_API_KEY", default="YOUR_GOOGLE_PLACES_API_KEY")
LM_STUDIO_API_URL = config("LM_STUDIO_API_URL", default="http://127.0.0.1:1234/v1/completions")
LM_STUDIO_MODEL = config("LM_STUDIO_MODEL", default="mistral-7b-instruct-v0.2")


def get_total_emails(mbox_path):
    """Count total number of emails in the mbox file."""
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
        for key in mbox.keys():
            count += 1
            if count % 100 == 0:  # Progress update every 100 emails
                print(f"Counted {count} messages...")
                
        print(f"Found {count} total messages")
        return count
    except FileNotFoundError:
        print(f"Error: Could not open mbox file at {mbox_path}")
        return 0
    except Exception as e:
        print(f"Error counting emails: {str(e)}")
        return 0

def is_email_processed(session, message_id):
    """Check if an email has already been processed."""
    return session.query(ProcessedEmail).filter_by(message_id=message_id).first() is not None

def save_processed_email(session, message_id, is_hotel=False, error=None):
    """Record a processed email in the database."""
    try:
        processed = ProcessedEmail(
            message_id=message_id,
            is_hotel_related=is_hotel,
            error=error
        )
        session.add(processed)
        session.commit()
    except SQLAlchemyError as e:
        logging.error(f"Database error saving processed email: {e}")
        session.rollback()

def save_contact(session, contact_data):
    """Save or update contact information in the database."""
    try:
        contact = session.query(Contact).filter_by(email=contact_data["Email"]).first()
        if not contact:
            contact = Contact(
                email=contact_data["Email"],
                display_name=contact_data["Display Name"],
                hotel_name=contact_data.get("Hotel Name", ""),
                website=contact_data.get("Website", ""),
                address=contact_data.get("Address", ""),
                coordinates=contact_data.get("Coordinates", ""),
                contact=contact_data.get("Contact", ""),
                subjects=json.dumps(list(contact_data["Subjects"])),
                dates=json.dumps(list(contact_data["Dates"]))
            )
            session.add(contact)
        else:
            # Update existing contact
            subjects = set(json.loads(contact.subjects)) if contact.subjects else set()
            dates = set(json.loads(contact.dates)) if contact.dates else set()
            subjects.update(contact_data["Subjects"])
            dates.update(contact_data["Dates"])
            
            contact.subjects = json.dumps(list(subjects))
            contact.dates = json.dumps(list(dates))
            if not contact.hotel_name and contact_data.get("Hotel Name"):
                contact.hotel_name = contact_data["Hotel Name"]
                contact.website = contact_data.get("Website", "")
                contact.address = contact_data.get("Address", "")
                contact.coordinates = contact_data.get("Coordinates", "")
                contact.contact = contact_data.get("Contact", "")
        
        session.commit()
    except SQLAlchemyError as e:
        logging.error(f"Database error saving contact: {e}")
        session.rollback()

def check_services():
    """Check if required services (LM Studio) are running."""
    try:
        # For LM Studio, we'll just try to get a simple completion
        test_payload = {
            "model": LM_STUDIO_MODEL,
            "prompt": "test",
            "max_tokens": 5,
            "temperature": 0
        }
        response = requests.post(LM_STUDIO_API_URL, json=test_payload, timeout=5)
        response.raise_for_status()
        logging.info("LM Studio is running and accessible")
        return True
    except requests.exceptions.ConnectionError:
        error_msg = f"\nError: Could not connect to LM Studio at {LM_STUDIO_API_URL}"
        logging.error(error_msg)
        print(error_msg)
        print("Please ensure:")
        print("1. LM Studio is running")
        print("2. The correct model is loaded:", LM_STUDIO_MODEL)
        print("3. API is enabled in LM Studio settings")
        print("4. The port (1234) is not blocked\n")
        return False
    except Exception as e:
        error_msg = f"\nError connecting to LM Studio: {str(e)}"
        logging.error(error_msg)
        print(error_msg)
        print("Please ensure LM Studio is properly configured and running.\n")
        return False

def query_lm_studio_batch(prompts):
    """
    Queries LM Studio with a batch of prompts for GPU efficiency.
    Returns a list of responses, one for each prompt.
    """
    if not prompts:
        return []
        
    # Split prompts into batches
    batches = [prompts[i:i + BATCH_SIZE] for i in range(0, len(prompts), BATCH_SIZE)]
    all_results = []
    
    for batch in batches:
        payload = {
            "model": LM_STUDIO_MODEL,
            "prompt": batch,
            "max_tokens": 1500,
            "temperature": 0.0,
            "stop": ["\n", "\""],  # Stop at newlines or quotes for clean JSON
            "stream": False,
            "batch_size": len(batch),
            "gpu_memory": GPU_MEMORY
        }
        
        try:
            response = requests.post(LM_STUDIO_API_URL, json=payload, timeout=30)  # Increased timeout for batch
            response.raise_for_status()
            result = response.json()
            
            if "choices" in result:
                all_results.extend([{"text": choice["text"]} for choice in result["choices"]])
            else:
                all_results.extend([None] * len(batch))
                logging.error("Unexpected response format from LM Studio")
        except requests.exceptions.ConnectionError:
            logging.error(f"Could not connect to LM Studio at {LM_STUDIO_API_URL}")
            all_results.extend([None] * len(batch))
        except requests.exceptions.Timeout:
            logging.error("LM Studio request timed out")
            all_results.extend([None] * len(batch))
        except Exception as e:
            logging.error(f"Error querying LM Studio: {e}")
            all_results.extend([None] * len(batch))
    
    return all_results

def query_lm_studio(prompt):
    """
    Single prompt version of LM Studio query.
    Uses the batch version internally for consistency.
    """
    results = query_lm_studio_batch([prompt])
    return results[0] if results else None


def classify_email(subject, body):
    """
    Uses LM Studio to determine if the email is related to hotels, lodges, 
    or camps (potential TipJAR customers by Shukran).

    Returns "Hotel" if the email is related; otherwise, returns "Other".
    """
    prompt_text = (
        "<s>[INST]Analyze this email and determine if it's related to hotels, lodges, or camps.\n\n"
        f"Subject: {subject}\n"
        f"Body: {body}\n\n"
        "Return ONLY a JSON object with one key 'classification' and value either 'Hotel' or 'Other'.[/INST]"
        "{\n  \"classification\": \""
    )
    result = query_lm_studio(prompt_text)
    if result:
        try:
            # Complete the JSON from Mistral's response
            text = result.get("text", "")
            if text:
                # Add closing quotes and brace
                json_str = '{"classification": "' + text.strip() + '"}'
                output = json.loads(json_str)
                classification = output.get("classification", "Other").strip()
                return classification
        except Exception as e:
            print(f"Error parsing LM Studio response for classification: {e}")
            return "Other"
    return "Other"


def extract_hotel_name(subject, body):
    """
    Uses LM Studio to extract the name of the hotel, lodge, or camp mentioned in the email.
    Returns the extracted name or an empty string if none is found.
    """
    prompt_text = (
        "<s>[INST]Extract the name of any hotel, lodge, or camp mentioned in this email.\n\n"
        f"Subject: {subject}\n"
        f"Body: {body}\n\n"
        "Return ONLY a JSON object with one key 'hotel_name'. If no hotel/lodge/camp is found, use empty string.[/INST]"
        "{\n  \"hotel_name\": \""
    )
    result = query_lm_studio(prompt_text)
    if result:
        try:
            # Complete the JSON from Mistral's response
            text = result.get("text", "")
            if text:
                # Add closing quotes and brace
                json_str = '{"hotel_name": "' + text.strip() + '"}'
                output = json.loads(json_str)
                hotel_name = output.get("hotel_name", "").strip()
                return hotel_name
        except Exception as e:
            print(f"Error parsing LM Studio response for hotel name extraction: {e}")
            return ""
    return ""


def get_hotel_details_from_google(hotel_name):
    """
    Uses the Google Places API to search for hotel details using the hotel name
    appended with 'Nairobi Kenya'. Returns a dictionary with keys:
    'Hotel Name', 'Website', 'Address', 'Coordinates', and 'Contact'.
    """
    query = f"{hotel_name} Nairobi Kenya"
    url = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
    params = {
        "input": query,
        "inputtype": "textquery",
        "fields": "formatted_address,geometry,website,name,formatted_phone_number",
        "key": GOOGLE_PLACES_API_KEY
    }
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if data.get("candidates"):
                candidate = data["candidates"][0]
                loc = candidate.get("geometry", {}).get("location", {})
                coordinates = f"{loc.get('lat', '')}, {loc.get('lng', '')}" if loc else ""
                details = {
                    "Hotel Name": candidate.get("name", ""),
                    "Website": candidate.get("website", ""),
                    "Address": candidate.get("formatted_address", ""),
                    "Coordinates": coordinates,
                    "Contact": candidate.get("formatted_phone_number", "")
                }
                return details
    except Exception as e:
        print(f"Error fetching hotel details from Google: {e}")
    # Return empty details if lookup fails
    return {"Hotel Name": "", "Website": "", "Address": "", "Coordinates": "", "Contact": ""}


def extract_addresses_and_names(header_value):
    """
    Extracts email addresses and display names from a header string.
    Returns a list of tuples: (email, display_name) using email.utils.getaddresses.
    """
    addresses = []
    if header_value:
        parsed = getaddresses([header_value])
        for name, email in parsed:
            if email:
                addresses.append((email, name.strip()))
    return addresses


def create_session():
    """Create a new scoped session for thread-safe database access."""
    return scoped_session(Session)

def process_emails_in_batch(emails, session):
    """Process a batch of emails together for GPU efficiency."""
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
    for message, result in zip(emails, results):
        try:
            classification = "Other"
            if result:
                # Complete the JSON from Mistral's response
                text = result.get("text", "")
                if text:
                    # Add closing quotes and brace
                    json_str = '{"classification": "' + text.strip() + '"}'
                    output = json.loads(json_str)
                    classification = output.get("classification", "Other").strip()
            
            is_hotel = classification.lower() == "hotel"
            if is_hotel:
                process_hotel_email(message, session)
            
            save_processed_email(session, message.get("Message-ID", ""), is_hotel=is_hotel)
        except Exception as e:
            error_msg = f"Error processing email: {str(e)}"
            logging.error(error_msg)
            save_processed_email(session, message.get("Message-ID", ""), is_hotel=False, error=error_msg)

def process_hotel_email(message, session):
    """Process an email that has been classified as hotel-related."""
    contacts = {}
    subject = message.get("subject", "")
    date = message.get("date", "")
    
    # Extract email addresses and display names from several headers.
    headers = ["From", "To", "Cc", "Bcc"]
    all_addresses = []
    for header in headers:
        header_value = message.get(header)
        if header_value:
            all_addresses.extend(extract_addresses_and_names(header_value))
    
    # Remove duplicate contacts based on email address.
    unique_addresses = {}
    for email, display_name in all_addresses:
        if email not in unique_addresses:
            unique_addresses[email] = display_name
        else:
            # If a display name is not already stored and this one is non-empty, update it.
            if not unique_addresses[email] and display_name:
                unique_addresses[email] = display_name
    
    # For each unique contact, update the aggregated information.
    for email, display_name in unique_addresses.items():
        contact_data = {
            "Email": email,
            "Display Name": display_name,
            "Subjects": {subject} if subject else set(),
            "Dates": {date} if date else set(),
            "Hotel Name": "",
            "Website": "",
            "Address": "",
            "Coordinates": "",
            "Contact": ""
        }
        
        # Extract hotel name and get details
        if subject or message.get_content():
            hotel_name = extract_hotel_name(subject, message.get_content())
            if hotel_name:
                details = get_hotel_details_from_google(hotel_name)
                contact_data["Hotel Name"] = details.get("Hotel Name", "")
                contact_data["Website"] = details.get("Website", "")
                contact_data["Address"] = details.get("Address", "")
                contact_data["Coordinates"] = details.get("Coordinates", "")
                contact_data["Contact"] = details.get("Contact", "")
        
        # Save contact data
        save_contact(session, contact_data)
        contacts[email] = contact_data
    
    return contacts

def process_mbox(mbox_path, session):
    """
    Processes the MBOX file and extracts data from emails that are classified as
    hospitality-related (i.e. hotels, lodges, or camps). Uses batched processing
    for better GPU utilization.
    """
    contacts = {}
    total_emails = get_total_emails(mbox_path)
    mbox = mailbox.mbox(mbox_path, factory=lambda f: BytesParser(policy=policy.default).parse(f))
    
    logging.info(f"Starting to process {total_emails} emails")
    
    # Process emails in batches
    current_batch = []
    
    with tqdm(total=total_emails, desc="Processing emails") as pbar:
        for message in mbox:
            message_id = message.get("Message-ID", "")
            if not message_id:
                logging.warning("Email without Message-ID encountered")
                pbar.update(1)
                continue
                
            if is_email_processed(session, message_id):
                logging.info(f"Skipping already processed email: {message_id}")
                pbar.update(1)
                continue
            
            # Add message to current batch
            current_batch.append(message)
            
            # Process batch when it reaches the batch size
            if len(current_batch) >= BATCH_SIZE:
                process_emails_in_batch(current_batch, session)
                pbar.update(len(current_batch))
                current_batch = []
        
        # Process remaining emails in the last batch
        if current_batch:
            process_emails_in_batch(current_batch, session)
            pbar.update(len(current_batch))
    return contacts


def write_csv(session, output_file, append=False):
    """
    Writes the aggregated contact and hotel data to a CSV file.
    The CSV contains the following columns:
    Email, Display Name, Hotel Name, Website, Address, Coordinates, Contact, Subjects, Dates.
    Sets (for subjects and dates) are converted into semicolon-separated strings.
    """
    mode = "a" if append else "w"  # Append or write mode
    logging.info(f"Writing results to CSV: {output_file}")
    with open(output_file, mode, newline="", encoding="utf-8") as csvfile:
        try:
            fieldnames = [
                "Email", "Display Name", "Hotel Name", "Website", 
                "Address", "Coordinates", "Contact", "Subjects", "Dates"
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            contacts = session.query(Contact).all()
            for contact in contacts:
                writer.writerow({
                    "Email": contact.email,
                    "Display Name": contact.display_name,
                    "Hotel Name": contact.hotel_name,
                    "Website": contact.website,
                    "Address": contact.address,
                    "Coordinates": contact.coordinates,
                    "Contact": contact.contact,
                    "Subjects": "; ".join(json.loads(contact.subjects) if contact.subjects else []),
                    "Dates": "; ".join(json.loads(contact.dates) if contact.dates else [])
                })
        except Exception as e:
            logging.error(f"Error writing CSV: {e}")
            raise


def process_mbox_in_chunks(mbox_path, output_csv):
    """Process mbox file in chunks, writing results as we go."""
    try:
        chunk_size = CHUNK_SIZE_MB * 1024 * 1024  # Convert MB to bytes
        total_size = os.path.getsize(mbox_path)
        
        if total_size <= chunk_size:
            # Small enough to process directly
            session = Session()
            try:
                process_mbox(mbox_path, session)
                write_csv(session, output_csv)
            finally:
                session.close()
            return
            
        print(f"Processing {mbox_path} ({total_size/1024/1024:.2f} MB) in {CHUNK_SIZE_MB}MB chunks...")
        
        current_chunk = []
        current_size = 0
        chunk_number = 1
        
        with open(mbox_path, 'rb') as f:
            # Process the file chunk by chunk
            for line in f:
                if line.startswith(b'From ') and current_size > chunk_size:
                    # Process current chunk
                    chunk_path = f"{mbox_path}.temp_chunk"
                    with open(chunk_path, 'wb') as chunk_file:
                        chunk_file.writelines(current_chunk)
                    
                    print(f"\nProcessing chunk {chunk_number} ({current_size/1024/1024:.2f} MB)")
                    session = Session()
                    try:
                        process_mbox(chunk_path, session)
                        # Append results to main CSV
                        write_csv(session, output_csv, append=chunk_number > 1)
                    finally:
                        session.close()
                        os.remove(chunk_path)  # Clean up temp chunk
                    
                    current_chunk = []
                    current_size = 0
                    chunk_number += 1
                
                current_chunk.append(line)
                current_size += len(line)
            
            # Process final chunk if there's data
            if current_chunk:
                chunk_path = f"{mbox_path}.temp_chunk"
                with open(chunk_path, 'wb') as chunk_file:
                    chunk_file.writelines(current_chunk)
                
                print(f"\nProcessing final chunk {chunk_number} ({current_size/1024/1024:.2f} MB)")
                session = Session()
                try:
                    process_mbox(chunk_path, session)
                    write_csv(session, output_csv, append=chunk_number > 1)
                finally:
                    session.close()
                    os.remove(chunk_path)  # Clean up temp chunk
        
        print("\nProcessing complete!")
        print(f"Results written to: {output_csv}")
        
    except Exception as e:
        print(f"Error processing mbox: {str(e)}")
        logging.error(f"Error processing mbox: {str(e)}")
        raise

def main():
    """Main entry point for the email processor."""
    logging.info("Starting email processing job")
    try:
        # Check if required services are running
        if not check_services():
            return
            
        # Process mbox file in chunks, writing results as we go
        process_mbox_in_chunks(MBOX_FILE, OUTPUT_CSV)
        
        logging.info("Email processing job completed successfully")
    except Exception as e:
        logging.error(f"Fatal error in main: {e}")
        raise


if __name__ == "__main__":
    main()
