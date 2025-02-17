# HotelMail Analyzer

## About

HotelMail Analyzer is an intelligent email processing system designed to streamline the identification and extraction of hospitality-related communications. Born from the need to efficiently process large email datasets and identify potential TipJAR customers in the hospitality sector, this tool combines the power of AI (using LM Studio with Mistral-7B) with robust data processing capabilities.

### Why HotelMail Analyzer?

- **Time-Saving**: Automatically processes thousands of emails to identify hotel-related communications
- **Intelligent Analysis**: Uses AI to understand context and extract relevant information
- **Data Enrichment**: Enhances contact information with Google Places API integration
- **Memory Efficient**: Handles large MBOX files through chunked processing
- **Reliable Storage**: Maintains a SQLite database for processed data with resume capability
- **Structured Output**: Exports findings in CSV format for easy integration with other tools

### Alternative Names

If you're looking to rebrand or customize this tool, here are some suggested names:
- HospitalityMail Inspector
- LodgeMail Processor
- HotelComm Analyzer
- AccomMail Scanner
- StayMail Processor
- HotelLeads Extractor
- HospitalityIntel
- LodgingMail Finder

## Features

- Processes large MBOX files in manageable chunks
- Uses LM Studio (with Mistral-7B) for intelligent email classification
- Extracts hotel names and contact information
- Integrates with Google Places API for additional hotel details
- Maintains a SQLite database for processed emails and contacts
- Exports results to CSV format
- Supports batch processing for improved GPU efficiency
- Includes comprehensive logging

## Requirements

- Python 3.x
- LM Studio running locally with Mistral-7B model
- Google Places API key
- Required Python packages (install via pipenv):
  - sqlalchemy
  - requests
  - python-decouple
  - tqdm
  - mailbox
  - email

## Setup

1. Install dependencies using Pipenv:
   ```bash
   pipenv install
   ```

2. Create a `.env` file in the project root with the following variables:
   ```
   MBOX_FILE=path/to/your_downloaded_file.mbox
   OUTPUT_CSV=tipjar_customers.csv
   GOOGLE_PLACES_API_KEY=your_google_places_api_key
   LM_STUDIO_API_URL=http://127.0.0.1:1234/v1/completions
   LM_STUDIO_MODEL=mistral-7b-instruct-v0.2
   ```

3. Ensure LM Studio is running with:
   - The Mistral-7B model loaded
   - API enabled in settings
   - Running on port 1234

## Usage

1. Start LM Studio and load the Mistral-7B model

2. Run the script:
   ```bash
   pipenv run python chatgpt_email.py
   ```

The script will:
- Process the MBOX file in chunks (default 100MB)
- Classify emails using LM Studio
- Extract hotel and contact information
- Store results in a SQLite database
- Export findings to CSV

## Technical Details

### Processing Flow

1. **Chunked Processing**: Large MBOX files are processed in chunks to manage memory usage
2. **Email Classification**: Each email is analyzed by Mistral-7B to determine if it's hotel-related
3. **Information Extraction**: For hotel-related emails:
   - Extracts email addresses and display names
   - Identifies hotel names
   - Fetches additional details via Google Places API
4. **Data Storage**: 
   - Maintains a SQLite database for processed emails and contacts
   - Tracks processing status to enable resume capability
   - Stores extracted information in structured format

### Database Schema

Two main tables:
1. `processed_emails`:
   - Tracks processed emails
   - Stores processing status and errors
   - Prevents duplicate processing

2. `contacts`:
   - Stores contact information
   - Maintains hotel details
   - Tracks email subjects and dates
   - Supports incremental updates

### GPU Optimization

- Implements batch processing for GPU efficiency
- Default batch size: 4 emails
- Configurable GPU memory allocation (default: 4GB)
- Supports chunked processing for large files

## Limitations and Considerations

1. **Resource Requirements**:
   - Requires LM Studio running locally
   - GPU recommended for efficient processing
   - Memory usage scales with chunk size

2. **API Dependencies**:
   - Requires active internet connection
   - Google Places API key needed
   - API rate limits may apply

3. **Processing Time**:
   - Large MBOX files may take significant time
   - Processing speed depends on:
     - GPU capabilities
     - Chunk size
     - Batch size
     - File size

4. **Data Quality**:
   - Hotel name extraction accuracy depends on email content
   - Google Places API matches may not always be perfect
   - Some emails may be misclassified

## Logging

The application maintains detailed logs in `email_processor.log`, including:
- Processing progress
- Error messages
- API responses
- Classification results

## Output

Results are exported to:
1. SQLite database (`email_processing.db`)
2. CSV file (configurable in .env)

The CSV includes:
- Email addresses
- Display names
- Hotel information
- Contact details
- Subject lines
- Communication dates

## Contributing

Feel free to submit issues and enhancement requests.

## License

[MIT License](LICENSE)
