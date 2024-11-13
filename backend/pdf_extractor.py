import re
from PyPDF2 import PdfReader

# Function to extract data from PDF using PyPDF2
def extract_pdf_data(filepath):
    reader = PdfReader(filepath)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    # Normalize the text (remove newlines, etc.)
    normalized_text = normalize_text(text)
    print("Extracted Text from PDF (Before Normalization):\n", text)
    print("Normalized Text from PDF:\n", normalized_text)

    # Extract patient details using regex
    data = {
        'tissue_type': extract_value_with_regex(normalized_text, r"Blood Type\s*:\s*(\w+)"),
        'medical_history': extract_value_with_regex(normalized_text, r"Medical History\s*:\s*(None|Diabetes|Hypertension|[\w\s]+)"),
        'demographics': extract_value_with_regex(normalized_text, r"Demographics\s*:\s*(\w+)"),
        'age': convert_to_int(extract_value_with_regex(normalized_text, r"Age\s*:\s*(\d+)")),
        'sex': map_sex(extract_value_with_regex(normalized_text, r"Sex\s*:\s*(Male|Female)")),
        'weight': convert_to_int(extract_value_with_regex(normalized_text, r"Weight\(in kg\)\s*:\s*(\d+)")),
        'height': convert_to_int(extract_value_with_regex(normalized_text, r"Height\(in cm\)\s*:\s*(\d+)")),
        'waiting_time': convert_to_int(extract_value_with_regex(normalized_text, r"Waiting Time\s*(\(in months\))?\s*:\s*(\d+)")),
        'urgency': extract_value_with_regex(normalized_text, r"Urgency\s*:\s*(\w+)")
    }
    return data

# Utility function to normalize text by removing extra newlines
def normalize_text(text):
    return ' '.join(text.split())

# Utility function to extract values using regex
def extract_value_with_regex(text, pattern):
    match = re.search(pattern, text)
    if match:
        value = match.group(1).strip()
        print(f"Extracted value: '{value}' for pattern '{pattern}'")
        return value
    else:
        print(f"Pattern '{pattern}' not found in text.")
        return ""

# Utility function to safely convert values to int
def convert_to_int(value):
    try:
        return int(value)
    except ValueError:
        print(f"Warning: Unable to convert '{value}' to int. Returning 0.")
        return 0

# Utility function to map 'Male'/'Female' to 'M'/'F'
def map_sex(sex_value):
    if sex_value == 'Male':
        return 'M'
    elif sex_value == 'Female':
        return 'F'
    else:
        return ''  # Default case if unknown value
