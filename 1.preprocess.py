import pandas as pd
import re
from nltk.tokenize import sent_tokenize
import nltk
import os
from email.parser import BytesParser
from email.policy import default
from email.header import decode_header
import glob
import ssl
from datetime import datetime
import pytz
from difflib import SequenceMatcher

# Bypass SSL verification for NLTK download
try:
    _create_unverified_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_context

try:
    nltk.download('punkt', quiet=True)
except Exception as e:
    print(f"Failed to download punkt: {e}")
    print("Ensure punkt_tab is manually downloaded to ~/nltk_data/tokenizers/punkt_tab")

def clean_email(text):
    """Clean email text by removing headers, signatures, and noise."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'--+\n.*', '', text, flags=re.DOTALL)  # Remove signatures
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '', text)  # Remove email addresses
    text = re.sub(r'\n\s*\n', '\n', text)  # Remove extra newlines
    text = re.sub(r'From:.*|To:.*|Cc:.*', '', text, flags=re.IGNORECASE)  # Remove headers
    text = re.sub(r'[^\w\s.,!?]', '', text)  # Remove special characters
    return text.strip()

def normalize_subject(subject):
    """Normalize subject for thread grouping."""
    if not subject:
        return ''
    subject = re.sub(r'^(Re:|Fwd:|\[.*?\])\s*', '', subject, flags=re.IGNORECASE)
    subject = re.sub(r'[^\w\s]', '', subject)
    return subject.strip().lower()

def decode_email_header(header_value):
    """Safely decode email header values."""
    if not header_value:
        return ''
    try:
        decoded_parts = decode_header(header_value)
        decoded_string = ''
        for part, encoding in decoded_parts:
            if isinstance(part, bytes):
                decoded_string += part.decode(encoding or 'utf-8', errors='ignore')
            else:
                decoded_string += part
        return decoded_string.strip()
    except Exception:
        return str(header_value).strip()

def parse_timestamp(date_str):
    """Convert email date to datetime with UTC timezone."""
    if not date_str:
        return pd.NaT
    try:
        for fmt in (
            "%a, %d %b %Y %H:%M:%S %z",
            "%d %b %Y %H:%M:%S %z",
            "%Y-%m-%d %H:%M:%S"
        ):
            try:
                dt = datetime.strptime(date_str, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=pytz.UTC)
                return dt
            except ValueError:
                continue
        print(f"Warning: Unparseable date: {date_str}")
        return pd.NaT
    except Exception as e:
        print(f"Error parsing date {date_str}: {e}")
        return pd.NaT

def get_thread_id(msg, subject, file_path):
    """Generate thread ID based on headers or subject."""
    if msg['message-id']:
        msg_id = msg['message-id'].strip()
    else:
        msg_id = file_path
    if msg['in-reply-to']:
        return msg['in-reply-to'].strip()
    elif msg['references']:
        return msg['references'].split()[0].strip()
    return normalize_subject(subject) or msg_id

def is_near_duplicate(text1, text2, threshold=0.95):
    """Check if two texts are near duplicates using SequenceMatcher."""
    if not text1 or not text2:
        return False
    return SequenceMatcher(None, text1, text2).ratio() > threshold

def extract_emails_from_maildir(maildir_path):
    """Extract emails from maildir and group into threads."""
    emails = []
    target_subfolders = ['inbox', 'sent', 'sent_items', 'all_documents']
    parser = BytesParser(policy=default)
    print(f"Processing maildir: {maildir_path}")
    
    if not os.path.exists(maildir_path):
        print(f"Error: {maildir_path} does not exist.")
        return pd.DataFrame(emails)
    
    email_files = []
    for subfolder in target_subfolders:
        pattern = os.path.join(maildir_path, f'**/{subfolder}/*')
        files = glob.glob(pattern, recursive=True)
        email_files.extend([f for f in files if os.path.isfile(f)])
        print(f"Found {len(files)} files in {subfolder}")
    
    print(f"Total email files found: {len(email_files)}")
    
    for file_path in email_files:
        try:
            try:
                with open(file_path, 'rb') as f:
                    msg = parser.parse(f)
            except UnicodeDecodeError:
                with open(file_path, 'rb') as f:
                    msg = parser.parse(f, policy=default)
            
            subject = decode_email_header(msg['subject'])
            from_addr = decode_email_header(msg['from'])
            to_addr = decode_email_header(msg['to'])
            
            text = ''
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == 'text/plain':
                        payload = part.get_payload(decode=True)
                        if payload:
                            text = payload.decode(errors='ignore')
                            break
            else:
                payload = msg.get_payload(decode=True)
                if payload:
                    text = payload.decode(errors='ignore')
            
            timestamp = parse_timestamp(msg['date'])
            
            emails.append({
                'thread_id': get_thread_id(msg, subject, file_path),
                'message_id': msg['message-id'] or file_path,
                'references': msg['references'] or '',
                'in_reply_to': msg['in_reply_to'] or '',
                'timestamp': timestamp,
                'from': from_addr,
                'to': to_addr,
                'text': text,
                'subject': subject,
                'normalized_subject': normalize_subject(subject),
                'file_path': file_path
            })
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    df = pd.DataFrame(emails)
    print(f"Total emails collected: {len(df)}")
    
    if not df.empty:
        # Clean text and filter by length
        df['cleaned_text'] = df['text'].apply(clean_email)
        df = df[df['cleaned_text'].str.len() > 50]
        
        # Improve thread grouping
        subject_groups = df.groupby('normalized_subject')
        for norm_subj, group in subject_groups:
            if len(group) >= 2:
                thread_id = group['thread_id'].iloc[0]
                df.loc[df['normalized_subject'] == norm_subj, 'thread_id'] = thread_id
        
        # Deduplicate by message_id
        df = df.drop_duplicates(subset='message_id', keep='first')
        
        # Deduplicate near-duplicate emails within threads
        df = df.sort_values(['thread_id', 'timestamp'])
        to_drop = []
        for thread_id, group in df.groupby('thread_id'):
            for i in range(1, len(group)):
                prev_row = group.iloc[i-1]
                curr_row = group.iloc[i]
                if (prev_row['normalized_subject'] == curr_row['normalized_subject'] and
                    is_near_duplicate(prev_row['cleaned_text'], curr_row['cleaned_text'])):
                    to_drop.append(curr_row.name)
        df = df.drop(to_drop)
        
        # Filter threads with 2–3 emails
        thread_counts = df['thread_id'].value_counts()
        valid_thread_ids = thread_counts[(thread_counts >= 2) & (thread_counts <= 3)].index
        df = df[df['thread_id'].isin(valid_thread_ids)]
        
        # Limit to max 3 emails per thread
        df = df.groupby('thread_id').apply(lambda x: x.sort_values('timestamp').head(5)).reset_index(drop=True)
        
        # Recompute thread counts to ensure no single-email threads
        thread_counts = df['thread_id'].value_counts()
        valid_thread_ids = thread_counts[(thread_counts >= 2) & (thread_counts <= 3)].index
        df = df[df['thread_id'].isin(valid_thread_ids)]
        
        print(f"Emails after deduplication and filtering: {len(df)}")
        print(f"Unique threads (2–3 emails): {df['thread_id'].nunique()}")
        
        # Add sentences
        df['sentences'] = df['cleaned_text'].apply(sent_tokenize)
    
    return df

def preprocess_enron(maildir_path="/Users/satya/EmotionDriftDetection/maildir", output_path="data/processed_enron.pkl"):
    """Preprocess Enron dataset and save to pickle."""
    emails = extract_emails_from_maildir(maildir_path)
    if emails.empty:
        print("Warning: No emails processed from maildir. Check dataset path and subfolders.")
        return
    
    # Verify columns
    required_columns = ['thread_id', 'message_id', 'timestamp', 'from', 'to', 'subject', 'cleaned_text']
    print(f"Columns in processed data: {list(emails.columns)}")
    print(f"Sample data:")
    print(emails[required_columns].head())
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    emails.to_pickle(output_path)
    print(f"Preprocessed data saved to {output_path}")

if __name__ == "__main__":
    preprocess_enron()