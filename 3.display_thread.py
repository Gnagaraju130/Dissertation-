import pickle
import pandas as pd
import random
from difflib import SequenceMatcher

def is_near_duplicate(text1, text2, threshold=0.95):
    """Check if two texts are near duplicates using SequenceMatcher."""
    if not text1 or not text2:
        return False
    return SequenceMatcher(None, text1, text2).ratio() > threshold

def load_and_analyze_threads(pickle_file='data/processed_enron.pkl'):
    """Load data and compute thread-level statistics."""
    try:
        df = pd.read_pickle(pickle_file)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return None, None
    
    print(f"Available columns: {list(df.columns)}")
    
    # Check for required columns
    required_columns = ['thread_id', 'message_id', 'timestamp', 'subject', 'cleaned_text']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing columns in dataset: {missing_columns}")
        return None, None
    
    # Check for optional columns
    optional_columns = ['from', 'to', 'in_reply_to']
    available_optional = [col for col in optional_columns if col in df.columns]
    missing_optional = [col for col in optional_columns if col not in df.columns]
    if missing_optional:
        print(f"Warning: Missing optional columns: {missing_optional}")
    if available_optional:
        print(f"Available optional columns: {available_optional}")
    
    # Compute thread counts
    thread_counts = df['thread_id'].value_counts()
    total_threads = thread_counts.size
    single_threads = (thread_counts == 1).sum()
    multi_threads = (thread_counts >= 2).sum()
    total_multi_emails = thread_counts[thread_counts >= 2].sum()
    
    # Verify thread sizes (2–5 emails)
    if single_threads > 0:
        print(f"Error: Found {single_threads} single-email threads. These should have been filtered.")
    if (thread_counts > 5).any():
        print(f"Error: Found threads with more than 5 emails: {thread_counts[thread_counts > 5].to_dict()}")
    
    # Check for duplicates within threads
    duplicate_count = 0
    for thread_id, group in df.groupby('thread_id'):
        for i in range(1, len(group)):
            prev_row = group.iloc[i-1]
            curr_row = group.iloc[i]
            if (prev_row['normalized_subject'] == curr_row['normalized_subject'] and
                is_near_duplicate(prev_row['cleaned_text'], curr_row['cleaned_text'])):
                duplicate_count += 1
    if duplicate_count > 0:
        print(f"Warning: Found {duplicate_count} near-duplicate emails within threads.")
    
    # Display summary
    print(f"\nThread Analysis:")
    print(f"Total emails loaded: {len(df)}")
    print(f"Total threads: {total_threads}")
    print(f"Multi-email threads (2–5 emails): {multi_threads} ({multi_threads/total_threads:.1%})")
    print(f"Total emails in multi-email threads: {total_multi_emails}")
    
    # List top threads by size
    top_threads = thread_counts.nlargest(5)
    print("\nTop 5 threads by number of emails:")
    for thread_id, count in top_threads.items():
        print(f"  {thread_id[:50]}... : {count} emails")
    
    # Thread length distribution
    thread_lengths = thread_counts.value_counts().sort_index()
    print(f"\nThread length distribution:")
    for size, count in thread_lengths.items():
        print(f"  {size} emails: {count} threads")
    
    # Time span analysis
    if multi_threads > 0:
        time_spans = []
        for thread_id in thread_counts.index:
            thread_data = df[df['thread_id'] == thread_id]
            if len(thread_data) >= 2:
                span = (thread_data['timestamp'].max() - thread_data['timestamp'].min()).total_seconds() / 3600
                time_spans.append(span)
        if time_spans:
            print(f"\nTime span analysis for multi-email threads:")
            print(f"  Mean duration: {sum(time_spans)/len(time_spans):.1f} hours")
            print(f"  Median duration: {sorted(time_spans)[len(time_spans)//2]:.1f} hours")
            print(f"  Max duration: {max(time_spans):.1f} hours")
    
    return df, thread_counts

def display_thread_example(df, thread_id=None, exact_size=None):
    """Display full details of a thread with exact number of emails, including complete message."""
    if df is None:
        print("Error: No data to display.")
        return
    
    # Select a thread
    if thread_id is None:
        if exact_size is not None:
            valid_threads = df['thread_id'].value_counts()[df['thread_id'].value_counts() == exact_size].index
            if len(valid_threads) == 0:
                print(f"Error: No threads with exactly {exact_size} emails found.")
                print("Check preprocess.py threading logic or dataset size.")
                return
            thread_id = random.choice(valid_threads)
        else:
            multi_threads = df['thread_id'].value_counts()[df['thread_id'].value_counts() >= 2].index
            if len(multi_threads) == 0:
                print("Error: No multi-email threads found. Check preprocessing for threading issues.")
                return
            thread_id = random.choice(multi_threads)
    
    thread_emails = df[df['thread_id'] == thread_id].sort_values('timestamp')
    print(f"\nThread Example: {thread_id[:50]}... ({len(thread_emails)} emails)")
    print("=" * 80)
    
    for i, row in enumerate(thread_emails.itertuples(), 1):
        print(f"Email {i}:")
        print(f"  Date     : {row.timestamp}")
        print(f"  From     : {getattr(row, 'from', 'N/A')}")
        print(f"  To       : {getattr(row, 'to', 'N/A')}")
        print(f"  Subject  : {row.subject}")
        if hasattr(row, 'in_reply_to') and row.in_reply_to:
            print(f"  InReplyTo: {row.in_reply_to[:50]}...")
        print(f"  MessageID: {row.message_id[:50]}...")
        print(f"  Full Message:")
        print(f"  {row.cleaned_text}")
        print("-" * 80)

def main():
    df, thread_counts = load_and_analyze_threads()
    if df is not None and thread_counts is not None:
        # Display a thread with exactly 5 emails
        display_thread_example(df, exact_size=5)

if __name__ == '__main__':
    main()