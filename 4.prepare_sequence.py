import pandas as pd
import os

def prepare_sequences(input_path='data/processed_enron_with_emotions.pkl', output_path='data/sequences_enron.pkl'):
    """Prepare thread-based sequences for training."""
    try:
        df = pd.read_pickle(input_path)
    except Exception as e:
        print(f"Error loading {input_path}: {e}")
        return
    
    required_columns = ['thread_id', 'cleaned_text', 'emotion_idx', 'timestamp']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing columns in dataset: {missing_columns}")
        return
    
    # Group by thread_id and create sequences
    sequences = []
    for thread_id, group in df.groupby('thread_id'):
        group = group.sort_values('timestamp')
        if len(group) < 2 or len(group) > 3:
            print(f"Warning: Thread {thread_id[:50]} has {len(group)} emails, skipping.")
            continue
        sequences.append({
            'thread_id': thread_id,
            'texts': group['cleaned_text'].tolist(),
            'emotions': group['emotion_idx'].tolist()
        })
    
    seq_df = pd.DataFrame(sequences)
    
    # Verify sequence lengths
    seq_lengths = seq_df['texts'].apply(len)
    print(f"Sequence length distribution:\n{seq_lengths.value_counts().sort_index()}")
    
    # Verify emotions
    valid_emotions = set(range(6))  # 0: joy, 1: anger, 2: sadness, 3: fear, 4: trust, 5: neutral
    invalid_emotions = []
    for seq in seq_df['emotions']:
        for e in seq:
            if e not in valid_emotions:
                invalid_emotions.append(e)
    if invalid_emotions:
        print(f"Error: Invalid emotion indices found: {set(invalid_emotions)}")
    
    # Sample sequences
    print(f"Sample sequences:")
    print(seq_df[['thread_id', 'texts', 'emotions']].head())
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    seq_df.to_pickle(output_path)
    print(f"Sequences saved to {output_path}")

if __name__ == "__main__":
    prepare_sequences()