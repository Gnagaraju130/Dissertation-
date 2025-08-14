import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import os
import logging
import numpy as np

def setup_logging():
    """Set up logging to track progress."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

def initialize_model():
    """Initialize RoBERTa model and tokenizer for emotion classification."""
    model_name = "cardiffnlp/twitter-roberta-base-emotion"
    logger = setup_logging()
    try:
        logger.info(f"Loading tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info(f"Loading model for {model_name}...")
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        logger.info("Model and tokenizer loaded successfully.")
        return tokenizer, model
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        return None, None

def sample_threads(df, threads_per_length={2: 2000, 3: 2000}):
    """Sample specified number of threads for each thread length."""
    logger = setup_logging()
    thread_counts = df['thread_id'].value_counts()
    sampled_thread_ids = []
    
    for length, num_threads in threads_per_length.items():
        # Get threads with the specified length
        valid_threads = thread_counts[thread_counts == length].index
        if len(valid_threads) < num_threads:
            logger.warning(f"Only {len(valid_threads)} threads with {length} emails available, sampling all.")
            sampled_threads = valid_threads
        else:
            sampled_threads = np.random.choice(valid_threads, size=num_threads, replace=False)
        sampled_thread_ids.extend(sampled_threads)
    
    logger.info(f"Sampled {len(sampled_thread_ids)} threads: {threads_per_length}")
    return df[df['thread_id'].isin(sampled_thread_ids)]

def annotate_emotions(input_path='data/processed_enron.pkl', output_path='data/processed_enron_with_emotions.pkl'):
    """Annotate emails with emotions using RoBERTa model."""
    logger = setup_logging()
    try:
        logger.info(f"Loading {input_path}...")
        df = pd.read_pickle(input_path)
        logger.info(f"Loaded {len(df)} emails.")
    except Exception as e:
        logger.error(f"Error loading {input_path}: {e}")
        return
    
    if 'cleaned_text' not in df.columns:
        logger.error("Error: 'cleaned_text' column missing in dataset.")
        return
    
    # Sample threads
    logger.info("Sampling threads (2000 each of 2, 3 emails)...")
    df = sample_threads(df)
    logger.info(f"After sampling: {len(df)} emails in {df['thread_id'].nunique()} threads.")
    
    tokenizer, model = initialize_model()
    if tokenizer is None or model is None:
        logger.error("Error: Model initialization failed.")
        return
    
    device = torch.device("cpu")  # Use CPU for consistency
    model = model.to(device)
    model.eval()
    
    # Emotion mapping
    model_emotions = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise', 'optimism']
    target_emotions = {'joy': 0, 'anger': 1, 'sadness': 2, 'fear': 3, 'trust': 4, 'neutral': 5}
    emotion_map = {
        'anger': 'anger',
        'disgust': 'neutral',
        'fear': 'fear',
        'joy': 'joy',
        'neutral': 'neutral',
        'sadness': 'sadness',
        'surprise': 'neutral',
        'optimism': 'trust'
    }
    
    emotions = []
    logger.info(f"Processing {len(df)} emails...")
    with torch.no_grad():
        for i, text in enumerate(tqdm(df['cleaned_text'], desc="Annotating emails")):
            if not text or len(text.strip()) < 10:
                emotions.append(('neutral', 5))
                continue
            try:
                inputs = tokenizer(text, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                pred_idx = torch.argmax(probs, dim=-1).item()
                pred_emotion = model_emotions[pred_idx]
                mapped_emotion = emotion_map.get(pred_emotion, 'neutral')
                emotions.append((mapped_emotion, target_emotions[mapped_emotion]))
            except Exception as e:
                logger.error(f"Error processing email {i}: {e}")
                emotions.append(('neutral', 5))
    
    df['emotion'] = [e[0] for e in emotions]
    df['emotion_idx'] = [e[1] for e in emotions]
    
    # Verify emotions
    logger.info(f"Emotion distribution:\n{df['emotion'].value_counts()}")
    
    # Verify thread lengths
    thread_counts = df['thread_id'].value_counts()
    logger.info(f"Thread length distribution:\n{thread_counts.value_counts().sort_index()}")
    
    # Verify columns
    required_columns = ['thread_id', 'message_id', 'timestamp', 'subject', 'cleaned_text', 'emotion', 'emotion_idx']
    logger.info(f"Columns in annotated data: {list(df.columns)}")
    logger.info("Sample data:")
    logger.info(f"\n{df[required_columns].head().to_string()}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_pickle(output_path)
    logger.info(f"Annotated data saved to {output_path}")

if __name__ == "__main__":
    print('Program Started')
    annotate_emotions()