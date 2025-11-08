"""
Data Preprocessing Script
Preprocesses email datasets and prepares them for training
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
import random  # --- MODIFICATION 1: ADDED IMPORT ---

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.dataset_loader import DatasetLoader
from src.utils.email_parser import EmailParser

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def preprocess_emails(emails: list, email_parser: EmailParser):
    """
    Preprocess emails: extract components
    
    Args:
        emails: List of email dictionaries with 'text' key
        email_parser: EmailParser instance
    
    Returns:
        List of preprocessed email dictionaries
    """
    preprocessed = []
    
    for email in emails:
        try:
            # Parse email text
            # Try as email message first
            try:
                parsed = email_parser.parse_email_string(email['text'])
            except:
                # If not valid email format, treat as plain text
                parsed = {
                    'body_text': email['text'],
                    'urls': email_parser._extract_urls(email['text']),
                    'metadata': {
                        'sender': '',
                        'sender_domain': '',
                        'spf_pass': False,
                        'dkim_pass': False,
                        'dmarc_pass': False
                    },
                    'phishing_keyword_count': email_parser._count_phishing_keywords(email['text'])
                }
            
            # Add label and metadata
            preprocessed_email = {
                'id': email.get('id', 'unknown'),
                'source': email.get('source', 'unknown'),
                'label': email['label'],
                'body_text': parsed['body_text'],
                'urls': parsed['urls'],
                'metadata': parsed['metadata'],
                'phishing_keyword_count': parsed.get('phishing_keyword_count', 0),
                'original_text': email['text']  # Keep original for reference
            }
            
            preprocessed.append(preprocessed_email)
            
        except Exception as e:
            logger.warning(f"Error preprocessing email {email.get('id', 'unknown')}: {e}")
            continue
    
    return preprocessed


def main():
    parser = argparse.ArgumentParser(description='Preprocess email datasets')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Data directory')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                        help='Output directory for processed data')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='Test set ratio (ignored for mini-dataset)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    splits_dir = Path(args.data_dir) / 'splits'
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    dataset_loader = DatasetLoader(args.data_dir)
    email_parser = EmailParser()
    
    logger.info("Loading all datasets...")
    phishing_emails, legitimate_emails = dataset_loader.load_all_datasets()
    
    if len(phishing_emails) == 0:
        logger.error("No phishing emails loaded. Please check dataset files.")
        return
    
    if len(legitimate_emails) == 0:
        logger.error("No legitimate emails loaded. Please check dataset files.")
        return
    
    logger.info(f"Loaded {len(phishing_emails)} phishing and {len(legitimate_emails)} legitimate emails")

    # --- MODIFICATION 2: CREATE MINI-DATASET ---
    # ---------------------------------------------------------------
    logger.info("CREATING MINI-DATASET (1000 train, 200 test)...")
    
    MINI_TRAIN_PHISHING = 250
    MINI_TRAIN_LEGIT = 250
    MINI_TEST_PHISHING = 50
    MINI_TEST_LEGIT = 50
    
    TOTAL_PHISHING_NEEDED = MINI_TRAIN_PHISHING + MINI_TEST_PHISHING
    TOTAL_LEGIT_NEEDED = MINI_TRAIN_LEGIT + MINI_TEST_LEGIT

    if len(phishing_emails) < TOTAL_PHISHING_NEEDED:
        logger.warning(f"Not enough phishing emails! Need {TOTAL_PHISHING_NEEDED}, have {len(phishing_emails)}")
        TOTAL_PHISHING_NEEDED = len(phishing_emails) # Use all available

    if len(legitimate_emails) < TOTAL_LEGIT_NEEDED:
        logger.warning(f"Not enough legitimate emails! Need {TOTAL_LEGIT_NEEDED}, have {len(legitimate_emails)}")
        TOTAL_LEGIT_NEEDED = len(legitimate_emails) # Use all available

    # Shuffle for randomness
    random.seed(args.random_seed)
    random.shuffle(phishing_emails)
    random.shuffle(legitimate_emails)
    
    # Select the mini-dataset
    phishing_emails = phishing_emails[:TOTAL_PHISHING_NEEDED]
    legitimate_emails = legitimate_emails[:TOTAL_LEGIT_NEEDED]
    
    logger.info(f"Mini-dataset created. Now preprocessing {len(phishing_emails)} phishing and {len(legitimate_emails)} total emails...")
    # ---------------------------------------------------------------
    # --- END OF MODIFICATION 2 ---

    
    # Preprocess emails
    logger.info("Preprocessing emails...")
    phishing_preprocessed = preprocess_emails(phishing_emails, email_parser)
    legitimate_preprocessed = preprocess_emails(legitimate_emails, email_parser)
    
    logger.info(f"Preprocessed {len(phishing_preprocessed)} phishing and {len(legitimate_preprocessed)} legitimate emails")
    
    # --- MODIFICATION 3: REPLACE SPLIT LOGIC ---
    # ---------------------------------------------------------------
    logger.info("Splitting mini-dataset into train/test sets...")
    
    # We defined these above, but re-defining here for clarity
    MINI_TRAIN_PHISHING = 250   
    MINI_TRAIN_LEGIT = 250
    
    # Check if we have enough preprocessed emails (some might have failed)
    actual_phishing_train_count = min(MINI_TRAIN_PHISHING, len(phishing_preprocessed))
    actual_legit_train_count = min(MINI_TRAIN_LEGIT, len(legitimate_preprocessed))

    # We shuffle again *just in case* preprocessing changed the order (it shouldn't)
    random.seed(args.random_seed)
    random.shuffle(phishing_preprocessed)
    random.shuffle(legitimate_preprocessed)
    
    phishing_train = phishing_preprocessed[:actual_phishing_train_count]
    phishing_test = phishing_preprocessed[actual_phishing_train_count:] # The rest are for testing
    
    legitimate_train = legitimate_preprocessed[:actual_legit_train_count]
    legitimate_test = legitimate_preprocessed[actual_legit_train_count:] # The rest are for testing

    logger.info("--- Mini-Dataset Split ---")
    logger.info(f"Phishing: {len(phishing_train)} train / {len(phishing_test)} test")
    logger.info(f"Legitimate: {len(legitimate_train)} train / {len(legitimate_test)} test")
    # ---------------------------------------------------------------
    # --- END OF MODIFICATION 3 ---

    
    # Combine train and test sets
    train_emails = phishing_train + legitimate_train
    test_emails = phishing_test + legitimate_test
    
    logger.info(f"Train set: {len(train_emails)} emails ({len(phishing_train)} phishing, {len(legitimate_train)} legitimate)")
    logger.info(f"Test set: {len(test_emails)} emails ({len(phishing_test)} phishing, {len(legitimate_test)} legitimate)")
    
    # Save processed data
    logger.info("Saving processed data...")
    
    # Save all processed emails
    with open(output_dir / 'train.json', 'w') as f:
        json.dump(train_emails, f, indent=2)
    
    with open(output_dir / 'test.json', 'w') as f:
        json.dump(test_emails, f, indent=2)
    
    # Save splits separately
    with open(splits_dir / 'train.json', 'w') as f:
        json.dump(train_emails, f, indent=2)
    
    with open(splits_dir / 'test.json', 'w') as f:
        json.dump(test_emails, f, indent=2)
    
    logger.info("Preprocessing complete!")
    logger.info(f"Processed data saved to {output_dir}")
    logger.info(f"Train/test splits saved to {splits_dir}")


if __name__ == "__main__":
    main()