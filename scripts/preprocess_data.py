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
                       help='Test set ratio')
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
    
    # Preprocess emails
    logger.info("Preprocessing emails...")
    phishing_preprocessed = preprocess_emails(phishing_emails, email_parser)
    legitimate_preprocessed = preprocess_emails(legitimate_emails, email_parser)
    
    logger.info(f"Preprocessed {len(phishing_preprocessed)} phishing and {len(legitimate_preprocessed)} legitimate emails")
    
    # Split into train/test
    logger.info("Splitting into train/test sets...")
    phishing_train, phishing_test = dataset_loader.split_train_test(
        phishing_preprocessed, args.test_ratio, args.random_seed
    )
    legitimate_train, legitimate_test = dataset_loader.split_train_test(
        legitimate_preprocessed, args.test_ratio, args.random_seed
    )
    
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





