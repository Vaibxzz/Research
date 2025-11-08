"""
Extracts and Saves Features
---------------------------
Runs all agents on train and test sets and saves the
9-dimensional feature vectors and labels to a JSON file.
This prevents running expensive API calls repeatedly.
"""

import os
import sys
import json
import logging
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import all the necessary classes from your project
from src.agents.text_agent import TextAgent
from src.agents.url_agent import URLAgent
from src.agents.metadata_agent import MetadataAgent
from src.ppo.feature_extractor import FeatureExtractor # The file you just sent
from src.utils.email_parser import EmailParser

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml"):
    """Load configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_agents_on_email(email_data: dict, text_agent: TextAgent,
                        url_agent: URLAgent, metadata_agent: MetadataAgent):
    """(Copied from evaluate.py) Run all three agents on an email"""
    body_text = email_data.get('body_text', '')
    urls = email_data.get('urls', [])
    metadata = email_data.get('metadata', {})
    
    text_result = text_agent.analyze(body_text)
    url_result = url_agent.analyze(urls)
    metadata_result = metadata_agent.analyze(metadata)
    
    return {
        'text_agent': text_result,
        'url_agent': url_result,
        'metadata_agent': metadata_result,
    }

def process_dataset(emails: list, text_agent: TextAgent, url_agent: URLAgent, 
                    metadata_agent: MetadataAgent, feature_extractor: FeatureExtractor) -> (list, list):
    """Helper function to run processing loop"""
    
    feature_list = []
    label_list = []
    
    for email in tqdm(emails, desc="Processing emails"):
        # 1. Run API calls
        agent_results = run_agents_on_email(email, text_agent, url_agent, metadata_agent)
        
        # 2. Extract 9-feature vector
        features = feature_extractor.extract_features(email, agent_results)
        
        # 3. Add to our lists
        # We use .tolist() to convert numpy.float32 to a normal float for JSON
        feature_list.append(features.tolist()) 
        label_list.append(email['label'])
        
    return feature_list, label_list


def main():
    parser = argparse.ArgumentParser(description='Extract and save features for PPO and XGBoost')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--output_file', type=str, default='data/features.json', help='Output JSON file')
    
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    llm_config = config['llm']
    provider = llm_config.get('provider', 'google')
    model = llm_config.get('model', None)
    
    # Initialize all components
    logger.info("Initializing agents and feature extractor...")
    text_agent = TextAgent(provider=provider, model=model)
    url_agent = URLAgent(provider=provider, model=model)
    metadata_agent = MetadataAgent(provider=provider, model=model)
    feature_extractor = FeatureExtractor()
    
    # --- Load TRAIN data ---
    logger.info("Loading TRAINING data (500 emails)...")
    train_path = Path(args.data_dir) / 'processed' / 'train.json'
    with open(train_path, 'r') as f:
        train_emails = json.load(f)
    
    # --- Process TRAIN data ---
    logger.info(f"Running API calls on {len(train_emails)} training emails...")
    train_X, train_y = process_dataset(train_emails, text_agent, url_agent, metadata_agent, feature_extractor)
    
    # --- Load TEST data ---
    logger.info("Loading TEST data (100 emails)...")
    test_path = Path(args.data_dir) / 'processed' / 'test.json'
    with open(test_path, 'r') as f:
        test_emails = json.load(f)
        
    # --- Process TEST data ---
    logger.info(f"Running API calls on {len(test_emails)} test emails...")
    test_X, test_y = process_dataset(test_emails, text_agent, url_agent, metadata_agent, feature_extractor)
    
    # --- Save all data to one file ---
    logger.info(f"Saving all features to {args.output_file}...")
    
    all_data = {
        "train_X": train_X,
        "train_y": train_y,
        "test_X": test_X,
        "test_y": test_y
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(all_data, f, indent=2)
        
    logger.info("Feature extraction complete!")
    logger.info(f"Train features: {len(train_X)} samples")
    logger.info(f"Test features: {len(test_X)} samples")


if __name__ == "__main__":
    main()