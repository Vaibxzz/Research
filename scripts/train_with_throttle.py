"""
Training Script with API Rate Limiting
Processes emails in batches to respect API rate limits
"""

import os
import sys
import json
import logging
import time
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.text_agent import TextAgent
from src.agents.url_agent import URLAgent
from src.agents.metadata_agent import MetadataAgent
from src.utils.email_parser import EmailParser
from src.ppo.feature_extractor import FeatureExtractor
from src.ppo.ppo_module import PPOModule
from src.train import run_agents_on_email
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml"):
    """Load configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def process_with_rate_limit(emails, agents, batch_size=9, delay_between_batches=6):
    """
    Process emails with rate limiting
    Free tier: 10 requests/minute = ~9 requests per batch with delay
    """
    email_parser = EmailParser()
    feature_extractor = FeatureExtractor()
    
    features_list = []
    labels_list = []
    agent_probs_list = []
    
    text_agent, url_agent, metadata_agent = agents
    
    total_batches = (len(emails) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(total_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(emails))
        batch = emails[start_idx:end_idx]
        
        # Process batch
        for email in batch:
            try:
                agent_results = run_agents_on_email(email, text_agent, url_agent, metadata_agent)
                features = feature_extractor.extract_features(email, agent_results)
                
                features_list.append(features)
                labels_list.append(email['label'])
                agent_probs_list.append({
                    'text': agent_results['text_prob'],
                    'url': agent_results['url_prob'],
                    'meta': agent_results['meta_prob']
                })
            except Exception as e:
                logger.error(f"Error processing email {email.get('id', 'unknown')}: {e}")
                # Add default values
                features = feature_extractor.extract_features(email, {
                    'text_agent': {'phishing_probability': 0.5},
                    'url_agent': {'phishing_probability': 0.5},
                    'metadata_agent': {'phishing_probability': 0.5}
                })
                features_list.append(features)
                labels_list.append(email['label'])
                agent_probs_list.append({
                    'text': 0.5,
                    'url': 0.5,
                    'meta': 0.5
                })
        
        # Wait between batches to respect rate limits (except for last batch)
        if batch_idx < total_batches - 1:
            time.sleep(delay_between_batches)
    
    return features_list, labels_list, agent_probs_list


def main():
    # Load config
    config = load_config()
    
    # Load environment variables from .env file
    env_path = Path('.') / '.env'
    if env_path.exists():
        load_dotenv(env_path)
    
    # Ensure API key is set
    if not os.getenv('GOOGLE_API_KEY'):
        raise ValueError(
            "GOOGLE_API_KEY not found. Please set it in .env file or environment variable.\n"
            "Create a .env file with: GOOGLE_API_KEY=your_key_here"
        )
    
    # Load training data
    logger.info("Loading training data...")
    train_path = Path('data/processed/train.json')
    with open(train_path, 'r') as f:
        train_emails = json.load(f)
    
    # Use smaller subset for now (can increase later)
    sample_size = min(500, len(train_emails))  # Start with 500 emails
    logger.info(f"Using {sample_size} emails for training")
    train_sample = train_emails[:sample_size]
    
    # Initialize agents
    logger.info("Initializing agents...")
    llm_config = config['llm']
    provider = llm_config.get('provider', 'google')
    model = llm_config.get('model', None)
    
    text_agent = TextAgent(provider=provider, model=model, temperature=0.3, max_tokens=500)
    url_agent = URLAgent(provider=provider, model=model, temperature=0.3, max_tokens=500)
    metadata_agent = MetadataAgent(provider=provider, model=model, temperature=0.3, max_tokens=500)
    
    agents = (text_agent, url_agent, metadata_agent)
    
    # Process with rate limiting
    logger.info("Processing emails with rate limiting (batch size: 9, delay: 6s)...")
    logger.info("This will take approximately: {} minutes".format(
        (len(train_sample) // 9) * (6 / 60)
    ))
    
    features, labels, agent_probs = process_with_rate_limit(
        train_sample, 
        agents,
        batch_size=9,  # 9 requests per batch (leaves 1 for buffer)
        delay_between_batches=6  # 6 seconds between batches = ~10 batches/minute
    )
    
    logger.info(f"Processed {len(features)} emails")
    
    # Initialize PPO
    logger.info("Initializing PPO module...")
    feature_extractor = FeatureExtractor()
    ppo_module = PPOModule(
        feature_dim=feature_extractor.get_feature_dim(),
        config=config['ppo']
    )
    
    # Train PPO
    logger.info("Starting PPO training...")
    model_dir = Path('models')
    model_dir.mkdir(parents=True, exist_ok=True)
    
    ppo_module.train(
        features=features,
        labels=labels,
        agent_probs=agent_probs,
        total_timesteps=50000,  # Moderate training
        save_path=str(model_dir / 'ppo_model_trained.zip'),
        log_dir=str(Path('logs'))
    )
    
    logger.info("✅ Training complete!")
    logger.info(f"Model saved to {model_dir / 'ppo_model_trained.zip'}")


if __name__ == "__main__":
    main()

