"""
Training Pipeline for MultiPhishGuard
Orchestrates agents, PPO training, and adversarial data augmentation
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

# Add config to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.text_agent import TextAgent
from src.agents.url_agent import URLAgent
from src.agents.metadata_agent import MetadataAgent
from src.agents.adversarial_agent import AdversarialAgent
from src.utils.email_parser import EmailParser
from src.ppo.feature_extractor import FeatureExtractor
from src.ppo.ppo_module import PPOModule

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_agents_on_email(email_data: dict, text_agent: TextAgent, 
                        url_agent: URLAgent, metadata_agent: MetadataAgent):
    """
    Run all three agents on an email
    
    Returns:
        Dictionary with agent results and probabilities
    """
    body_text = email_data.get('body_text', '')
    urls = email_data.get('urls', [])
    metadata = email_data.get('metadata', {})
    
    # Run agents
    text_result = text_agent.analyze(body_text)
    url_result = url_agent.analyze(urls)
    metadata_result = metadata_agent.analyze(metadata)
    
    return {
        'text_agent': text_result,
        'url_agent': url_result,
        'metadata_agent': metadata_result,
        'text_prob': text_result.get('phishing_probability', text_result.get('confidence', 0.5)),
        'url_prob': url_result.get('phishing_probability', url_result.get('confidence', 0.5)),
        'meta_prob': metadata_result.get('phishing_probability', metadata_result.get('confidence', 0.5))
    }


def prepare_training_data(train_emails: list, config: dict, use_adversarial: bool = True):
    """
    Prepare training data: run agents and extract features
    
    Args:
        train_emails: List of preprocessed training emails
        config: Configuration dictionary
        use_adversarial: Whether to use adversarial augmentation
    
    Returns:
        Tuple of (features, labels, agent_probs)
    """
    logger.info("Initializing agents...")
    llm_config = config['llm']
    provider = llm_config.get('provider', 'google')
    model = llm_config.get('model', None)
    
    text_agent = TextAgent(
        provider=provider,
        model=model,
        temperature=llm_config.get('temperature', 0.3),
        max_tokens=llm_config.get('max_tokens', 500)
    )
    url_agent = URLAgent(
        provider=provider,
        model=model,
        temperature=llm_config.get('temperature', 0.3),
        max_tokens=llm_config.get('max_tokens', 500)
    )
    metadata_agent = MetadataAgent(
        provider=provider,
        model=model,
        temperature=llm_config.get('temperature', 0.3),
        max_tokens=llm_config.get('max_tokens', 500)
    )
    
    adversarial_agent = None
    if use_adversarial and config['training'].get('use_adversarial_augmentation', False):
        adversarial_agent = AdversarialAgent(
            provider=provider,
            model=model,
            temperature=0.7,
            max_tokens=1000
        )
    
    logger.info("Running agents on training emails...")
    features_list = []
    labels_list = []
    agent_probs_list = []
    
    email_parser = EmailParser()
    feature_extractor = FeatureExtractor()
    
    # Process emails
    for email in tqdm(train_emails, desc="Processing emails"):
        # Run agents
        agent_results = run_agents_on_email(email, text_agent, url_agent, metadata_agent)
        
        # Extract features
        features = feature_extractor.extract_features(email, agent_results)
        
        features_list.append(features)
        labels_list.append(email['label'])
        agent_probs_list.append({
            'text': agent_results['text_prob'],
            'url': agent_results['url_prob'],
            'meta': agent_results['meta_prob']
        })
        
        # Adversarial augmentation for phishing emails
        if use_adversarial and adversarial_agent and email['label'] == 1:
            if np.random.random() < 0.1:  # 10% chance to generate variant
                try:
                    variant_text = adversarial_agent.generate_variant(email['body_text'])
                    # Parse variant
                    variant_parsed = email_parser.parse_email_string(variant_text)
                    variant_email = {
                        **email,
                        'body_text': variant_parsed['body_text'],
                        'urls': variant_parsed['urls'],
                        'id': email['id'] + '_adversarial'
                    }
                    
                    # Run agents on variant
                    variant_agent_results = run_agents_on_email(
                        variant_email, text_agent, url_agent, metadata_agent
                    )
                    variant_features = feature_extractor.extract_features(
                        variant_email, variant_agent_results
                    )
                    
                    features_list.append(variant_features)
                    labels_list.append(1)  # Still phishing
                    agent_probs_list.append({
                        'text': variant_agent_results['text_prob'],
                        'url': variant_agent_results['url_prob'],
                        'meta': variant_agent_results['meta_prob']
                    })
                except Exception as e:
                    logger.warning(f"Error generating adversarial variant: {e}")
    
    logger.info(f"Prepared {len(features_list)} training samples")
    
    return features_list, labels_list, agent_probs_list


def main():
    parser = argparse.ArgumentParser(description='Train MultiPhishGuard model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Data directory')
    parser.add_argument('--model_dir', type=str, default='models',
                       help='Model save directory')
    parser.add_argument('--use_adversarial', action='store_true',
                       help='Use adversarial data augmentation')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Create directories
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = Path(config['paths']['logs'])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Load training data
    logger.info("Loading training data...")
    train_path = Path(args.data_dir) / 'processed' / 'train.json'
    
    if not train_path.exists():
        logger.error(f"Training data not found: {train_path}")
        logger.error("Please run scripts/preprocess_data.py first")
        return
    
    with open(train_path, 'r') as f:
        train_emails = json.load(f)
    
    logger.info(f"Loaded {len(train_emails)} training emails")
    
    # Prepare training data
    features, labels, agent_probs = prepare_training_data(
        train_emails, config, use_adversarial=args.use_adversarial
    )
    
    # Initialize PPO module
    logger.info("Initializing PPO module...")
    feature_extractor = FeatureExtractor()
    ppo_module = PPOModule(
        feature_dim=feature_extractor.get_feature_dim(),
        config=config['ppo']
    )
    
    # Train PPO
    logger.info("Starting PPO training...")
    model_save_path = model_dir / 'ppo_model_final.zip'
    
    ppo_module.train(
        features=features,
        labels=labels,
        agent_probs=agent_probs,
        total_timesteps=config['training'].get('total_timesteps', 100000),
        save_path=str(model_save_path),
        log_dir=str(log_dir)
    )
    
    logger.info("Training complete!")
    logger.info(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    main()


