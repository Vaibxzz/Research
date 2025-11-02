"""
Sample Training Script
Trains on a small subset first to verify everything works
"""

import os
import sys
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.train import prepare_training_data
from src.ppo.ppo_module import PPOModule
from src.ppo.feature_extractor import FeatureExtractor
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml"):
    """Load configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    # Load config
    config = load_config()
    
    # Load training data
    train_path = Path('data/processed/train.json')
    with open(train_path, 'r') as f:
        train_emails = json.load(f)
    
    # Use smaller sample for testing (first 100 emails)
    sample_size = 100
    logger.info(f"Using sample of {sample_size} emails for training test")
    train_sample = train_emails[:sample_size]
    
    logger.info("Preparing training data...")
    features, labels, agent_probs = prepare_training_data(
        train_sample, 
        config, 
        use_adversarial=False  # Skip adversarial for quick test
    )
    
    logger.info(f"Prepared {len(features)} training samples")
    
    # Initialize PPO
    feature_extractor = FeatureExtractor()
    ppo_module = PPOModule(
        feature_dim=feature_extractor.get_feature_dim(),
        config=config['ppo']
    )
    
    # Train with smaller timesteps for testing
    logger.info("Starting PPO training (test run)...")
    model_dir = Path('models')
    model_dir.mkdir(exist_ok=True)
    
    ppo_module.train(
        features=features,
        labels=labels,
        agent_probs=agent_probs,
        total_timesteps=10000,  # Small for testing
        save_path=str(model_dir / 'ppo_model_test.zip'),
        log_dir=str(Path('logs'))
    )
    
    logger.info("✅ Sample training complete!")
    logger.info("Model saved to models/ppo_model_test.zip")


if __name__ == "__main__":
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
    
    main()

