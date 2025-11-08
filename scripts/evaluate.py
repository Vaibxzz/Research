"""
Evaluation Script
Evaluates trained model and compares with baseline
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

from src.agents.text_agent import TextAgent
from src.agents.url_agent import URLAgent
from src.agents.metadata_agent import MetadataAgent
from src.agents.explanation_simplifier import ExplanationSimplifier
from src.utils.email_parser import EmailParser
from src.utils.metrics import calculate_metrics, print_metrics_report, plot_confusion_matrix
from src.ppo.feature_extractor import FeatureExtractor
from src.ppo.ppo_module import PPOModule, MultiPhishGuardEnv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml"):
    """Load configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_agents_on_email(email_data: dict, text_agent: TextAgent,
                        url_agent: URLAgent, metadata_agent: MetadataAgent):
    """Run all three agents on an email"""
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
        'text_prob': text_result.get('phishing_probability', text_result.get('confidence', 0.5)),
        'url_prob': url_result.get('phishing_probability', url_result.get('confidence', 0.5)),
        'meta_prob': metadata_result.get('phishing_probability', metadata_result.get('confidence', 0.5))
    }


def evaluate_model(test_emails: list, model_path: str, config: dict, 
                   results_dir: str = "results"):
    """
    Evaluate trained PPO model
    
    Args:
        test_emails: List of test emails
        model_path: Path to trained PPO model
        config: Configuration dictionary
        results_dir: Directory to save results
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
    explanation_simplifier = ExplanationSimplifier(
        provider=provider,
        model=model,
        temperature=0.5,
        max_tokens=300
    )
    
    logger.info("Running agents on test emails...")
    feature_extractor = FeatureExtractor()
    
    features_list = []
    labels_list = []
    agent_probs_list = []
    agent_results_list = []
    
    for email in tqdm(test_emails, desc="Processing test emails"):
        agent_results = run_agents_on_email(email, text_agent, url_agent, metadata_agent)
        features = feature_extractor.extract_features(email, agent_results)
        
        features_list.append(features)
        labels_list.append(email['label'])
        agent_probs_list.append({
            'text': agent_results['text_prob'],
            'url': agent_results['url_prob'],
            'meta': agent_results['meta_prob']
        })
        agent_results_list.append(agent_results)
    
    # Load PPO model
    logger.info(f"Loading PPO model from {model_path}...")
    ppo_module = PPOModule(
        feature_dim=feature_extractor.get_feature_dim(),
        config=config['ppo']
    )
    
    # Create dummy environment for loading (needed by stable-baselines3)
    dummy_env = MultiPhishGuardEnv(features_list[:10], labels_list[:10], agent_probs_list[:10])
    ppo_module.load_model(model_path, env=dummy_env)
    
    # Make predictions
    logger.info("Making predictions...")
    predictions = []
    prediction_probas = []
    weights_list = []
    
    for i, features in enumerate(tqdm(features_list, desc="Predicting")):
        # Get weights from PPO
        weights = ppo_module.predict_weights(features)
        
        # --- FIX 1: Convert numpy array to standard list ---
        weights_list.append(weights.tolist()) 
        
        # Calculate final score
        agent_probs = agent_probs_list[i]
        final_score = (
            weights[0] * agent_probs['text'] +
            weights[1] * agent_probs['url'] +
            weights[2] * agent_probs['meta']
        )
        
        # --- FIX 2: Convert numpy.float32 to standard float ---
        prediction_probas.append(float(final_score))
        
        # Threshold for classification
        threshold = config['evaluation'].get('phishing_threshold', 0.5)
        prediction = 1 if final_score >= threshold else 0
        predictions.append(prediction)
    
    # Calculate metrics
    logger.info("Calculating metrics...")
    metrics = calculate_metrics(labels_list, predictions, prediction_probas)
    
    # Baseline comparison
    baseline_metrics = {
        'accuracy': 0.9789,
        'f1': 0.9588
    }
    
    # Print report
    print_metrics_report(metrics, baseline_metrics)
    
    # Save results
    results_dir_path = Path(results_dir)
    results_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    with open(results_dir_path / 'metrics.json', 'w') as f:
        # Note: metrics dict is already JSON-friendly, but we fixed inputs just in case
        json.dump(metrics, f, indent=2)
    
    # Save predictions
    results_data = {
        'predictions': predictions,
        'prediction_probas': prediction_probas,
        'true_labels': labels_list,
        'weights': weights_list, # This is now a list of lists, (Fix 1)
        'metrics': metrics
    }
    
    # This dump will now work
    with open(results_dir_path / 'results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    
    # Plot confusion matrix
    cm_path = results_dir_path / 'confusion_matrix.png'
    plot_confusion_matrix(np.array(metrics['confusion_matrix']), str(cm_path))
    
    logger.info(f"Results saved to {results_dir_path}")
    
    # Generate explanations for a few examples
    logger.info("Generating explanations for sample emails...")
    num_examples = min(5, len(test_emails))
    example_indices = np.random.choice(len(test_emails), num_examples, replace=False)
    
    examples = []
    for idx in example_indices:
        email = test_emails[idx]
        agent_results = agent_results_list[idx]
        pred = predictions[idx]
        true_label = labels_list[idx]
        
        explanation = explanation_simplifier.simplify(
            agent_results['text_agent']['rationale'],
            agent_results['url_agent']['rationale'],
            agent_results['metadata_agent']['rationale']
        )
        
        examples.append({
            'email_id': email['id'],
            'true_label': 'phishing' if true_label == 1 else 'legitimate',
            'predicted_label': 'phishing' if pred == 1 else 'legitimate',
            'final_score': float(prediction_probas[idx]),
            'weights': weights_list[idx], # Already a list, no .tolist() needed
            'explanation': explanation
        })
    
    with open(results_dir_path / 'examples.json', 'w') as f:
        json.dump(examples, f, indent=2)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate MultiPhishGuard model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--model', type=str, default='models/ppo_model_final.zip',
                        help='Path to trained model')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Data directory')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Results directory')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Load test data
    logger.info("Loading test data...")
    test_path = Path(args.data_dir) / 'processed' / 'test.json'
    
    if not test_path.exists():
        logger.error(f"Test data not found: {test_path}")
        logger.error("Please run scripts/preprocess_data.py first")
        return
    
    with open(test_path, 'r') as f:
        test_emails = json.load(f)
    
    logger.info(f"Loaded {len(test_emails)} test emails")
    
    # Evaluate
    metrics = evaluate_model(
        test_emails,
        args.model,
        config,
        args.results_dir
    )
    
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()