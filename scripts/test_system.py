"""
Test Script for MultiPhishGuard
Tests the system on a small subset of emails to verify everything works
"""

import os
import sys
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.text_agent import TextAgent
from src.agents.url_agent import URLAgent
from src.agents.metadata_agent import MetadataAgent
from src.utils.email_parser import EmailParser
from src.utils.metrics import calculate_metrics, print_metrics_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_single_email(email_text: str, label: int):
    """Test a single email through all agents"""
    print("\n" + "="*70)
    print(f"Testing Email (Label: {'PHISHING' if label == 1 else 'LEGITIMATE'})")
    print("="*70)
    print(f"Email Text: {email_text[:200]}...")
    print()
    
    # Parse email
    parser = EmailParser()
    parsed = parser.parse_email_string(email_text) if email_text else {
        'body_text': email_text or '',
        'urls': [],
        'metadata': {}
    }
    
    # Initialize agents
    text_agent = TextAgent(provider='google', model='gemini-2.5-flash')
    url_agent = URLAgent(provider='google', model='gemini-2.5-flash')
    metadata_agent = MetadataAgent(provider='google', model='gemini-2.5-flash')
    
    # Run agents
    print("Running Text Agent...")
    text_result = text_agent.analyze(parsed['body_text'])
    print(f"  Verdict: {text_result['verdict']}")
    print(f"  Confidence: {text_result['confidence']:.2f}")
    print(f"  Phishing Probability: {text_result.get('phishing_probability', 0.5):.2f}")
    
    print("\nRunning URL Agent...")
    url_result = url_agent.analyze(parsed['urls'])
    print(f"  Verdict: {url_result['verdict']}")
    print(f"  Confidence: {url_result['confidence']:.2f}")
    print(f"  Phishing Probability: {url_result.get('phishing_probability', 0.5):.2f}")
    
    print("\nRunning Metadata Agent...")
    metadata_result = metadata_agent.analyze(parsed['metadata'])
    print(f"  Verdict: {metadata_result['verdict']}")
    print(f"  Confidence: {metadata_result['confidence']:.2f}")
    print(f"  Phishing Probability: {metadata_result.get('phishing_probability', 0.5):.2f}")
    
    # Calculate weighted average (simple equal weights for testing)
    final_score = (
        0.33 * text_result.get('phishing_probability', 0.5) +
        0.33 * url_result.get('phishing_probability', 0.5) +
        0.34 * metadata_result.get('phishing_probability', 0.5)
    )
    
    prediction = 1 if final_score >= 0.5 else 0
    correct = "✓" if prediction == label else "✗"
    
    print("\n" + "-"*70)
    print(f"Final Score: {final_score:.2f}")
    print(f"Prediction: {'PHISHING' if prediction == 1 else 'LEGITIMATE'}")
    print(f"True Label: {'PHISHING' if label == 1 else 'LEGITIMATE'}")
    print(f"Result: {correct}")
    print("="*70)
    
    return {
        'prediction': prediction,
        'true_label': label,
        'final_score': final_score,
        'correct': prediction == label,
        'text_prob': text_result.get('phishing_probability', 0.5),
        'url_prob': url_result.get('phishing_probability', 0.5),
        'meta_prob': metadata_result.get('phishing_probability', 0.5)
    }


def test_on_sample_data(num_samples: int = 10):
    """Test on a small sample from the test set"""
    print(f"\n{'='*70}")
    print(f"Testing MultiPhishGuard on {num_samples} sample emails")
    print(f"{'='*70}\n")
    
    # Load test data
    test_path = Path('data/processed/test.json')
    if not test_path.exists():
        logger.error(f"Test data not found: {test_path}")
        logger.error("Please run scripts/preprocess_data.py first")
        return
    
    with open(test_path, 'r') as f:
        test_emails = json.load(f)
    
    # Get balanced sample
    import random
    random.seed(42)
    
    phishing_emails = [e for e in test_emails if e['label'] == 1]
    legitimate_emails = [e for e in test_emails if e['label'] == 0]
    
    sample_phishing = random.sample(phishing_emails, min(num_samples // 2, len(phishing_emails)))
    sample_legitimate = random.sample(legitimate_emails, min(num_samples // 2, len(legitimate_emails)))
    
    sample_emails = sample_phishing + sample_legitimate
    random.shuffle(sample_emails)
    
    # Test each email
    results = []
    for i, email in enumerate(sample_emails, 1):
        print(f"\n[Test {i}/{len(sample_emails)}]")
        result = test_single_email(
            email.get('body_text', email.get('original_text', '')),
            email['label']
        )
        results.append(result)
    
    # Calculate metrics
    predictions = [r['prediction'] for r in results]
    true_labels = [r['true_label'] for r in results]
    probas = [r['final_score'] for r in results]
    
    metrics = calculate_metrics(true_labels, predictions, probas)
    
    print("\n\n" + "="*70)
    print("TEST RESULTS SUMMARY")
    print("="*70)
    print_metrics_report(metrics)
    
    print("\nAgent Performance:")
    avg_text_prob = sum(r['text_prob'] for r in results) / len(results)
    avg_url_prob = sum(r['url_prob'] for r in results) / len(results)
    avg_meta_prob = sum(r['meta_prob'] for r in results) / len(results)
    
    print(f"  Average Text Agent Probability: {avg_text_prob:.2f}")
    print(f"  Average URL Agent Probability: {avg_url_prob:.2f}")
    print(f"  Average Metadata Agent Probability: {avg_meta_prob:.2f}")
    
    print("\n" + "="*70)
    print("✅ System test complete!")
    print("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test MultiPhishGuard system')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of emails to test (default: 10)')
    parser.add_argument('--single_email', type=str, default=None,
                       help='Test a single email text')
    
    args = parser.parse_args()
    
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
    
    if args.single_email:
        # Test single email
        test_single_email(args.single_email, 1)  # Assume phishing for single test
    else:
        # Test on sample data
        test_on_sample_data(args.num_samples)

