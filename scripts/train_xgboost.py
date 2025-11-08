"""
XGBoost Training & Evaluation Script
-----------------------------------
Loads the pre-extracted features and trains a simple
XGBoost model for comparison with PPO.
"""

import json
import logging
import argparse
from pathlib import Path
import numpy as np
import xgboost as xgb
import sys

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import metric calculators from your project
from src.utils.metrics import calculate_metrics, print_metrics_report, plot_confusion_matrix

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate XGBoost on extracted features')
    parser.add_argument('--feature_file', type=str, default='data/features.json',
                        help='Path to the .json file containing extracted features')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory to save XGBoost results')
    
    args = parser.parse_args()
    
    # --- 1. Load Data ---
    logger.info(f"Loading features from {args.feature_file}...")
    with open(args.feature_file, 'r') as f:
        data = json.load(f)
        
    # Convert data to numpy arrays (XGBoost prefers this)
    X_train = np.array(data['train_X'])
    y_train = np.array(data['train_y'])
    X_test = np.array(data['test_X'])
    y_test = np.array(data['test_y'])
    
    logger.info(f"Loaded {len(X_train)} training samples and {len(X_test)} test samples.")
    
    # --- 2. Train XGBoost Model ---
    logger.info("Training XGBoost model...")
    
    # Initialize XGBClassifier
    model = xgb.XGBClassifier(
        objective='binary:logistic',  # For binary classification (phishing/legit)
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )
    
    # Train the model
    model.fit(X_train, y_train)
    logger.info("XGBoost training complete!")
    
    # --- 3. Evaluate Model ---
    logger.info("Evaluating XGBoost model on test set...")
    
    # Get predictions (0 or 1)
    predictions = model.predict(X_test)
    
    # Get prediction probabilities (score between 0.0 and 1.0)
    prediction_probas = model.predict_proba(X_test)[:, 1] # Get prob for 'phishing'
    
    # Calculate metrics
    logger.info("Calculating metrics...")
    
    # Convert from numpy types to standard python types for JSON
    predictions_list = [int(p) for p in predictions]
    probas_list = [float(p) for p in prediction_probas]
    labels_list = [int(l) for l in y_test]
    
    metrics = calculate_metrics(labels_list, predictions_list, probas_list)
    
    # --- 4. Print Report ---
    # We compare XGBoost against the PPO results
    baseline_metrics = {
        'accuracy': 0.9800, # Your PPO score
        'f1': 0.9804      # Your PPO score
    }
    logger.info("---[ XGBoost vs. PPO (Baseline) ]---")
    print_metrics_report(metrics, baseline_metrics)
    
    # --- 5. Save All Results (as promised) ---
    logger.info(f"Saving XGBoost results to {args.results_dir}...")
    results_dir_path = Path(args.results_dir)
    results_dir_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Save Metrics
    with open(results_dir_path / 'xgboost_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
        
    # 2. Save Confusion Matrix
    cm_path = results_dir_path / 'xgboost_confusion_matrix.png'
    plot_confusion_matrix(np.array(metrics['confusion_matrix']), str(cm_path))
    
    # 3. Save full results
    results_data = {
        'predictions': predictions_list,
        'prediction_probas': probas_list,
        'true_labels': labels_list,
        'metrics': metrics
    }
    with open(results_dir_path / 'xgboost_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
        
    # 4. Save simplified examples (we can't get explanations without API calls)
    # We load the original test.json just to get email IDs
    test_json_path = Path('data/processed/test.json')
    examples = []
    if test_json_path.exists():
        with open(test_json_path, 'r') as f:
            test_emails = json.load(f)
        for i, email in enumerate(test_emails):
            examples.append({
                'email_id': email['id'],
                'true_label': 'phishing' if labels_list[i] == 1 else 'legitimate',
                'predicted_label': 'phishing' if predictions_list[i] == 1 else 'legitimate',
                'final_score': probas_list[i]
            })
    
    with open(results_dir_path / 'xgboost_examples.json', 'w') as f:
        json.dump(examples, f, indent=2)
        
    logger.info("XGBoost evaluation complete!")

if __name__ == "__main__":
    main()