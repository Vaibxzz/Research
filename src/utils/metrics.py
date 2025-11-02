"""
Metrics calculation for evaluation
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_metrics(y_true: List[int], y_pred: List[int], 
                     y_pred_proba: List[float] = None) -> Dict:
    """
    Calculate classification metrics
    
    Args:
        y_true: True labels (0=legitimate, 1=phishing)
        y_pred: Predicted labels
        y_pred_proba: Prediction probabilities (optional)
    
    Returns:
        Dictionary with metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    # Add ROC AUC if probabilities provided
    if y_pred_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        except:
            metrics['roc_auc'] = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    metrics['true_negatives'] = int(cm[0][0])
    metrics['false_positives'] = int(cm[0][1])
    metrics['false_negatives'] = int(cm[1][0])
    metrics['true_positives'] = int(cm[1][1])
    
    return metrics


def print_metrics_report(metrics: Dict, baseline_metrics: Dict = None):
    """Print formatted metrics report"""
    print("\n" + "="*50)
    print("EVALUATION METRICS")
    print("="*50)
    print(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f} ({metrics['f1']*100:.2f}%)")
    
    if 'roc_auc' in metrics:
        print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
    
    print("\nConfusion Matrix:")
    print(f"  True Negatives:  {metrics['true_negatives']}")
    print(f"  False Positives: {metrics['false_positives']}")
    print(f"  False Negatives: {metrics['false_negatives']}")
    print(f"  True Positives:  {metrics['true_positives']}")
    
    if baseline_metrics:
        print("\n" + "="*50)
        print("BASELINE COMPARISON")
        print("="*50)
        print(f"Target Accuracy: 97.89%")
        print(f"Your Accuracy:   {metrics['accuracy']*100:.2f}%")
        print(f"Difference:      {abs(97.89 - metrics['accuracy']*100):.2f}%")
        print(f"\nTarget F1:       95.88%")
        print(f"Your F1:         {metrics['f1']*100:.2f}%")
        print(f"Difference:      {abs(95.88 - metrics['f1']*100):.2f}%")
    
    print("="*50 + "\n")


def plot_confusion_matrix(cm: np.ndarray, save_path: str = None):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Legitimate', 'Phishing'],
                yticklabels=['Legitimate', 'Phishing'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def plot_roc_curve(y_true: List[int], y_pred_proba: List[float], 
                   save_path: str = None):
    """Plot ROC curve"""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()





