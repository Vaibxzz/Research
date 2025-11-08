# MultiPhishGuard - PPO vs. XGBoost Comparison Guide

This guide explains how to run the final, comparative analysis of the **PPO Baseline** vs. the **XGBoost Model**. This guide assumes you have already run the 4-hour `scripts/extract_features.py` script at least once.

## Prerequisites

This guide requires two key components to be present:

1.  **`models/ppo_model_final.zip`**: The PPO model trained on the 500-sample "mini-dataset". This is created by `python src/train.py`.
2.  **`data/features.json`**: The saved agent features for all 600 samples. This is created by `python scripts/extract_features.py`.

If you do not have these files, you must run the scripts from the `README.md`'s "Usage" section first.

## How to Run the Final Comparison

### 1. Test the XGBoost Model (10-Second Test)

This is the fastest test. It uses the pre-saved features and requires **no API calls**. It validates our Phase 2 conclusion.

```bash
# Make sure your venv is active
.\venv\Scripts\activate.ps1

# Run the XGBoost script
python scripts/train_xgboost.py