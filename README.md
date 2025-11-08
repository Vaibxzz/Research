# MultiPhishGuard

Multi-Agent LLM-Based Phishing Email Detection System with Dynamic Weight Learning using PPO. This repository documents the original PPO implementation and a comparative analysis replacing PPO with XGBoost.

## Overview

MultiPhishGuard is an intelligent system designed to detect phishing emails. Instead of a single model, it uses **multiple AI agents** (Text, URL, Metadata), each analyzing specific clues. The original paper combines these agent outputs using a Proximal Policy Optimization (PPO) reinforcement learning module.

This repository implements the PPO baseline and a **Phase 2 extension** that replaces the complex PPO module with a standard **XGBoost classifier**, which demonstrates superior performance and efficiency.

## Comparative Performance (on 500-Sample Training Set)

The primary goal of this project was to reproduce the baseline and compare it against a simpler ensemble method (XGBoost). Both models were trained on the same 9-dimensional feature vector extracted from the LLM agents.

| Model | Accuracy | F1 Score | Training Time (Post-Features) |
| :--- | :--- | :--- | :--- |
| **PPO (Baseline)** | 98.00% | 98.04% | ~20 Seconds |
| **XGBoost (Phase 2)** | **100.00%** | **100.00%** | **< 10 Seconds** |

The results clearly indicate that a simple XGBoost model not only outperforms the complex PPO module but is also significantly faster to train.

## Project Structure

research/
├── data/
│    ├── raw/
│   ├── processed/         # train.json/test.json (500/100 split)
│   ├── splits/
│   └── features.json      # <-- NEW: Saved features for XGBoost
├── src/
│   ├── agents/
│   ├── ppo/
│   ├── utils/
│   └── train.py           # Original PPO training script
├── config/
│   ├── prompts.py
│   └── config.yaml        # IMPORTANT: Modified for OpenRouter
├── scripts/
│   ├── download_datasets.py
│   ├── preprocess_data.py # IMPORTANT: Modified for 500/100 split
│   ├── evaluate.py
│   ├── extract_features.py  # <-- NEW: Saves features for all models
│   └── train_xgboost.py   # <-- NEW: Trains/evaluates XGBoost
├── models/
├── logs/
├── results/               # <-- NOW CONTAINS PPO *AND* XGBOOST RESULTS
│   ├── metrics.json
│   ├── xgboost_metrics.json
│   └── ...
├── requirements.txt
└── README.md

## Installation

1.  **Clone or navigate to the project directory:**
    ```bash
    cd D:\Research\Research
    ```

2.  **Create virtual environment:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate.ps1  # On Windows PowerShell
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    
    # Install PPO extras (like TensorBoard)
    pip install stable-baselines3[extra]
    
    # Install XGBoost
    pip install xgboost
    ```

4.  **Set up environment variables:**
    * Create a new file named `.env` in the root directory.
    * Add your API key. (We recommend OpenRouter to avoid Google's safety filters).
    ```
    OPENAI_API_KEY="sk-or-xxxxxxxxxxxxxxx"
    ```

## Configuration (Crucial for OpenRouter)

To avoid Google's safety filters blocking phishing content, this project is configured to use **OpenRouter**.

1.  **`config/config.yaml`**:
    Ensure the `llm` section points to the "openai" provider (which OpenRouter uses) and a model like Mistral.
    ```yaml
    llm:
      provider: "openai"
      model: "mistralai/mistral-7b-instruct"
    ```

2.  **`src/agents/base_agent.py`**:
    You **must** add the `base_url` to the `OpenAI` client (around line 59) to point it to OpenRouter's API.
    ```python
    self.client = OpenAI(
        api_key=api_key,
        base_url="[https://openrouter.ai/api/v1](https://openrouter.ai/api/v1)"  # <-- ADD THIS LINE
    )
    ```

## Datasets & "Mini-Dataset"

The full project uses 6 large datasets. Due to API costs and 70+ hour processing times, we use a "mini-dataset" for rapid comparison.

The `scripts/preprocess_data.py` script has been modified to sample:
* **Training Set:** 500 emails (250 phishing, 250 legit)
* **Test Set:** 100 emails (50 phishing, 50 legit)

## Usage: PPO vs. XGBoost Workflow

This workflow reproduces the baseline and runs the comparative analysis.

### Part 1: Run the PPO Baseline

This part runs the original PPO model to get the 98% accuracy baseline.

1.  **Download Datasets:**
    ```bash
    python scripts/download_datasets.py
    ```
    *(Note: You may need to manually extract `spamassassin_hard_ham.tar.bz2` into the `data/raw/spamassassin` folder as shown in our logs).*

2.  **Create Mini-Dataset:**
    *(This script is already modified to create the 500/100 split).*
    ```bash
    python scripts/preprocess_data.py
    ```

3.  **Train PPO Model (Baseline):**
    *(This is a 3-4 hour API-intensive task. It processes 500 emails and trains the PPO model).*
    ```bash
    python src/train.py
    ```
    *Note: We run without `--use_adversarial` to save API calls.*

4.  **Evaluate PPO Model (Baseline):**
    *(This is a 20-30 minute API-intensive task. It processes 100 test emails and saves results to `results/metrics.json`)*.
    ```bash
    python scripts/evaluate.py --model models/ppo_model_final.zip
    ```

### Part 2: Run the XGBoost Comparison

This part re-uses the agent logic but saves the features to compare models.

5.  **Extract All Features to File:**
    *(This is a 3-4 hour API-intensive task. It runs the agents on all 600 emails and saves the features to `data/features.json` to prevent future API calls).*
    ```bash
    python scripts/extract_features.py
    ```

6.  **Train & Evaluate XGBoost:**
    *(This is a **10-second task**. It uses the saved features from `data/features.json`. No API calls are made).*
    ```bash
    python scripts/train_xgboost.py
    ```
    *This script will print the 100% accuracy result and save it to `results/xgboost_metrics.json`.*

## System Architecture
(This section can be copied from the original file)

...

## Phase 2 Extension (Completed)

* **Option 2 (Chosen): Replace PPO with an alternative algorithm.**
* **Method:** We replaced the PPO module with an `XGBoost` classifier.
* **Result:** XGBoost (100% Accuracy) outperformed PPO (98% Accuracy), proving the PPO module's complexity is not necessary.

## Results

Evaluation results are saved in `results/`:
* `metrics.json`: PPO baseline performance
* `confusion_matrix.png`: PPO baseline visualization
* `examples.json`: PPO baseline sample explanations
* **`xgboost_metrics.json`**: XGBoost comparison performance
* **`xgboost_confusion_matrix.png`**: XGBoost visualization
* **`xgboost_examples.json`**: XGBoost sample predictions

## Troubleshooting (New Issues Found)

1.  **`ImportError: tensorboard is not installed`**
    * **Fix:** The PPO library requires extras. Run `pip install stable-baselines3[extra]`.

2.  **`TypeError: Object of type float32 is not JSON serializable`**
    * **Fix:** Occurs in `evaluate.py`. The `numpy.float32` type must be cast to a standard `float()` before saving to JSON.

3.  **`JSONDecodeError: Unterminated string`**
    * **Fix:** The LLM is providing verbose, non-JSON explanations. Modify `src/agents/base_agent.py` to add a prompt instruction: `"\n\nIMPORTANT: Respond ONLY with valid JSON..."`.

4.  **`google/gemini... is not a valid model ID` (OpenRouter)**
    * **Fix:** The model name in `config.yaml` is wrong. Use the correct OpenRouter ID, e.g., `mistralai/mistral-7b-instruct`.

5.  **`Safety filter blocking` (Google Gemini)**
    * **Fix:** Google's API blocks phishing content. Switch to a provider like **OpenRouter** (see `Configuration` section).