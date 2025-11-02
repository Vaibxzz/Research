# MultiPhishGuard

Multi-Agent LLM-Based Phishing Email Detection System with Dynamic Weight Learning using PPO

## Overview

MultiPhishGuard is an intelligent system designed to detect phishing emails — messages that attempt to steal user information or passwords.

Instead of a single model, it uses **multiple AI agents**, each performing a specific role, similar to a team of detectives analyzing different clues. The system combines specialized LLM agents with a Proximal Policy Optimization (PPO) reinforcement learning module for dynamic weight optimization.

**Key Strengths:**
- ✅ Considers multiple aspects of an email (text, URL, metadata)
- ✅ Adaptive weight system using reinforcement learning
- ✅ Human-understandable explanations
- ✅ Learns from new phishing patterns

**Performance:**
- Accuracy: ~98%
- Precision: 92% (fewer false positives)
- Recall: 99.8% (detects nearly all phishing emails)
- Performs better than single-agent or older models such as RoBERTa

## Project Structure

```
research/
├── data/
│   ├── raw/              # Downloaded raw datasets
│   ├── processed/        # Preprocessed email data
│   └── splits/           # Train/test splits
├── src/
│   ├── agents/           # LLM-based agents
│   │   ├── base_agent.py
│   │   ├── text_agent.py
│   │   ├── url_agent.py
│   │   ├── metadata_agent.py
│   │   ├── adversarial_agent.py
│   │   └── explanation_simplifier.py
│   ├── ppo/              # PPO module
│   │   ├── ppo_module.py
│   │   └── feature_extractor.py
│   ├── utils/            # Utilities
│   │   ├── email_parser.py
│   │   ├── dataset_loader.py
│   │   └── metrics.py
│   └── train.py          # Training script
├── config/
│   ├── prompts.py        # LLM prompts
│   └── config.yaml       # Configuration
├── scripts/
│   ├── download_datasets.py
│   ├── preprocess_data.py
│   └── evaluate.py
├── notebooks/
│   └── analysis.ipynb
├── models/               # Trained models
├── logs/                # Training logs
├── results/             # Evaluation results
├── requirements.txt
└── README.md
```

## Installation

1. **Clone or navigate to the project directory:**
```bash
cd /Users/vaibhavsrivastava/Documents/research
```

2. **Create virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables:**
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

## Datasets

The system requires 6 datasets:

**Phishing (979 emails):**
- Nazario Phishing Corpus (402 emails)
- Nigerian Fraud Dataset (577 emails)

**Legitimate (3,000 emails):**
- Enron-Spam (1,500 ham emails)
- SpamAssassin (500 hard ham emails)
- TREC 2007 (500 ham emails)
- CEAS 2008 (500 ham emails)

## Usage

### Step 1: Download Datasets

```bash
python scripts/download_datasets.py
```

**Note:** Some datasets (Enron, SpamAssassin, TREC 2007, CEAS 2008) require manual download. The script will provide instructions.

### Step 2: Preprocess Data

```bash
python scripts/preprocess_data.py
```

This will:
- Parse emails and extract components (body text, URLs, metadata)
- Create train/test splits (80/20 by default)
- Save processed data to `data/processed/`

### Step 3: Train Model

```bash
python src/train.py --use_adversarial
```

This will:
- Run all three agents on training emails
- Generate adversarial variants for training augmentation
- Train PPO model to learn optimal weights
- Save model to `models/ppo_model_final.zip`

### Step 4: Evaluate Model

```bash
python scripts/evaluate.py --model models/ppo_model_final.zip
```

This will:
- Run evaluation on test set
- Calculate metrics (accuracy, precision, recall, F1)
- Compare with baseline (97.89% accuracy, 95.88% F1)
- Generate confusion matrix and examples
- Save results to `results/`

## Configuration

Edit `config/config.yaml` to customize:
- LLM model and parameters
- PPO hyperparameters
- Training settings
- File paths

## System Architecture

### Agents

1. **Text Agent**: Reads email text and checks for suspicious/urgent tones
   - Example: "Click here to claim your reward!" → likely phishing
2. **URL Agent**: Analyzes links to detect malicious or fake websites
3. **Metadata Agent**: Examines sender and header details
   - Example: Email claims from "@bank.com" but reply address is "@gmail.com" → mismatch detected
4. **Explanation Agent**: Provides clear, human-friendly explanations of why emails are marked safe or unsafe
5. **Adversarial Agent**: Creates slightly altered fake emails to help the system learn from new, smarter phishing attempts

### Decision Process (PPO Module)

Each agent gives a confidence score (e.g., "80% phishing").

The Reinforcement Learning model (PPO) then decides how much to trust each agent based on the email type:

- **If the email contains many links** → more weight is given to the URL Agent
- **If it is mainly text-based** → the Text Agent's opinion is prioritized

This makes the system **adaptive and context-aware**.

**Technical Details:**
- **State**: Feature vector (URL count, keywords, domain reputation, agent confidences)
- **Action**: 3 weights [w_text, w_url, w_meta] that sum to 1
- **Reward**: +1 for correct prediction, -1 for incorrect
- **Output**: Final score = (w_text × p_text) + (w_url × p_url) + (w_meta × p_meta)

### Supporting Agents

- **Adversarial Agent**: Generates phishing variants for training augmentation
- **Explanation Simplifier**: Combines agent rationales into user-friendly explanations

## Performance Targets (Paper Results)

- **Accuracy**: ~98% (Baseline: 97.89%)
- **Precision**: 92% (fewer false positives)
- **Recall**: 99.8% (detects nearly all phishing emails)
- **F1 Score**: 95.88%

## Phase 2 Extension

The system supports Phase 2 extensions:
- **Option 1**: Cross-lingual dataset integration
- **Option 2**: Replace PPO with alternative algorithm (A2C, Gradient Boosting, etc.)

## Results

Evaluation results are saved in `results/`:
- `metrics.json`: Performance metrics
- `results.json`: Detailed predictions and weights
- `confusion_matrix.png`: Confusion matrix visualization
- `examples.json`: Sample predictions with explanations

## Troubleshooting

1. **API Key Error**: Ensure `OPENAI_API_KEY` is set in `.env`
2. **Dataset Not Found**: Check that datasets are downloaded and placed in `data/raw/`
3. **Memory Error**: Reduce batch size or number of training samples in config

## License

This project is for research purposes.

## Citation

If you use this implementation, please cite the original MultiPhishGuard paper.





