# MultiPhishGuard Testing Guide

## How to Test the System

### Quick Test (Recommended First Step)

Test the system on a small sample of emails:

```bash
python3 scripts/test_system.py --num_samples 10
```

This will:
- Load 10 emails from your test set (5 phishing, 5 legitimate)
- Run all 3 agents (Text, URL, Metadata) on each email
- Show confidence scores and predictions
- Calculate accuracy, precision, recall, F1

### Test Single Email

Test one specific email:

```bash
python3 scripts/test_system.py --single_email "Your email text here"
```

### Full Training Test

Before running full training, you can verify the system works:

1. **Test Preprocessing** (if not done):
```bash
python3 scripts/preprocess_data.py
```

2. **Test with Small Sample**:
```bash
python3 scripts/test_system.py --num_samples 5
```

3. **If tests pass, proceed with training**:
```bash
python3 src/train.py --use_adversarial
```

### Understanding Test Results

- **Verdict**: Agent's decision (phishing/legitimate)
- **Confidence**: How certain the agent is (0.0-1.0)
- **Phishing Probability**: Probability that email is phishing
- **Final Score**: Weighted combination of all agents
- **Result**: ✓ = Correct, ✗ = Incorrect

### Current Status

⚠️ **Note**: Currently, some API responses may be blocked by Google's safety filters when analyzing actual phishing content. This is expected behavior for security reasons. The system handles this gracefully by:
- Detecting blocked responses
- Using default values (0.5 confidence)
- Continuing processing

For production use, you may need to:
1. Adjust safety settings in Google Cloud Console
2. Use a different API key with different restrictions
3. Pre-process emails to sanitize content

### Troubleshooting

1. **API Key Error**: Ensure `GOOGLE_API_KEY` is set in `.env`
2. **Module Not Found**: Run `pip3 install -r requirements.txt`
3. **Safety Filter Blocks**: This is normal - system handles it automatically
4. **Slow Responses**: API calls take time - be patient during testing

