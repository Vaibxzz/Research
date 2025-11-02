"""Quick test to verify agents work"""
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, '.')

# Load environment variables from .env file
env_path = Path('.') / '.env'
if env_path.exists():
    load_dotenv(env_path)
elif not os.getenv('GOOGLE_API_KEY'):
    raise ValueError(
        "GOOGLE_API_KEY not found. Please set it in .env file or environment variable.\n"
        "Create a .env file with: GOOGLE_API_KEY=your_key_here"
    )

from src.agents.text_agent import TextAgent

agent = TextAgent(provider='google', model='gemini-2.5-flash')

# Test with simple email
email = "Hello, this is a normal business email about our meeting next week."
result = agent.analyze(email)

print("="*50)
print("QUICK TEST RESULTS")
print("="*50)
print(f"Verdict: {result['verdict']}")
print(f"Confidence: {result['confidence']}")
print(f"Rationale: {result['rationale'][:150]}...")
print()
if result['confidence'] == 0.5 and 'blocked' in result['rationale'].lower():
    print("⚠️  Warning: Response was blocked by safety filters")
    print("The API is working but some prompts/content trigger safety filters")
else:
    print("✅ Agent is working correctly!")

