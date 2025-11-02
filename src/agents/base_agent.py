"""
Base Agent Class
Common functionality for all LLM-based agents
Supports both OpenAI and Google Gemini APIs
"""

import json
import os
import time
import logging
from typing import Dict, Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Try importing both providers
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI library not available")

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    logger.warning("Google GenerativeAI library not available")


class BaseAgent:
    """Base class for all LLM agents"""
    
    def __init__(self, provider: str = "google", model: str = None, temperature: float = 0.3, max_tokens: int = 500):
        """
        Initialize base agent
        
        Args:
            provider: "openai" or "google"
            model: Model name (defaults based on provider)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        self.provider = provider.lower()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = 3
        self.retry_delay = 1.0
        
        if self.provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI library not installed. Install with: pip install openai")
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            
            self.client = OpenAI(api_key=api_key)
            self.model = model or "gpt-4"
            
        elif self.provider == "google":
            if not GOOGLE_AVAILABLE:
                raise ImportError("Google GenerativeAI library not installed. Install with: pip install google-generativeai")
            
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY or GOOGLE_GEMINI_API_KEY not found in environment variables")
            
            genai.configure(api_key=api_key)
            
            # Configure safety settings at model initialization (needed for phishing analysis)
            import google.generativeai.types as genai_types
            safety_config = {
                genai_types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai_types.HarmBlockThreshold.BLOCK_NONE,
                genai_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai_types.HarmBlockThreshold.BLOCK_NONE,
                genai_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai_types.HarmBlockThreshold.BLOCK_NONE,
                genai_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai_types.HarmBlockThreshold.BLOCK_NONE,
            }
            
            self.model = genai.GenerativeModel(model or "gemini-2.5-flash", safety_settings=safety_config)
            
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'google'")
    
    def call_llm(self, prompt: str, response_format: Optional[Dict] = None) -> str:
        """
        Call LLM with prompt
        
        Args:
            prompt: Prompt string
            response_format: Optional JSON schema for structured output
        
        Returns:
            Response text
        """
        for attempt in range(self.max_retries):
            try:
                if self.provider == "openai":
                    kwargs = {
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": self.temperature,
                        "max_tokens": self.max_tokens
                    }
                    
                    # Add response format if specified (for structured outputs)
                    if response_format:
                        kwargs["response_format"] = response_format
                    
                    response = self.client.chat.completions.create(**kwargs)
                    return response.choices[0].message.content
                
                elif self.provider == "google":
                    # For Google Gemini, add JSON format instruction in prompt if needed
                    if response_format:
                        prompt_with_format = prompt + "\n\nIMPORTANT: Respond ONLY with valid JSON in the requested format. Do not include any text outside the JSON."
                    else:
                        prompt_with_format = prompt
                    
                    generation_config = {
                        "temperature": self.temperature,
                        "max_output_tokens": self.max_tokens,
                    }
                    
                    # Also pass safety settings per request (in addition to model init)
                    import google.generativeai.types as genai_types
                    safety_settings = {
                        genai_types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai_types.HarmBlockThreshold.BLOCK_NONE,
                        genai_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai_types.HarmBlockThreshold.BLOCK_NONE,
                        genai_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai_types.HarmBlockThreshold.BLOCK_NONE,
                        genai_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai_types.HarmBlockThreshold.BLOCK_NONE,
                    }
                    
                    # Generate content with safety settings
                    response = self.model.generate_content(
                        prompt_with_format,
                        generation_config=generation_config,
                        safety_settings=safety_settings
                    )
                    
                    # Handle response - check candidates first (before trying response.text)
                    if hasattr(response, 'candidates') and response.candidates:
                        candidate = response.candidates[0]
                        
                        # Check finish_reason - handle both enum and int
                        finish_reason = candidate.finish_reason if hasattr(candidate, 'finish_reason') else None
                        finish_reason_value = finish_reason.value if hasattr(finish_reason, 'value') else (finish_reason if isinstance(finish_reason, int) else None)
                        
                        # SAFETY = 2 means blocked
                        if finish_reason_value == 2:
                            logger.warning("Response blocked by safety filters, using default")
                            return '{"verdict": "legitimate", "confidence": 0.5, "rationale": "Response blocked by safety filters"}'
                        
                        # If successful (STOP = 1), extract text from candidate
                        if finish_reason_value == 1 or finish_reason is None:
                            # Try to get text from candidate
                            if hasattr(candidate, 'content') and candidate.content:
                                if hasattr(candidate.content, 'parts') and candidate.content.parts:
                                    text_parts = []
                                    for part in candidate.content.parts:
                                        if hasattr(part, 'text') and part.text:
                                            text_parts.append(part.text)
                                    if text_parts:
                                        return ''.join(text_parts)
                    
                    # Fallback: try direct text access
                    try:
                        return response.text
                    except (ValueError, AttributeError):
                        pass
                    
                    # If all else fails, return default
                    logger.warning("Could not extract text from response, using default")
                    return '{"verdict": "legitimate", "confidence": 0.5, "rationale": "Could not extract response from model"}'
                
            except Exception as e:
                error_str = str(e)
                # Handle API rate limits - extract retry delay if available
                if "Quota exceeded" in error_str or "rate limit" in error_str.lower():
                    # Try to extract retry delay from error message
                    import re
                    delay_match = re.search(r'retry in ([\d.]+)s', error_str, re.IGNORECASE)
                    if delay_match:
                        wait_time = float(delay_match.group(1)) + 5  # Add buffer
                    else:
                        wait_time = 60  # Default 60 seconds for rate limits
                    
                    if attempt < self.max_retries - 1:
                        logger.warning(f"API rate limit hit (attempt {attempt + 1}/{self.max_retries}). Waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"API rate limit exceeded after {self.max_retries} attempts")
                        raise
                
                # Handle other errors
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(f"LLM call failed (attempt {attempt + 1}/{self.max_retries}): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"LLM call failed after {self.max_retries} attempts: {e}")
                    raise
    
    def parse_json_response(self, response: str) -> Dict:
        """
        Parse JSON response from LLM
        
        Args:
            response: LLM response string
        
        Returns:
            Parsed JSON dictionary
        """
        try:
            # Try to extract JSON from response if wrapped in text
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}\nResponse: {response}")
            # Return default structure
            return {
                'verdict': 'legitimate',
                'confidence': 0.5,
                'rationale': f'Error parsing response: {str(e)}'
            }
    
    def validate_response(self, response_dict: Dict) -> bool:
        """
        Validate agent response structure
        
        Args:
            response_dict: Parsed response dictionary
        
        Returns:
            True if valid, False otherwise
        """
        required_keys = ['verdict', 'confidence', 'rationale']
        if not all(key in response_dict for key in required_keys):
            return False
        
        if response_dict['verdict'] not in ['phishing', 'legitimate']:
            return False
        
        try:
            conf = float(response_dict['confidence'])
            if not 0.0 <= conf <= 1.0:
                return False
        except (ValueError, TypeError):
            return False
        
        return True


