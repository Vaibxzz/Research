"""
Feature Extractor for PPO
Extracts features from emails for reinforcement learning
"""

import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract features from email for PPO input"""
    
    def __init__(self):
        self.feature_names = [
            'url_count',
            'phishing_keyword_count',
            'sender_domain_reputation',  # Placeholder - normalized
            'spf_pass',
            'dkim_pass',
            'dmarc_pass',
            'text_agent_confidence',
            'url_agent_confidence',
            'metadata_agent_confidence'
        ]
    
    def extract_features(self, email_data: Dict, agent_results: Dict) -> np.ndarray:
        """
        Extract feature vector from email data and agent results
        
        Args:
            email_data: Dictionary with parsed email data
                - urls: List of URLs
                - phishing_keyword_count: int
                - metadata: Dict with spf_pass, dkim_pass, dmarc_pass, sender_domain
            agent_results: Dictionary with agent outputs
                - text_agent: Dict with 'confidence' or 'phishing_probability'
                - url_agent: Dict with 'confidence' or 'phishing_probability'
                - metadata_agent: Dict with 'confidence' or 'phishing_probability'
        
        Returns:
            Normalized feature vector (numpy array)
        """
        features = []
        
        # 1. URL count (normalize by max, assume max 10 URLs)
        url_count = len(email_data.get('urls', []))
        features.append(min(url_count / 10.0, 1.0))
        
        # 2. Phishing keyword count (normalize by max, assume max 20 keywords)
        keyword_count = email_data.get('phishing_keyword_count', 0)
        features.append(min(keyword_count / 20.0, 1.0))
        
        # 3. Sender domain reputation (placeholder - binary for now)
        # In real implementation, could use reputation API
        metadata = email_data.get('metadata', {})
        sender_domain = metadata.get('sender_domain', '')
        # Simple heuristic: common domains = lower risk
        common_domains = ['gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com', 'company.com']
        domain_reputation = 0.5  # Default neutral
        if sender_domain in common_domains:
            domain_reputation = 0.3  # Slightly lower risk
        elif not sender_domain:
            domain_reputation = 0.7  # Higher risk if no domain
        features.append(domain_reputation)
        
        # 4. SPF pass (binary)
        features.append(1.0 if metadata.get('spf_pass', False) else 0.0)
        
        # 5. DKIM pass (binary)
        features.append(1.0 if metadata.get('dkim_pass', False) else 0.0)
        
        # 6. DMARC pass (binary)
        features.append(1.0 if metadata.get('dmarc_pass', False) else 0.0)
        
        # 7-9. Agent confidence scores
        text_result = agent_results.get('text_agent', {})
        text_conf = text_result.get('phishing_probability', 
                                   text_result.get('confidence', 0.5))
        features.append(float(text_conf))
        
        url_result = agent_results.get('url_agent', {})
        url_conf = url_result.get('phishing_probability',
                                 url_result.get('confidence', 0.5))
        features.append(float(url_conf))
        
        meta_result = agent_results.get('metadata_agent', {})
        meta_conf = meta_result.get('phishing_probability',
                                   meta_result.get('confidence', 0.5))
        features.append(float(meta_conf))
        
        return np.array(features, dtype=np.float32)
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        return self.feature_names.copy()
    
    def get_feature_dim(self) -> int:
        """Get feature vector dimension"""
        return len(self.feature_names)





