"""
Dataset Loader
Loads and processes datasets from various sources
"""

import os
import re
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Load datasets from various formats"""
    
    def __init__(self, data_dir: str):
        """
        Initialize dataset loader
        
        Args:
            data_dir: Directory containing datasets
        """
        self.data_dir = data_dir
    
    def load_phishing_dataset(self, dataset_name: str, file_path: str) -> List[Dict]:
        """
        Load phishing dataset
        
        Args:
            dataset_name: Name of dataset (e.g., 'nazario', 'nigerian')
            file_path: Path to dataset file
        
        Returns:
            List of email dictionaries with 'text' and 'label' keys
        """
        emails = []
        
        if not os.path.exists(file_path):
            logger.warning(f"Dataset file not found: {file_path}")
            return emails
        
        try:
            # Try CSV format first
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
                # Common column names for phishing datasets
                text_col = None
                for col in ['text', 'body', 'content', 'email', 'message']:
                    if col in df.columns:
                        text_col = col
                        break
                
                if text_col:
                    for idx, row in df.iterrows():
                        emails.append({
                            'text': str(row[text_col]),
                            'label': 1,  # Phishing = 1
                            'source': dataset_name,
                            'id': f"{dataset_name}_{idx}"
                        })
                else:
                    logger.warning(f"No text column found in {file_path}")
            
            # Try text file format (one email per line or file)
            elif file_path.endswith('.txt') or file_path.endswith('.eml'):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    # If multiple emails separated by some delimiter
                    if '---' in content or '\n\n\n' in content:
                        parts = re.split(r'---|\n\n\n', content)
                        for i, part in enumerate(parts):
                            if part.strip():
                                emails.append({
                                    'text': part.strip(),
                                    'label': 1,
                                    'source': dataset_name,
                                    'id': f"{dataset_name}_{i}"
                                })
                    else:
                        emails.append({
                            'text': content,
                            'label': 1,
                            'source': dataset_name,
                            'id': f"{dataset_name}_0"
                        })
            
            logger.info(f"Loaded {len(emails)} emails from {dataset_name}")
            
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {e}")
        
        return emails
    
    def load_legitimate_dataset(self, dataset_name: str, file_path: str) -> List[Dict]:
        """
        Load legitimate (ham) dataset
        
        Args:
            dataset_name: Name of dataset
            file_path: Path to dataset file or directory
        
        Returns:
            List of email dictionaries
        """
        emails = []
        
        if not os.path.exists(file_path):
            logger.warning(f"Dataset path not found: {file_path}")
            return emails
        
        try:
            # If directory, load all files
            if os.path.isdir(file_path):

                # --- MODIFICATION START ---
                # Original code only checked for .txt, .eml, .msg
                # We need to also get files with no extension (like SpamAssassin)
                
                files = []
                for f in os.listdir(file_path):
                    full_path = os.path.join(file_path, f)
                    
                    # Make sure it's a file, not a directory (like easy_ham_2)
                    if os.path.isfile(full_path):
                        # Keep it if it has a valid extension
                        if f.endswith(('.txt', '.eml', '.msg')):
                            files.append(f)
                        # OR keep it if it starts with a digit (like '00395.1a0d...')
                        # This avoids files like 'cmds' or '.tar' files
                        elif f[0].isdigit():
                            files.append(f)
                
                # --- MODIFICATION END ---
                
                for i, filename in enumerate(files):
                    file_path_full = os.path.join(file_path, filename)
                    with open(file_path_full, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        emails.append({
                            'text': content,
                            'label': 0,  # Legitimate = 0
                            'source': dataset_name,
                            'id': f"{dataset_name}_{i}"
                        })
            
            # If single file
            else:
                # Try CSV
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    text_col = None
                    for col in ['text', 'body', 'content', 'email', 'message', 'ham']:
                        if col in df.columns:
                            text_col = col
                            break
                    
                    if text_col:
                        for idx, row in df.iterrows():
                            emails.append({
                                'text': str(row[text_col]),
                                'label': 0,
                                'source': dataset_name,
                                'id': f"{dataset_name}_{idx}"
                            })
                
                # Try text file
                else:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        emails.append({
                            'text': content,
                            'label': 0,
                            'source': dataset_name,
                            'id': f"{dataset_name}_0"
                        })
            
            logger.info(f"Loaded {len(emails)} emails from {dataset_name}")
            
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {e}")
        
        return emails
    
    def load_all_datasets(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Load all 6 datasets from CSV files in root directory
        
        Returns:
            Tuple of (phishing_emails, legitimate_emails)
        """
        phishing_emails = []
        legitimate_emails = []
        
        # Get root directory (parent of data_dir)
        if isinstance(self.data_dir, str):
            root_dir = Path(self.data_dir).parent if self.data_dir.startswith('data') else Path('.')
        else:
            root_dir = Path(self.data_dir).parent if str(self.data_dir).startswith('data') else Path('.')
        
        # If data_dir is just 'data', root should be current directory
        if str(self.data_dir) == 'data':
            root_dir = Path('.')
        
        # Phishing datasets - check root directory first
        nazario_path = root_dir / 'Nazario.csv'
        if not nazario_path.exists():
            nazario_path = os.path.join(self.data_dir, 'raw', 'nazario.csv')
        
        nigerian_path = root_dir / 'Nigerian_Fraud.csv'
        if not nigerian_path.exists():
            nigerian_path = os.path.join(self.data_dir, 'raw', 'nigerian_fraud.csv')
        
        nazario = self.load_phishing_dataset('nazario', str(nazario_path))
        nigerian = self.load_phishing_dataset('nigerian_fraud', str(nigerian_path))
        
        phishing_emails.extend(nazario)
        phishing_emails.extend(nigerian)
        
        # Legitimate datasets - check root directory CSV files
        enron_path = root_dir / 'Enron.csv'
        spamassassin_path = root_dir / 'SpamAssassin.csv'
        trec_path = None  # TREC might not be in root
        ceas_path = root_dir / 'CEAS_08.csv'
        
        # Try loading from CSV files in root
        if enron_path.exists():
            enron = self.load_legitimate_dataset('enron', str(enron_path))
        else:
            enron_path = os.path.join(self.data_dir, 'raw', 'enron')
            enron = self.load_legitimate_dataset('enron', enron_path)
        
        if spamassassin_path.exists():
            spamassassin = self.load_legitimate_dataset('spamassassin', str(spamassassin_path))
        else:
            spamassassin_path = os.path.join(self.data_dir, 'raw', 'spamassassin')
            spamassassin = self.load_legitimate_dataset('spamassassin', spamassassin_path)
        
        if ceas_path.exists():
            ceas = self.load_legitimate_dataset('ceas2008', str(ceas_path))
        else:
            ceas_path = os.path.join(self.data_dir, 'raw', 'ceas2008')
            ceas = self.load_legitimate_dataset('ceas2008', ceas_path)
        
        # TREC - try to find in root or data/raw
        trec_path = os.path.join(self.data_dir, 'raw', 'trec2007')
        trec = self.load_legitimate_dataset('trec2007', trec_path)
        
        legitimate_emails.extend(enron)
        legitimate_emails.extend(spamassassin)
        legitimate_emails.extend(trec)
        legitimate_emails.extend(ceas)
        
        logger.info(f"Total phishing emails: {len(phishing_emails)}")
        logger.info(f"Total legitimate emails: {len(legitimate_emails)}")
        
        return phishing_emails, legitimate_emails
    
    def split_train_test(self, emails: List[Dict], test_ratio: float = 0.2, 
                         random_seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
        """
        Split emails into train and test sets
        
        Args:
            emails: List of email dictionaries
            test_ratio: Ratio of test set
            random_seed: Random seed for reproducibility
        
        Returns:
            Tuple of (train_emails, test_emails)
        """
        np.random.seed(random_seed)
        indices = np.random.permutation(len(emails))
        split_idx = int(len(emails) * (1 - test_ratio))
        
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
        
        train_emails = [emails[i] for i in train_indices]
        test_emails = [emails[i] for i in test_indices]
        
        return train_emails, test_emails