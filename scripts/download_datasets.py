"""
Dataset Download Script
Downloads the 6 required datasets for MultiPhishGuard
"""

import os
import requests
import zipfile
import tarfile
import logging
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetDownloader:
    """Download datasets from various sources"""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def download_file(self, url: str, filename: str, description: str = ""):
        """Download a file with progress bar"""
        filepath = self.data_dir / filename
        
        if filepath.exists():
            logger.info(f"{description} already exists: {filename}")
            return filepath
        
        try:
            logger.info(f"Downloading {description}: {filename}")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f, tqdm(
                desc=description,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            logger.info(f"Downloaded {filename}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error downloading {description}: {e}")
            return None
    
    def extract_archive(self, archive_path: Path, extract_dir: Path = None):
        """Extract archive file"""
        if extract_dir is None:
            extract_dir = self.data_dir
        
        try:
            logger.info(f"Extracting {archive_path.name}")
            
            if archive_path.suffix == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
            elif archive_path.suffix in ['.tar', '.gz', '.bz2']:
                mode = 'r'
                if archive_path.suffix == '.gz':
                    mode = 'r:gz'
                elif archive_path.suffix == '.bz2':
                    mode = 'r:bz2'
                
                with tarfile.open(archive_path, mode) as tar_ref:
                    tar_ref.extractall(extract_dir)
            
            logger.info(f"Extracted to {extract_dir}")
            
        except Exception as e:
            logger.error(f"Error extracting {archive_path.name}: {e}")
    
    def download_nazario(self):
        """Download Nazario Phishing Corpus (402 emails)"""
        url = "https://raw.githubusercontent.com/rokibulroni/Phishing-Email-Dataset/main/Nazario.csv"
        filename = "nazario.csv"
        return self.download_file(url, filename, "Nazario Phishing Corpus (402 emails)")
    
    def download_nigerian_fraud(self):
        """Download Nigerian Fraud Dataset (577 emails)"""
        # Try GitHub URL
        url = "https://raw.githubusercontent.com/rokibulroni/Phishing-Email-Dataset/main/Nigerian%20Fraud.csv"
        filename = "nigerian_fraud.csv"
        result = self.download_file(url, filename, "Nigerian Fraud Dataset (577 emails)")
        if result is None:
            # Try alternative URL format
            url2 = "https://raw.githubusercontent.com/rokibulroni/Phishing-Email-Dataset/main/Nigerian_Fraud.csv"
            result = self.download_file(url2, filename, "Nigerian Fraud Dataset (577 emails)")
        return result
    
    def download_enron(self):
        """Download Enron-Spam Dataset (1,500 ham emails)"""
        # Enron dataset - try to download from public mirror or GitHub
        # Note: Full Enron dataset is large, we need ham subset
        enron_dir = self.data_dir / "enron"
        enron_dir.mkdir(exist_ok=True)
        
        # Try GitHub mirror with ham subset
        logger.info("Attempting to download Enron-Spam ham subset...")
        logger.warning("Enron dataset is large. If auto-download fails, manual download may be required.")
        logger.info("Manual download: https://www.cs.cmu.edu/~enron/")
        logger.info("Or search for 'Enron ham emails dataset' on GitHub/kaggle")
        return enron_dir
    
    def download_spamassassin(self):
        """Download SpamAssassin Dataset (500 hard ham emails)"""
        spamassassin_dir = self.data_dir / "spamassassin"
        spamassassin_dir.mkdir(exist_ok=True)
        
        # SpamAssassin public corpus - hard ham subset
        logger.info("Downloading SpamAssassin hard ham dataset...")
        # Try direct download URL for hard ham
        url = "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham_2.tar.bz2"
        archive = self.download_file(url, "spamassassin_hard_ham.tar.bz2", "SpamAssassin Hard Ham")
        
        if archive:
            self.extract_archive(archive, spamassassin_dir)
            return spamassassin_dir
        else:
            logger.warning("Auto-download failed. Please download manually from:")
            logger.info("https://spamassassin.apache.org/old/publiccorpus/")
            return spamassassin_dir
    
    def download_trec2007(self):
        """Download TREC 2007 Dataset (500 ham emails)"""
        trec_dir = self.data_dir / "trec2007"
        trec_dir.mkdir(exist_ok=True)
        
        logger.info("Downloading TREC 2007 ham dataset...")
        logger.warning("TREC 2007 may require manual download from:")
        logger.info("https://plg.uwaterloo.ca/~gvcormac/treccorpus07/")
        logger.info("Or search for 'TREC 2007 spam corpus' on research repositories")
        return trec_dir
    
    def download_ceas2008(self):
        """Download CEAS 2008 Dataset (500 ham emails)"""
        ceas_dir = self.data_dir / "ceas2008"
        ceas_dir.mkdir(exist_ok=True)
        
        logger.info("Downloading CEAS 2008 ham dataset...")
        logger.warning("CEAS 2008 may require manual download from:")
        logger.info("http://www.ceas.cc/2008/challenge.html")
        logger.info("Or search for 'CEAS 2008 dataset' on research repositories")
        return ceas_dir
    
    def download_all(self):
        """Download all 6 required datasets"""
        logger.info("="*60)
        logger.info("MultiPhishGuard Dataset Downloader")
        logger.info("Downloading 6 datasets as specified:")
        logger.info("  Phishing: Nazario (402), Nigerian Fraud (577)")
        logger.info("  Legitimate: Enron (1500), SpamAssassin (500), TREC 2007 (500), CEAS 2008 (500)")
        logger.info("="*60)
        
        # Download phishing datasets
        logger.info("\n--- Phishing Datasets ---")
        nazario_path = self.download_nazario()
        nigerian_path = self.download_nigerian_fraud()
        
        # Download legitimate datasets
        logger.info("\n--- Legitimate Datasets ---")
        enron_path = self.download_enron()
        spamassassin_path = self.download_spamassassin()
        trec_path = self.download_trec2007()
        ceas_path = self.download_ceas2008()
        
        logger.info("\n" + "="*60)
        logger.info("Download Summary:")
        logger.info(f"  Nazario Phishing: {'✓' if nazario_path else '✗'}")
        logger.info(f"  Nigerian Fraud: {'✓' if nigerian_path else '✗'}")
        logger.info(f"  Enron-Spam: Directory created")
        logger.info(f"  SpamAssassin: Directory created")
        logger.info(f"  TREC 2007: Directory created")
        logger.info(f"  CEAS 2008: Directory created")
        logger.info("="*60)
        logger.info("\nNote: Some datasets may require manual download.")
        logger.info("Please verify all datasets are in data/raw/ before preprocessing.")


if __name__ == "__main__":
    downloader = DatasetDownloader()
    downloader.download_all()

