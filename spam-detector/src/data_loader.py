import os
import urllib.request
import zipfile
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DATA_FILE = os.path.join(DATA_DIR, "SMSSpamCollection")
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
BACKUP_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"

def get_data_dir() -> str:
    """Ensure data directory exists and return its path."""
    os.makedirs(DATA_DIR, exist_ok=True)
    return DATA_DIR

def load_spam_data() -> pd.DataFrame:
    """
    Download (if not already present) and load the SMS Spam Collection dataset.
    Returns a pandas DataFrame with columns: ['label', 'message'].
    """
    get_data_dir()
    
    if not os.path.exists(DATA_FILE):
        zip_path = os.path.join(DATA_DIR, "sms_spam_collection.zip")
        downloaded = False
        
        # Try UCI primary URL
        try:
            print("Downloading SMS Spam Collection from UCI repository...")
            urllib.request.urlretrieve(UCI_ZIP_URL, zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(DATA_DIR)
            downloaded = True
        except Exception as e:
            print(f"Primary download failed ({e}). Trying backup mirror...")
            
        # Fallback if primary fails
        if not downloaded or not os.path.exists(DATA_FILE):
            urllib.request.urlretrieve(BACKUP_URL, DATA_FILE)
            print("Downloaded from mirror successfully.")
            
        if os.path.exists(zip_path):
            try:
                os.remove(zip_path)
            except OSError:
                pass

    # Read tab-separated file without header
    df = pd.read_csv(
        DATA_FILE,
        sep='\t',
        names=['label', 'message'],
        header=None,
        encoding='utf-8',
        quoting=3 # csv.QUOTE_NONE
    )
    # Basic cleanup
    df = df.dropna().reset_index(drop=True)
    return df

if __name__ == "__main__":
    df = load_spam_data()
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    print(df.head())
    print("\nClass distribution:")
    print(df['label'].value_counts())
