import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure necessary NLTK data is downloaded
def _init_nltk():
    resources = [
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('corpora/stopwords', 'stopwords'),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except (LookupError, FileNotFoundError):
            nltk.download(name, quiet=True)

_init_nltk()

try:
    STOPWORDS = set(stopwords.words('english'))
except Exception:
    _init_nltk()
    STOPWORDS = set(stopwords.words('english'))

# Custom/extended stop words or domain-specific handling if needed
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
NUM_PATTERN = re.compile(r'\d+')
PUNCT_TRANSLATION = str.maketrans('', '', string.punctuation)

def clean_text(text: str) -> str:
    """
    Apply full text normalization pipeline:
    1. Lowercase text
    2. Strip URLs (e.g., http://..., www...)
    3. Strip numerical digits
    4. Remove punctuation
    5. Tokenize using NLTK word_tokenize
    6. Filter out English stopwords and tokens of length < 2
    7. Rejoin into a clean string for vectorizers
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Strip URLs
    text = URL_PATTERN.sub(' ', text)
    
    # 3. Strip numbers
    text = NUM_PATTERN.sub(' ', text)
    
    # 4. Remove punctuation
    text = text.translate(PUNCT_TRANSLATION)
    
    # 5. Tokenize
    tokens = word_tokenize(text)
    
    # 6. Stopwords & minimum length filtering
    cleaned_tokens = [
        word for word in tokens 
        if word not in STOPWORDS and len(word) > 1
    ]
    
    # 7. Recombine
    return ' '.join(cleaned_tokens)

def preprocess_series(series):
    """Clean an entire pandas Series of text."""
    return series.apply(clean_text)

if __name__ == "__main__":
    test_sample = "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Draw! Call 09061743810 now or visit www.prize-draw.com."
    print("Original:\n", test_sample)
    print("\nCleaned:\n", clean_text(test_sample))
