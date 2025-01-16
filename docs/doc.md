# Code Documentation
Generated on: 2025-01-16T21:11:28.736Z
Total files: 10

## Project Structure

```
└── language_detection
    ├── __init__.py
    ├── config
    │   └── settings.py
    ├── core
    │   ├── dictionary_based_detection.py
    │   └── script_analyzer.py
    ├── index.py
    ├── tests
    │   └── test_detector.py
    └── utils
        ├── DataLoader.py
        ├── extract_keywords.py
        └── text
            ├── TextCleaner.py
            └── TextNormalizer.py
```

## File: __init__.py
- Path: `/root/git/language_detection/__init__.py`
- Size: 0.00 B
- Extension: .py
- Lines of code: 0

```py

```

---------------------------------------------------------------------------

## File: index.py
- Path: `/root/git/language_detection/index.py`
- Size: 8.04 KB
- Extension: .py
- Lines of code: 169

```py
from utils.DataLoader import DataLoader
from utils.text.TextCleaner import TextCleaner
from utils.text.TextNormalizer import TextNormalizer
from core.script_analyzer import ScriptAnalyzer
import pandas as pd
import logging
from typing import Dict
from core.dictionary_based_detection import DictionaryBasedDetection
class DataProcessor:
"""
A class to handle the complete data processing pipeline including loading,
cleaning, and normalizing text data.
"""
def __init__(self, arabic_threshold: float = 0.7, mixed_threshold: float = 0.3):
self.loader = DataLoader()
self.cleaner = TextCleaner()
self.normalizer = TextNormalizer()
self.script_analyzer = ScriptAnalyzer(
arabic_threshold=arabic_threshold,
mixed_threshold=mixed_threshold
)
self.arabic_threshold = arabic_threshold
self.logger = logging.getLogger(__name__)
# Configure logging
logging.basicConfig(
level=logging.INFO,
format='%(asctime)s - %(levelname)s - %(message)s'
)
def textCleaning(self, text: str):
"""
Clean the text using the TextCleaner class.
Args:
text: Input text to clean
Returns:
Tuple containing processed text and analysis results
"""
# Clean the text
cleaned_text = self.cleaner.clean_text(text, keep_minimal_punct=True, min_length=1)
if not cleaned_text:
return None, {'status': "empty"}
# Normalize the text
normalized_text = self.normalizer.normalize(cleaned_text)
analysis = self.script_analyzer.analyze_text(normalized_text)
return normalized_text, {
'status': "processed", # status of the text processing
'primary_script': analysis.primary_script, # the most dominant script in the text
'confidence': analysis.confidence, # confidence in the primary script
'is_mixed': analysis.is_mixed, # whether the text is mixed
'script_ratios': analysis.script_ratios, # ratio of characters in the text by script
'char_counts': analysis.char_counts # count of characters in the text by script
}
def dictionaryBasedDetection(self, text: str, df: pd.DataFrame, idx: int):
pass # TODO: implement dictionary based detection
def statisticalDetection(self, text: str, df: pd.DataFrame, idx: int):
pass # TODO: implement statistical detection
def textDetection(self, text: str, df: pd.DataFrame, idx: int):
"""
Detect the text using the TextDetector class.
Args:
text: Input text to detect
df: DataFrame containing the text data
idx: Index of the text in the DataFrame
"""
processed_text, analysis = self.textCleaning(text)
if analysis['status'] != 'processed':
return
df.at[idx, 'processed_text'] = processed_text # processed text
df.at[idx, 'primary_script'] = analysis['primary_script'] # the most dominant script in the text
df.at[idx, 'script_confidence'] = analysis['confidence'] # confidence in the primary script
df.at[idx, 'is_mixed'] = analysis['is_mixed'] # whether the text is mixed
df.at[idx, 'arabic_ratio'] = analysis['script_ratios']['arabic'] # ratio of Arabic characters in the text
df.at[idx, 'latin_ratio'] = analysis['script_ratios']['latin']
# If the text has a arabic ratio greater than the threshold, then it is an arabic text
if analysis['script_ratios']['arabic'] > self.arabic_threshold:
df.at[idx, 'label'] = 'arabic'
return
# if text length is less than 3 words, then do dictionary based detection, otherwise do statistical detection
if len(processed_text.split()) < 3:
self.dictionaryBasedDetection(processed_text, df, idx)
else:
self.statisticalDetection(processed_text, df, idx)
def process_dataset(self, file_path: str, text_column: str = "text"):
"""
Process the complete dataset through cleaning and normalization pipeline.
Args:
file_path: Path to the CSV file
text_column: Name of the column containing text data
Returns:
DataFrame with original and processed text
"""
try:
# Load the dataset
self.logger.info(f"Loading dataset from {file_path}")
df = self.loader.load_csv(file_path, text_column)
# Initialize new columns
df['processed_text'] = None
df['primary_script'] = None
df['script_confidence'] = None
df['is_mixed'] = None
df['arabic_ratio'] = None
df['latin_ratio'] = None
# Process each row
total_rows = len(df)
for idx, row in df.iterrows():
if idx % 100 == 0:
self.logger.info(f"Processing row {idx}/{total_rows}")
self.textDetection(row[text_column], df, idx)
# Generate and log processing statistics
stats = self.generate_statistics(df)
self.log_statistics(stats)
return df
except Exception as e:
self.logger.error(f"Error processing dataset: {str(e)}")
raise
def generate_statistics(self, df: pd.DataFrame):
"""
Generate statistics about the processed dataset.
Args:
df: Processed DataFrame
text_column: Name of the original text column
Returns:
Dictionary containing various statistics
"""
stats = {
'total_rows': len(df),
'empty_rows': df['processed_text'].isna().sum(),
'script_distribution': df['primary_script'].value_counts().to_dict(),
'mixed_texts': df['is_mixed'].sum(),
'avg_confidence': df['script_confidence'].mean(),
'script_confidence_stats': {
'mean': df['script_confidence'].mean(),
'median': df['script_confidence'].median(),
'std': df['script_confidence'].std()
},
'arabic_ratio_stats': {
'mean': df['arabic_ratio'].mean(),
'median': df['arabic_ratio'].median(),
'std': df['arabic_ratio'].std()
}
}
return stats
def log_statistics(self, stats: Dict):
"""
Log the processing statistics.
Args:
stats: Dictionary of statistics to log
"""
self.logger.info("\nDataset Processing Statistics:")
self.logger.info(f"Total rows processed: {stats['total_rows']}")
self.logger.info(f"Empty rows: {stats['empty_rows']}")
self.logger.info("\nScript Distribution:")
for script, count in stats['script_distribution'].items():
self.logger.info(f"  {script}: {count}")
self.logger.info(f"\nMixed texts: {stats['mixed_texts']}")
self.logger.info(f"Average confidence: {stats['avg_confidence']:.2f}")
self.logger.info("\nScript Confidence Statistics:")
for metric, value in stats['script_confidence_stats'].items():
self.logger.info(f"  {metric}: {value:.2f}")
self.logger.info("\nArabic Ratio Statistics:")
for metric, value in stats['arabic_ratio_stats'].items():
self.logger.info(f"  {metric}: {value:.2f}")
# Example usage
if __name__ == "__main__":
processor = DataProcessor(arabic_threshold=0.7, mixed_threshold=0.3)
processed_df = processor.process_dataset('data/dataset1.csv', 'text')
# Save the processed dataset
output_path = 'data/processed_dataset1.json'
processed_df.to_json(output_path, index=False)
print(f"Processed dataset saved to {output_path}")
```

---------------------------------------------------------------------------

## File: settings.py
- Path: `/root/git/language_detection/config/settings.py`
- Size: 0.00 B
- Extension: .py
- Lines of code: 0

```py

```

---------------------------------------------------------------------------

## File: dictionary_based_detection.py
- Path: `/root/git/language_detection/core/dictionary_based_detection.py`
- Size: 3.21 KB
- Extension: .py
- Lines of code: 63

```py
import os
import string
from collections import defaultdict
from typing import Dict, List, Set
path = os.path.dirname(os.path.abspath(__file__))
from utils.extract_keywords import extract_keywords
class DictionaryBasedDetection:
def __init__(self, languages: List[str], short_text_threshold: int = 3):
"""
Initialize the detector with language dictionaries.
:param languages: List of language codes or names (e.g., ['english', 'french', 'arabic']).
:param short_text_threshold: Maximum number of tokens to use dictionary-based detection.
"""
self.languages = languages
self.short_text_threshold = short_text_threshold
self.language_dictionaries = self.load_dictionaries(languages)
def load_dictionaries(self, languages: List[str]):
"""
Load word lists for each language into a dictionary.
:param languages: List of language names corresponding to dictionary files.
:return: A dictionary mapping language names to sets of words.
"""
language_dictionaries = {}
language_dictionaries['english'] = english_keywords
language_dictionaries['arabic'] = arabic_keywords
language_dictionaries['french'] = french_keywords
return language_dictionaries
def tokenize(self, text: str):
"""
Tokenize the input text into words.
:param text: The text to tokenize.
:return: A list of lowercased tokens without punctuation.
"""
tokens = text.lower().split()
return tokens
def detect_language(self, text: str) -> Tuple[str, float]:
"""
Detect the language of the input text using dictionary-based detection.
:param text: The text to detect.
:return: The detected language code.
"""
tokens = self.tokenize(text)
num_tokens = len(tokens) # get the number of tokens in the text
if num_tokens >= self.short_text_threshold:
raise ValueError("Text is too long to detect language")
# Initialize a dictionary to count matches for each language
language_matches = defaultdict(int)
for token in tokens:
for lang, vocab in self.language_dictionaries.items():
if token in vocab:
language_matches[lang] += 1 # increment the count for the language
if not language_matches:
return "unknown", 0.0
detected_language = max(language_matches, key=lambda lang: language_matches[lang]) # get the language with the most matches
max_matches = language_matches[detected_language] # get the number of matches for the detected language
total_matches = sum(language_matches.values()) # get the total number of matches
confidence = max_matches / total_matches # calculate the confidence
return detected_language, confidence # return the detected language and confidence
if __name__ == "__main__":
detector = DictionaryBasedDetection(["eng", "fra"])
text = "Hello, how are you?"
language, confidence = detector.detect_language(text)
print(f"Detected language: {language}, Confidence: {confidence}")
```

---------------------------------------------------------------------------

## File: script_analyzer.py
- Path: `/root/git/language_detection/core/script_analyzer.py`
- Size: 5.62 KB
- Extension: .py
- Lines of code: 120

```py
import re
from typing import Dict, Tuple
from dataclasses import dataclass
@dataclass
class ScriptAnalysis:
"""Data class to hold script analysis results"""
primary_script: str
confidence: float
script_ratios: Dict[str, float]
is_mixed: bool
char_counts: Dict[str, int]
class ScriptAnalyzer:
"""
A class for detailed script analysis of text, focusing on Arabic and Latin scripts.
Provides detailed character set analysis and script classification.
"""
def __init__(self, arabic_threshold: float = 0.7, mixed_threshold: float = 0.3):
"""
Initialize the ScriptAnalyzer with configurable thresholds.
Args:
arabic_threshold: Threshold above which text is classified as Arabic
mixed_threshold: Threshold below which text is considered mixed
"""
self.arabic_threshold = arabic_threshold
self.mixed_threshold = mixed_threshold
# Compile regex patterns for different scripts
self.patterns = {
'arabic': re.compile('[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]'),
'latin': re.compile('[a-zA-Z\u00C0-\u00FF\u0100-\u017F]'),
'numbers': re.compile('[\u0660-\u0669\u06F0-\u06F9\u0030-\u0039]'),
'punctuation': re.compile('[\u0021-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E\u060C\u061B\u061F]')
}
def count_characters(self, text: str) -> Dict[str, int]:
"""
Count characters by script category.
Args:
text: Input text to analyze
Returns:
Dictionary with character counts per script
"""
counts = {script: len(pattern.findall(text)) for script, pattern in self.patterns.items()}
counts['total'] = sum(len(pattern.findall(text)) for pattern in self.patterns.values())
return counts
def calculate_script_ratios(self, char_counts: Dict[str, int]) -> Dict[str, float]:
"""
Calculate the ratio of each script relative to total meaningful characters.
Args:
char_counts: Dictionary of character counts by script
Returns:
Dictionary with script ratios
"""
# Calculate total meaningful characters (excluding punctuation and numbers)
meaningful_total = char_counts['arabic'] + char_counts['latin']
if meaningful_total == 0:
return {'arabic': 0.0, 'latin': 0.0}
return {
'arabic': char_counts['arabic'] / meaningful_total if meaningful_total > 0 else 0,
'latin': char_counts['latin'] / meaningful_total if meaningful_total > 0 else 0
}
def analyze_text(self, text: str):
"""
Perform comprehensive script analysis on the input text.
Args:
text: Input text to analyze
Returns:
ScriptAnalysis object containing detailed analysis results
"""
if not text or text.strip() == '':
return ScriptAnalysis(
primary_script='empty', # the most dominant script in the text
confidence=0.0, # confidence in the primary script
script_ratios={'arabic': 0.0, 'latin': 0.0}, # ratio of characters in the text by script
is_mixed=False, # whether the text is mixed
char_counts={'arabic': 0, 'latin': 0, 'total': 0} # count of characters in the text by script
)
# Count characters by script
char_counts = self.count_characters(text)
# Calculate script ratios
script_ratios = self.calculate_script_ratios(char_counts)
# Determine primary script and confidence
arabic_ratio = script_ratios['arabic']
if arabic_ratio >= self.arabic_threshold:
primary_script = 'arabic'
confidence = arabic_ratio
elif arabic_ratio <= self.mixed_threshold:
primary_script = 'latin'
confidence = script_ratios['latin']
else:
primary_script = 'mixed'
confidence = max(script_ratios.values())
# Determine if the text is mixed
is_mixed = (
min(script_ratios.values()) > self.mixed_threshold
and max(script_ratios.values()) < self.arabic_threshold
)
return ScriptAnalysis(
primary_script=primary_script,
confidence=confidence,
script_ratios=script_ratios,
is_mixed=is_mixed,
char_counts=char_counts
)
def get_detailed_stats(self, text: str):
"""
Generate detailed statistics about the text's character composition.
Args:
text: Input text to analyze
Returns:
Dictionary containing detailed character statistics
"""
analysis = self.analyze_text(text)
return {
'total_length': len(text), # total number of characters in the text
'meaningful_chars': analysis.char_counts['arabic'] + analysis.char_counts['latin'], # total number of characters in the text that are not punctuation or numbers
'script_distribution': analysis.script_ratios, # ratio of characters in the text by script
'primary_script': analysis.primary_script, # the most dominant script in the text
'confidence': analysis.confidence, # confidence in the primary script
'is_mixed': analysis.is_mixed, # whether the text is mixed
'char_counts': analysis.char_counts # count of characters in the text by script
}
```

---------------------------------------------------------------------------

## File: test_detector.py
- Path: `/root/git/language_detection/tests/test_detector.py`
- Size: 0.00 B
- Extension: .py
- Lines of code: 0

```py

```

---------------------------------------------------------------------------

## File: DataLoader.py
- Path: `/root/git/language_detection/utils/DataLoader.py`
- Size: 4.51 KB
- Extension: .py
- Lines of code: 96

```py
from typing import List, Dict, Optional
import pandas as pd
import logging
from pathlib import Path
class DataLoader:
"""
A class to handle loading and preprocessing of multilingual text data from CSV files.
Specifically designed to handle Arabic, French, and English text.
"""
def __init__(self, encoding: str = 'utf-8'):
"""
Initialize the DataLoader.
Args:
encoding (str): The encoding to use when reading files. Defaults to 'utf-8'.
"""
self.encoding = encoding
self.logger = logging.getLogger(__name__)
def load_csv(self,
file_path: str,
text_column: str,
label_column: Optional[str] = None,
**kwargs) -> pd.DataFrame:
"""
Load data from a CSV file with proper encoding handling.
Args:
file_path (str): Path to the CSV file
text_column (str): Name of the column containing the text data
label_column (str, optional): Name of the column containing language labels
**kwargs: Additional arguments to pass to pd.read_csv
Returns:
pd.DataFrame: DataFrame containing the loaded data
Raises:
FileNotFoundError: If the file doesn't exist
ValueError: If required columns are not found in the CSV
"""
try:
# Ensure the file exists
if not Path(file_path).exists():
raise FileNotFoundError(f"File not found: {file_path}")
# Try to read the CSV file with the specified encoding
df = pd.read_csv(file_path, encoding=self.encoding, **kwargs)
# Verify required columns exist
if text_column not in df.columns:
raise ValueError(f"Text column '{text_column}' not found in CSV")
if label_column and label_column not in df.columns:
raise ValueError(f"Label column '{label_column}' not found in CSV")
# Basic data cleaning
# Remove rows where text is completely empty
df = df.dropna(subset=[text_column])
# Convert text to string type (in case it was read as something else)
df[text_column] = df[text_column].astype(str)
return df
except UnicodeDecodeError:
# If UTF-8 fails, try with another common encoding
self.logger.warning(f"UTF-8 decode failed, attempting with 'cp1256' encoding...")
df = pd.read_csv(file_path, encoding='cp1256', **kwargs)
return df
def load_data_batch(self, file_paths: List[str], text_column: str, label_column: Optional[str] = None) -> pd.DataFrame:
"""
Load and combine data from multiple CSV files.
Args:
file_paths (List[str]): List of paths to CSV files
text_column (str): Name of the column containing the text data
label_column (str, optional): Name of the column containing language labels
Returns:
pd.DataFrame: Combined DataFrame from all files
"""
dfs = []
for file_path in file_paths:
try:
df = self.load_csv(file_path, text_column, label_column)
dfs.append(df)
except Exception as e:
self.logger.error(f"Error loading file {file_path}: {str(e)}")
continue
if not dfs:
raise ValueError("No data was successfully loaded from any of the provided files")
return pd.concat(dfs, ignore_index=True)
def get_data_stats(self, df: pd.DataFrame, text_column: str) -> Dict:
"""
Get basic statistics about the loaded data.
Args:
df (pd.DataFrame): The loaded DataFrame
text_column (str): Name of the column containing the text data
Returns:
Dict: Dictionary containing various statistics about the data
"""
stats = {
'total_rows': len(df),
'empty_rows': df[text_column].str.strip().eq('').sum(),
'unique_texts': df[text_column].nunique(),
'avg_text_length': df[text_column].str.len().mean(),
'max_text_length': df[text_column].str.len().max(),
'min_text_length': df[text_column].str.len().min()
}
return stats
```

---------------------------------------------------------------------------

## File: extract_keywords.py
- Path: `/root/git/language_detection/utils/extract_keywords.py`
- Size: 1.30 KB
- Extension: .py
- Lines of code: 34

```py
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import wordnet as wn
import os
from text.TextCleaner import TextCleaner
from text.TextNormalizer import TextNormalizer
def extract_keywords(language: str):
"""
Extract keywords from the given language.
"""
# initialize the text cleaner and normalizer
textCleaner = TextCleaner()
textNormalizer = TextNormalizer()
# get the keywords from the wordnet
keywords = set()
keywords_cleaned = set()
# get the keywords from the wordnet
for synset in wn.all_synsets(pos=wn.NOUN):  # You can use other POS: VERB, ADJ, ADV
for lemma in synset.lemmas(language):  # Specify language
keywords.add(lemma.name().lower().replace('_', ' '))
# clean and normalize the keywords
for keyword in keywords:
keyword = textCleaner.clean_text(keyword) # clean the keyword
keyword = textNormalizer.normalize_text(keyword) # normalize the keyword
keywords_cleaned.add(keyword)
return keywords_cleaned
english_keywords = extract_keywords("eng")
french_keywords = extract_keywords("fra")
arabic_keywords = extract_keywords("ara")
if __name__ == "__main__":
# find if love exist in the intersection
love = "love" in english_keywords
print(love)
```

---------------------------------------------------------------------------

## File: TextCleaner.py
- Path: `/root/git/language_detection/utils/text/TextCleaner.py`
- Size: 4.55 KB
- Extension: .py
- Lines of code: 91

```py
import re # regular expressions
import emoji # emoji detection
from typing import Optional # type hinting
class TextCleaner:
"""
A class to handle text cleaning and normalization operations for language detection.
Focuses on removing unwanted tokens while preserving language-relevant characters.
"""
def __init__(self):
# URL pattern to match web addresses
self.url_pattern = re.compile(
r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
)
# Social media patterns
self.mention_pattern = re.compile(r'@[\w_]+')
self.hashtag_pattern = re.compile(r'#[\w_]+')
# HTML tags pattern
self.html_pattern = re.compile(r'<[^>]+>')
# Numbers pattern (both Arabic and Latin) - only matching standalone numbers
# Matches numbers that have spaces or start/end of string around them
self.numbers_pattern = re.compile(r'(?:^|\s)[0-9٠-٩]+(?:\s|$)')
# Special characters and punctuation pattern
# Excluding minimal punctuation that might be relevant for language detection
self.special_chars_pattern = re.compile(r'[^\w\s.,!?\'"-]')
# Multiple whitespace pattern
self.whitespace_pattern = re.compile(r'\s+')
# Parentheses pattern
self.parentheses_pattern = re.compile(r'[\(\)\[\]\{\}⟨⟩«»]')
def remove_urls(self, text: str) -> str:
"""Remove URLs from text."""
return self.url_pattern.sub(' ', text)
def remove_social_media_elements(self, text: str) -> str:
"""Remove mentions and hashtags."""
text = self.mention_pattern.sub(' ', text)
return self.hashtag_pattern.sub(' ', text)
def remove_html_tags(self, text: str) -> str:
"""Remove HTML tags."""
return self.html_pattern.sub(' ', text)
def remove_emojis(self, text: str) -> str:
"""Remove emojis using the emoji library."""
return emoji.replace_emoji(text, '')
def remove_numbers(self, text: str) -> str:
"""Remove numeric digits (both Arabic and Latin)."""
return self.numbers_pattern.sub(' ', text)
def remove_special_characters(self, text: str, keep_minimal_punct: bool = True) -> str:
"""
Remove special characters and punctuation.
Args:
text: Input text to clean
keep_minimal_punct: If True, keeps basic punctuation that might be relevant
for language detection
"""
if not keep_minimal_punct:
# If we don't want to keep any punctuation, use a more aggressive pattern
return re.sub(r'[^\w\s]', ' ', text)
return self.special_chars_pattern.sub(' ', text)
def remove_parentheses(self, text: str) -> str:
"""Remove various types of parentheses."""
return self.parentheses_pattern.sub(' ', text)
def normalize_whitespace(self, text: str) -> str:
"""Convert multiple whitespace characters into a single space."""
return self.whitespace_pattern.sub(' ', text.strip())
def remove_diacritics(self, text: str) -> str:
"""Remove Arabic diacritics."""
return text.translate(str.maketrans('', '', 'ًٌٍَُِّْٕٖٜٟٓٔٗ٘ٙٚٛٝٞ٠١٢٣٤٥٦٧٨٩٠١٢٣٤٥٦٧٨٩'))
def clean_text(self,
text: Optional[str],
keep_minimal_punct: bool = True,
min_length: int = 1) -> Optional[str]:
"""
Apply all cleaning operations in sequence.
Args:
text: Input text to clean
keep_minimal_punct: Whether to keep minimal punctuation
min_length: Minimum length of text to consider non-empty
Returns:
Cleaned text or None if the input is None or becomes empty after cleaning
"""
if not text:
return None
# Apply all cleaning operations in sequence
text = self.remove_urls(text)
text = self.remove_social_media_elements(text)
text = self.remove_html_tags(text)
text = self.remove_emojis(text)
text = self.remove_numbers(text)
text = self.remove_parentheses(text)
text = self.remove_special_characters(text, keep_minimal_punct)
text = self.normalize_whitespace(text)
# Check if the resulting text meets minimum length requirement
return text if len(text.strip()) >= min_length else None
```

---------------------------------------------------------------------------

## File: TextNormalizer.py
- Path: `/root/git/language_detection/utils/text/TextNormalizer.py`
- Size: 5.88 KB
- Extension: .py
- Lines of code: 141

```py
import re
from typing import Dict, Optional
class TextNormalizer:
"""
A class to normalize text across different languages (Arabic, French, English).
Handles tasks like lowercasing, diacritic removal, and character normalization.
"""
def __init__(self):
# Arabic letter normalization mappings
self.arabic_chars_map = {
# Alef variations
'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ٱ': 'ا', 'ٲ': 'ا', 'ٳ': 'ا', 'ٵ': 'ا',
# Yeh variations
'ى': 'ي', 'ئ': 'ي', 'ي': 'ي', 'ۍ': 'ي', 'ێ': 'ي', 'ې': 'ي', 'ٸ': 'ي',
'ﯨ': 'ي', 'ﯩ': 'ي', 'ﻰ': 'ي', 'ﻱ': 'ي', 'ﻲ': 'ي', 'ﻳ': 'ي', 'ﻴ': 'ي',
# Waw variations
'ؤ': 'و', 'ٶ': 'و', 'ٷ': 'و', 'ۄ': 'و', 'ۅ': 'و', 'ۆ': 'و', 'ۇ': 'و',
'ۈ': 'و', 'ۉ': 'و', 'ۊ': 'و', 'ۋ': 'و', 'ۏ': 'و', 'ݸ': 'و', 'ݹ': 'و',
# Hamza variations
'ء': '', 'ٔ': '', 'ٕ': '', 'ٚ': '',
# Teh Marbuta and Heh
'ة': 'ه', 'ۃ': 'ه', 'ە': 'ه', 'ۀ': 'ه',
# Kaf variations
'ك': 'ک', 'ڪ': 'ک', 'ػ': 'ک', 'ؼ': 'ک', 'ڬ': 'ک', 'ڭ': 'ک',
'ڮ': 'ک', 'ݢ': 'ک', 'ݣ': 'ک', 'ݤ': 'ک', 'ݿ': 'ک',
# Alef Maksura
'ى': 'ي',
# Tatweel/Kashida
'ـ': ''
# Add more mappings as needed
}
# Compile regex patterns for efficiency
self.arabic_diacritics = re.compile(r'[\u064B-\u065F\u0670]')
self.arabic_chars = re.compile('[\u0600-\u06FF]')
self.latin_chars = re.compile('[a-zA-Z]')
# Patterns for additional normalizations
self.consecutive_spaces = re.compile(r'\s+')
self.consecutive_punctuation = re.compile(r'([!?,.]){2,}')
def normalize_arabic(self, text: str) -> str:
"""
Normalize Arabic text by removing diacritics and standardizing characters.
Args:
text: Input Arabic text
Returns:
Normalized Arabic text
"""
# Remove diacritics (tashkeel)
text = self.arabic_diacritics.sub('', text)
# Normalize Arabic letters
for original, normalized in self.arabic_chars_map.items():
text = text.replace(original, normalized)
return text
def normalize_latin(self, text: str) -> str:
"""
Normalize Latin script text (French/English).
Args:
text: Input text in Latin script
Returns:
Normalized text
"""
# Convert to lowercase
return text.lower()
def normalize_punctuation(self, text: str) -> str:
"""
Normalize punctuation and whitespace.
Args:
text: Input text
Returns:
Text with normalized punctuation and whitespace
"""
# Replace multiple spaces with single space
text = self.consecutive_spaces.sub(' ', text)
# Replace multiple punctuation marks with single one
text = self.consecutive_punctuation.sub(r'\1', text)
# Strip leading/trailing whitespace
return text.strip()
def detect_script(self, text: str) -> str:
"""
Detect the predominant script in the text.
Args:
text: Input text
Returns:
'arabic' for Arabic script, 'latin' for Latin script,
'mixed' if no clear predominance
"""
arabic_count = len(self.arabic_chars.findall(text))
latin_count = len(self.latin_chars.findall(text))
# Use a threshold to determine predominant script
total = arabic_count + latin_count
if total == 0:
return 'unknown'
arabic_ratio = arabic_count / total
if arabic_ratio > 0.7:
return 'arabic'
elif arabic_ratio < 0.3:
return 'latin'
else:
return 'mixed'
def normalize(self, text: str, force_script: Optional[str] = None) -> str:
"""
Main normalization method that handles text in any supported script.
Args:
text: Input text to normalize
force_script: Optional script type to force ('arabic' or 'latin')
Returns:
Normalized text
"""
if not text:
return ''
# Detect script if not forced
script = force_script or self.detect_script(text)
# Apply appropriate normalization based on script
if script == 'arabic':
text = self.normalize_arabic(text)
elif script == 'latin':
text = self.normalize_latin(text)
elif script == 'mixed':
# For mixed text, apply both normalizations
text = self.normalize_arabic(text)
text = self.normalize_latin(text)
# Always normalize punctuation and whitespace
text = self.normalize_punctuation(text)
return text
def get_normalization_info(self, text: str) -> Dict[str, str]:
"""
Get information about the normalization process for debugging.
Args:
text: Input text
Returns:
Dictionary containing normalization information
"""
original_text = text
script = self.detect_script(text)
normalized_text = self.normalize(text)
return {
'original_text': original_text,
'detected_script': script,
'normalized_text': normalized_text,
'length_before': str(len(original_text)),
'length_after': str(len(normalized_text))
}
```

---------------------------------------------------------------------------