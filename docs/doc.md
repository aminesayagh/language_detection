# Code Documentation
Generated on: 2025-01-16T18:22:31.567Z
Total files: 12

## Project Structure

```
└── language_detection
    ├── __init__.py
    ├── config
    │   └── settings.py
    ├── core
    │   ├── __pycache__
    │   │   └── script_analyzer.cpython-310.pyc
    │   └── script_analyzer.py
    ├── index.py
    ├── tests
    │   └── test_detector.py
    └── utils
        ├── DataLoader.py
        ├── __pycache__
        │   └── DataLoader.cpython-310.pyc
        └── text
            ├── TextCleaner.py
            ├── TextNormalizer.py
            └── __pycache__
                ├── TextCleaner.cpython-310.pyc
                └── TextNormalizer.cpython-310.pyc
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
- Size: 7.98 KB
- Extension: .py
- Lines of code: 168

```py
from utils.DataLoader import DataLoader
from utils.text.TextCleaner import TextCleaner
from utils.text.TextNormalizer import TextNormalizer
from core.script_analyzer import ScriptAnalyzer
import pandas as pd
import logging
from typing import Dict
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

## File: TextCleaner.cpython-310.pyc
- Path: `/root/git/language_detection/utils/text/__pycache__/TextCleaner.cpython-310.pyc`
- Size: 4.14 KB
- Extension: .pyc
- Lines of code: 35

```pyc
o
    �>�g8  �                   @   s.   d dl Z d dlZd dlmZ G dd� d�ZdS )�    N)�Optionalc                
   @   s�   e Zd ZdZdd� Zdedefdd�Zdedefdd	�Zdedefd
d�Zdedefdd
�Z	dedefdd�Z
ddededefdd�Zdedefdd�Z
dedefdd�Zdedefdd�Z		d dee dededee fdd�ZdS )!�TextCleanerz�
A class to handle text cleaning and normalization operations for language detection.
Focuses on removing unwanted tokens while preserving language-relevant characters.
c                 C   sd   t �d�| _t �d�| _t �d�| _t �d�| _t �d�| _t �d�| _t �d�| _t �d�| _	d S )	NzPhttp[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+z@[\w_]+z#[\w_]+z<[^>]+>u   (?:^|\s)[0-9٠-٩]+(?:\s|$)z[^\w\s.,!?\'"-]z\s+u   [\(\)\[\]\{\}⟨⟩«»])
�re�compile�url_pattern�mention_pattern�hashtag_pattern�html_pattern�numbers_pattern�special_chars_pattern�whitespace_pattern�parentheses_pattern)�self� r   �6/root/git/language_detection/utils/text/TextCleaner.py�__init__   s   �zTextCleaner.__init__�text�returnc                 C   �   | j �d|�S )zRemove URLs from text.� )r   �sub�r   r   r   r   r   �remove_urls(   �   zTextCleaner.remove_urlsc                 C   s   | j �d|�}| j�d|�S )zRemove mentions and hashtags.r   )r   r   r   r   r   r   r   �remove_social_media_elements,   s   z(TextCleaner.remove_social_media_elementsc                 C   r   )zRemove HTML tags.r   )r	   r   r   r   r   r   �remove_html_tags1   r   zTextCleaner.remove_html_tagsc                 C   s   t �|d�S )z&Remove emojis using the emoji library.� )�emoji�
replace_emojir   r   r   r   �
remove_emojis5   s   zTextCleaner.remove_emojisc                 C   r   )z.Remove numeric digits (both Arabic and Latin).r   )r
   r   r   r   r   r   �remove_numbers9   r   zTextCleaner.remove_numbersT�keep_minimal_punctc                 C   s    |s	t �dd|�S | j�d|�S )a  
Remove special characters and punctuation.
Args:
text: Input text to clean
keep_minimal_punct: If True, keeps basic punctuation that might be relevant
for language detection
z[^\w\s]r   )r   r   r   )r   r   r!   r   r   r   �remove_special_characters=   s   	z%TextCleaner.remove_special_charactersc                 C   r   )z$Remove various types of parentheses.r   )r
   r   r   r   r   r   �remove_parenthesesK   r   zTextCleaner.remove_parenthesesc                 C   s   | j �d|�� �S )z;Convert multiple whitespace characters into a single space.r   )r   r   �stripr   r   r   r   �normalize_whitespaceO   s   z TextCleaner.normalize_whitespacec                 C   s   |� t�ddd��S )zRemove Arabic diacritics.r   uR   ًٌٍَُِّْٕٖٜٟٓٔٗ٘ٙٚٛٝٞ٠١٢٣٤٥٦٧٨٩٠١٢٣٤٥٦٧٨٩)�	translate�str�	maketransr   r   r   r   �remove_diacriticsT   s   zTextCleaner.remove_diacritics�   �
min_lengthc                 C   sr   |sdS | � |�}| �|�}| �|�}| �|�}| �|�}| �|�}| �||�}| �|�}t|�	� �|kr7|S dS )av  
Apply all cleaning operations in sequence.
Args:
text: Input text to clean
keep_minimal_punct: Whether to keep minimal punctuation
min_length: Minimum length of text to consider non-empty
Returns:
Cleaned text or None if the input is None or becomes empty after cleaning
N)
r   r   r   r   r    r#   r"   r%   �lenr$   )r   r   r!   r+   r   r   r   �
clean_textX   s   






zTextCleaner.clean_textN)T)Tr*   )�__name__�
__module__�__qualname__�__doc__r   r'   r   r   r   r   r    �boolr"   r#   r%   r)   r   �intr-   r   r   r   r   r      s.    �����r   )r   r   �typingr   r   r   r   r   r   �<module>   s    
```

---------------------------------------------------------------------------

## File: TextNormalizer.cpython-310.pyc
- Path: `/root/git/language_detection/utils/text/__pycache__/TextNormalizer.cpython-310.pyc`
- Size: 4.93 KB
- Extension: .pyc
- Lines of code: 82

```pyc
o
    �9�g�  �                   @   s*   d dl Z d dlmZmZ G dd� d�ZdS )�    N)�Dict�Optionalc                   @   s�   e Zd ZdZdd� Zdedefdd�Zdedefdd	�Zdedefd
d�Zdedefdd
�Z	ddede
e defdd�Zdedeeef fdd�Z
dS )�TextNormalizerz�
A class to normalize text across different languages (Arabic, French, English).
Handles tasks like lowercasing, diacritic removal, and character normalization.
c                 C   s�  i dd�dd�dd�dd�dd�dd�dd�d	d
�dd
�d
d
�dd
�d
d
�dd
�dd
�dd
�dd
�dd
�i dd
�dd
�dd
�dd
�dd�dd�dd�dd�dd�dd�dd�dd�d d�d!d�d"d�d#d�d$d��i d%d�d&d'�d(d'�d)d'�d*d'�d+d,�d-d,�d.d,�d/d,�d0d1�d2d1�d3d1�d4d1�d5d1�d6d1�d7d1�d8d1��d1d1d1d
d'd9��| _ t�d:�| _t�d;�| _t�d<�| _t�d=�| _t�d>�| _d S )?Nu   أu   اu   إu   آu   ٱu   ٲu   ٳu   ٵ�   ىu   يu   ئu   ۍu   ێu   ېu   ٸu   ﯨu   ﯩu   ﻰu   ﻱu   ﻲu   ﻳu   ﻴu   ؤu   وu   ٶu   ٷu   ۄu   ۅu   ۆu   ۇu   ۈu   ۉu   ۊu   ۋu   ۏu   ݸu   ݹu   ء� u   ٔu   ٕu   ٚu   ةu   هu   ۃu   ەu   ۀu   كu   کu   ڪu   ػu   ؼu   ڬu   ڭu   ڮu   ݢ)u   ݣu   ݤu   ݿr   u   ـz[\u064B-\u065F\u0670]u   [؀-ۿ]z[a-zA-Z]z\s+z([!?,.]){2,})�arabic_chars_map�re�compile�arabic_diacritics�arabic_chars�latin_chars�consecutive_spaces�consecutive_punctuation)�self� r   �9/root/git/language_detection/utils/text/TextNormalizer.py�__init__
   s�   ���������������������	�	�	�	�	�	�	�
�
�
�
�
�
�
�
�
�
�
�������������
� zTextNormalizer.__init__�text�returnc                 C   s2   | j �d|�}| j�� D ]
\}}|�||�}q|S )z�
Normalize Arabic text by removing diacritics and standardizing characters.
Args:
text: Input Arabic text
Returns:
Normalized Arabic text
r   )r
   �subr   �items�replace)r   r   �original�
normalizedr   r   r   �normalize_arabic4   s   zTextNormalizer.normalize_arabicc                 C   s   |� � S )z�
Normalize Latin script text (French/English).
Args:
text: Input text in Latin script
Returns:
Normalized text
)�lower�r   r   r   r   r   �normalize_latinG   s   zTextNormalizer.normalize_latinc                 C   s$   | j �d|�}| j�d|�}|�� S )z�
Normalize punctuation and whitespace.
Args:
text: Input text
Returns:
Text with normalized punctuation and whitespace
� z\1)r
   r   r   �stripr   r   r   r   �normalize_punctuationT   s   z$TextNormalizer.normalize_punctuationc                 C   sX   t | j�|��}t | j�|��}|| }|dkrdS || }|dkr$dS |dk r*dS dS )z�
Detect the predominant script in the text.
Args:
text: Input text
Returns:
'arabic' for Arabic script, 'latin' for Latin script,
'mixed' if no clear predominance
r   �unknowngffffff�?�arabicg333333�?�latin�mixed)�lenr   �findallr   )r   r   �arabic_count�latin_count�total�arabic_ratior   r   r   �
detect_scriptg   s   zTextNormalizer.detect_scriptN�force_scriptc                 C   sh   |sdS |p
| � |�}|dkr| �|�}n|dkr| �|�}n|dkr-| �|�}| �|�}| �|�}|S )a  
Main normalization method that handles text in any supported script.
Args:
text: Input text to normalize
force_script: Optional script type to force ('arabic' or 'latin')
Returns:
Normalized text
r   r"   r#   r$   )r+   r   r   r    )r   r   r,   �scriptr   r   r   �	normalize�   s   


zTextNormalizer.normalizec                 C   s8   |}| � |�}| �|�}|||tt|��tt|��d�S )z�
Get information about the normalization process for debugging.
Args:
text: Input text
Returns:
Dictionary containing normalization information
)�
original_text�detected_script�normalized_text�
length_before�length_after)r+   r.   �strr%   )r   r   r/   r-   r1   r   r   r   �get_normalization_info�   s   



�z%TextNormalizer.get_normalization_info)N)�__name__�
__module__�__qualname__�__doc__r   r4   r   r   r    r+   r   r.   r   r5   r   r   r   r   r      s    *
 r   )r   �typingr   r   r   r   r   r   r   �<module>   s    
```

---------------------------------------------------------------------------