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
