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