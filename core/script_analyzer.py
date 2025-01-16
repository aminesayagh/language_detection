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