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
        for lang in languages:
            language_dictionaries[lang] = extract_keywords(lang)
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