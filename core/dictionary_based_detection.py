import os
import string
from collections import defaultdict
from typing import Dict, List, Set

class DictionaryBasedDetection:
    def __init__(self, dictionary_path: str, languages: List[str], short_text_threshold: int = 3):
        """
        Initialize the detector with language dictionaries.

        :param dictionaries_path: Path to the directory containing language dictionary files.
        :param languages: List of language codes or names (e.g., ['english', 'french', 'arabic']).
        :param short_text_threshold: Maximum number of tokens to use dictionary-based detection.
        """
        self.languages = languages
        self.short_text_threshold = short_text_threshold
        self.language_dictionaries = self.load_dictionaries(dictionary_path, languages)
        
    def load_dictionaries(self, path: str, languages: List[str]):
        """
        Load word lists for each language into a dictionary.

        :param path: Directory path containing dictionary files.
        :param languages: List of language names corresponding to dictionary files.
        :return: A dictionary mapping language names to sets of words.
        """
        language_dict = {}
        for lang in languages:
            file_path = os.path.join(path, f"{lang}.txt")
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    words = set(line.strip().lower() for line in file if line.strip())
                    language_dict[lang] = words
                print(f"Loaded dictionary for language: {lang} with {len(words)} words.")
            except FileNotFoundError:
                print(f"Dictionary file for language '{lang}' not found at {file_path}.")
                language_dict[lang] = set()
        return language_dict
        
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize the input text into words.

        :param text: The text to tokenize.
        :return: A list of lowercased tokens without punctuation.
        """
        tokens = text.lower().split()
        return tokens
    
    def detect_language(self, text: str):
        """
        Detect the language of the input text using dictionary-based detection.

        :param text: The text to detect.
        :return: The detected language code.
        """
        tokens = self.tokenize(text)
        num_tokens = len(tokens)
        
        if num_tokens >= self.short_text_threshold:
            return "unknown"
        
        for token in tokens:
            for lang, vocab in self.language_dictionaries.items():
                if token in vocab:
                    language_counts[lang] += 1
                    
        if not language_matches:
            return "unknown"
        
        detected_language = max(language_matches, key=lambda lang: language_matches[lang]) # get the language with the most matches
        max_matches = language_matches[detected_language] # get the number of matches for the detected language
        total_matches = sum(language_matches.values()) # get the total number of matches
        confidence = max_matches / total_matches # calculate the confidence
        
        return detected_language, confidence # return the detected language and confidence