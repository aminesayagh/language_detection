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

