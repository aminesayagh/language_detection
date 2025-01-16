import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.corpus import wordnet as wn
import os

def extract_keywords(language: str) -> set:
    """
    Extract keywords from the given language.
    """
    keywords = set()
    for synset in wn.all_synsets(pos=wn.NOUN):  # You can use other POS: VERB, ADJ, ADV
        for lemma in synset.lemmas(language):  # Specify language
            keywords.add(lemma.name().lower().replace('_', ' '))
    return keywords


if __name__ == "__main__":

    english_keywords = extract_keywords("eng")
    french_keywords = extract_keywords("fra")
    
    # find if love exist in the intersection
    love = "love" in french_keywords
    print(love)

