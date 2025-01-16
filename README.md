# Language Detection

## 1. Preprocessing and Normalization

### 1.1 Text Cleaning

1. **Remove or replace unwanted tokens**:
    - URLs, HTML tags, social media tags (e.g., @username), emojis, punctuation, special characters, parentheses, numbers.
    - Keep only characters relevant to language detection (alphanumeric plus minimal punctuation, if needed).
2. **Normalize whitespace**:
    - Convert tabs, newlines, multiple spaces into a single space.
3. **Lowercasing** (for Latin scripts)
    - Convert text to lowercase for languages using the Latin alphabet.
4. **Arabic-specific normalization**:
    - Remove diacritics (tashkeel: َ ً ُ ٌ etc.).
    - Convert different forms of the same letter into a canonical form (e.g., “ى” and “ي” both normalized to “ي”).
    - Replace Arabic-specific punctuation forms with standard characters.

## 2. Quick Character Set Analysis

1. **Determine predominant character sets**:
    - Count Arabic script characters vs. Latin script characters (and optionally other scripts).
    - If Arabic characters > configurable threshold (e.g., 70%), **pre-classify** as Arabic family.
    - If Latin characters dominate, route to a different path (French, English).
2. **Mixed-content check**:
    - If no single script is overwhelmingly dominant, mark as potential “Mixed” and continue to deeper analysis.

## 3. Core Language Detection Methods

### 3.1 Dictionary-Based Detection (Short Text)

1. **Short-text threshold**:
    - If text length < X tokens (e.g., 3–5 words), dictionary-based approaches are often more reliable than statistical ones.
2. **Dictionary or lexicon approach**:
    - Maintain domain-specific word lists for Arabic, French, English, etc.
    - Check how many tokens map to each language dictionary.
3. **Heuristic rules**:
    - If a majority (or a weighted majority) of tokens match a given dictionary, classify accordingly.

### 3.2 Statistical Detection (Long Text)

1. **N-gram language modeling**:
    - Use classic approaches like character-level trigrams or word-level n-grams.
    - Build a language model or use existing language detection libraries (e.g., [langid](https://pypi.org/project/langid/), [langdetect](https://pypi.org/project/langdetect/)).
2. **TF-IDF + Classifier**:
    - Vectorize text (TF-IDF or count vector).
    - Train a multi-class model (e.g., Logistic Regression, Naive Bayes, SVM) on known labeled data for each language, including Darija samples.
3. **Compute probabilities**:
    - For each language model, compute a probability/score.
    - Select the language with the highest probability or combine results via ensemble methods.

## 4. Handling Mixed Language and Segmentation

1. **Identify language boundaries**:
    - If detection confidence for a single language is below a threshold, treat the text as potentially “Mixed.”
2. **Segmentation**:
    - Split text into smaller segments (e.g., sentence-level or phrase-level).
    - Run the detection model on each segment independently.
3. **Score aggregation**:
    - Each segment receives a language score distribution.
    - Aggregate (average or weighted average) segment scores across the text.
    - If a single language dominates above a configurable priority threshold, classify as that language.
    - Otherwise, tag as “Mixed.”
4. **Spelling correction (if needed)**:
    - If the text has many out-of-vocabulary words, apply a spelling correction / normalization step (especially relevant for user-generated text with typos).
    - Rerun detection on corrected tokens if needed.

## 5. Prioritization and Configurable Rules

Since you need **configurable prioritization** of complete languages (Arabic, French, English, Spanish, etc.):

1. **Set language priorities**:
    - For instance, if you must prefer “Arabic” (including Darija) over other languages when in doubt, define a threshold-based rule:
        - *If the final confidence for Arabic > P%*, classify as Arabic; otherwise proceed to the next probable language.
2. **Rule-based overrides**:
    - If certain keywords or features are found (e.g., Moroccan-specific slang indicating Darija usage), weight the overall score in favor of Arabic.
    - Conversely, if the text is mostly French except for a few Arabic words, final classification can remain “French,” or become “Mixed” if the few Arabic words are significant enough.

## 6. Model Training and Fine-Tuning

1. **Collect representative data**:
    - Include both standard Arabic and Darija samples with diverse writing styles.
    - Include French, English, Spanish data from Moroccan context (borrowed vocabulary, code-switching, etc.).
2. **Annotation / labeling**:
    - Properly label each text instance or segment.
    - Mark code-switched examples as “Mixed” or multi-label.
3. **Train or fine-tune**:
    - For dictionary-based approaches, expand your dictionary with real user-generated slang.
    - For statistical approaches (n-grams, TF-IDF + classifiers, or advanced language models), retrain using the labeled data.
4. **Evaluation**:
    - Evaluate on a held-out set of real user posts.
    - Measure metrics like Precision, Recall, and F1 for each language, plus a special “Mixed” category.

```mermaid
flowchart LR
    A[Raw Text Input] --> B[Preprocessing & Normalization]
    B --> C{Is Text Empty?}
    C -- "Yes" --> M[Label = EMPTY]
    C -- "No" --> D{Short Text?}
    D -- "Yes" --> E[Optional Dictionary-Based Detection]
    D -- "No" --> H[fastText Model Prediction]
    E --> F{Confident Dictionary Result?}
    F -- "No" --> H
    F -- "Yes" --> G[Label = {ar, fr, en}]
    H --> I{Confidence >= Threshold?}
    I -- "No" --> J[Label = UNKNOWN]
    I -- "Yes" --> K[Label = {ar, fr, en}]
    M & G & J & K --> L[End]
```
