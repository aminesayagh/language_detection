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