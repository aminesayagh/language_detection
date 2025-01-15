

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