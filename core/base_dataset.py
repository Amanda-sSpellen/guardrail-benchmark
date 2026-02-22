"""
base_dataset.py: Abstract base class for dataset loading and standardization.

This module defines the interface for dataset providers, ensuring that different
data sources (academic CSV, HuggingFace, JSON, etc.) can be normalized into a
consistent GuardrailRequest format for benchmarking.
"""

from abc import ABC, abstractmethod
from typing import List, Any, Dict
from core.schema import GuardrailRequest


class GuardrailDataset(ABC):
    """
    Abstract base class for guardrail dataset implementations.
    
    Subclasses implement loading logic for different data sources and standardize
    them into GuardrailRequest objects that can be evaluated by the benchmark engine.
    """
    
    def __init__(self, name: str, **kwargs):
        """
        Initialize the dataset loader.
        
        Args:
            name: Display name for the dataset
            **kwargs: Dataset-specific configuration
        """
        self.name = name
        self.config = kwargs
        self._raw_data = None
    
    @abstractmethod
    def load(self, path: str) -> List[Dict[str, Any]]:
        """
        Load raw data from the specified path.
        
        This method should handle loading data from various sources (CSV, JSON, HuggingFace, etc.)
        and return it as a list of dictionaries containing the raw data.
        
        Args:
            path: File path or identifier for the dataset
            
        Returns:
            List of dictionaries containing the raw dataset items
            
        Raises:
            FileNotFoundError: If the dataset path doesn't exist
            ValueError: If the dataset format is invalid
        """
        pass
    
    @abstractmethod
    def to_requests(self) -> List[GuardrailRequest]:
        """
        Standardize raw data into GuardrailRequest objects.
        
        This method transforms the raw data loaded via load() into the standard
        GuardrailRequest format. Each item in the raw data should be converted to
        a GuardrailRequest with:
        - text: The input string to evaluate
        - metadata: Dataset-specific metadata (original_label, source, etc.)
        
        Returns:
            List of standardized GuardrailRequest objects ready for evaluation
            
        Raises:
            ValueError: If raw_data hasn't been loaded yet
            KeyError: If required fields are missing from raw data
        """
        pass
    
    def load_and_standardize(self, path: str) -> List[GuardrailRequest]:
        """
        Convenience method to load and standardize data in one call.
        
        Args:
            path: File path or identifier for the dataset
            
        Returns:
            List of standardized GuardrailRequest objects
        """
        self._raw_data = self.load(path)
        return self.to_requests()
