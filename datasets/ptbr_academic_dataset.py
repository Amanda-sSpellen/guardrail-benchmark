"""
ptbr_academic_dataset.py: Implementation for loading PT-BR Academic guardrail dataset.

Handles the loading and processing of the Portuguese academic dataset JSON file
into GuardrailRequest objects. The dataset contains categorized messages
(safe, unethical, off-topic) with metadata.
"""

import json
from typing import List, Any, Dict
from pathlib import Path
import sys

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.schema import GuardrailRequest
from core.base_dataset import GuardrailDataset


class PTBRAcademicDataset(GuardrailDataset):
    """
    Dataset handler for Portuguese Brazilian academic guardrail dataset.
    
    Loads JSON data with categorized messages (safe, unethical, off-topic)
    and converts them to GuardrailRequest format.
    
    Dataset Schema:
    {
        "safe": [...],
        "unethical": [...],
        "off-topic": [...]
    }
    
    Each item has: id, parameters (aspect, style), message, explanation
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the PT-BR Academic dataset loader.
        
        Args:
            **kwargs: Additional dataset-specific configuration
        """
        super().__init__(name="PTBRAcademicDataset", **kwargs)
    
    def load(self, path: str) -> List[Dict[str, Any]]:
        """
        Load PT-BR Academic dataset from JSON file.
        
        Args:
            path: Path to the dataset JSON file
            
        Returns:
            List of dictionaries containing all dataset items (flattened from categories)
            
        Raises:
            FileNotFoundError: If the file path doesn't exist
            ValueError: If the file format is invalid
        """
        path_obj = Path(path)
        
        if not path_obj.exists():
            raise FileNotFoundError(f"Dataset not found at {path}")
        
        if path_obj.suffix != '.json':
            raise ValueError(f"Expected JSON file, got {path_obj.suffix}")
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {path}: {e}")
        
        # Flatten the categorized structure into a single list
        self._raw_data = self._flatten_categorized_data(data)
        return self._raw_data
    
    def to_requests(self) -> List[GuardrailRequest]:
        """
        Convert raw PT-BR Academic data to GuardrailRequest objects.
        
        Each item is converted to a GuardrailRequest with:
        - text: the 'message' field
        - metadata: id, category (safe/unethical/off-topic), parameters, explanation
        
        Returns:
            List of GuardrailRequest objects
            
        Raises:
            ValueError: If raw_data hasn't been loaded
            KeyError: If required fields are missing
        """
        if self._raw_data is None:
            raise ValueError("Data not loaded. Call load() first.")
        
        requests = []
        
        for item in self._raw_data:
            text = item.get('message', '')
            if not text:
                raise KeyError(f"Missing 'message' field in item: {item}")
            
            # Build metadata with category and parameters
            metadata = {
                'source': 'PTBRAcademicDataset',
                'original_id': item.get('id'),
                'category': item.get('category'),  # safe, unethical, off-topic
                'aspect': item.get('parameters', {}).get('aspect'),
                'style': item.get('parameters', {}).get('style'),
                'explanation': item.get('explanation', ''),
            }
            
            request = GuardrailRequest(text=text, metadata=metadata)
            requests.append(request)
        
        return requests
    
    def _flatten_categorized_data(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Flatten the categorized dataset structure into a single list.
        
        Converts from:
        {
            "safe": [...],
            "unethical": [...],
            "off-topic": [...]
        }
        
        To:
        [
            {..., "category": "safe"},
            {..., "category": "unethical"},
            {..., "category": "off-topic"},
            ...
        ]
        
        Args:
            data: Dictionary with category keys
            
        Returns:
            Flattened list of items with category field added
        """
        flattened = []
        
        valid_categories = ['safe', 'unethical', 'off-topic', 'unsafe']
        
        for category in valid_categories:
            if category in data and isinstance(data[category], list):
                for item in data[category]:
                    # Add category to each item
                    item_copy = item.copy() if isinstance(item, dict) else {}
                    item_copy['category'] = category
                    flattened.append(item_copy)
        
        return flattened