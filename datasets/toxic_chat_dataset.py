"""
toxic_chat_dataset.py: Implementation for loading ToxicChat dataset from HuggingFace.

This module handles the ToxicChat dataset, converting it from HuggingFace format
into standardized GuardrailRequest objects for benchmarking.
"""

import json
from typing import List, Any, Dict
from pathlib import Path
import sys

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.schema import GuardrailRequest
from core.base_dataset import GuardrailDataset


class ToxicChatDataset(GuardrailDataset):
    """
    Dataset handler for HuggingFace ToxicChat dataset.
    
    Loads ToxicChat data and converts it to GuardrailRequest format.
    The ToxicChat dataset contains conversations with toxicity labels.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the ToxicChat dataset loader.
        
        Args:
            **kwargs: Additional dataset-specific configuration
        """
        super().__init__(name="ToxicChat", **kwargs)
    
    def load(self, path: str) -> List[Dict[str, Any]]:
        """
        Load ToxicChat dataset from file.
        
        Supports loading from JSON files or can be extended to load directly
        from HuggingFace datasets library.
        
        Args:
            path: Path to the dataset file (JSON format) or HuggingFace identifier
            
        Returns:
            List of dictionaries containing dataset items
            
        Raises:
            FileNotFoundError: If the file path doesn't exist
            ValueError: If the file format is invalid
        """
        path_obj = Path(path)
        
        # Handle local JSON files
        if path_obj.exists() and path_obj.suffix == '.json':
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # If data is a dictionary with a 'data' key, extract it
                if isinstance(data, dict) and 'data' in data:
                    self._raw_data = data['data']
                else:
                    self._raw_data = data if isinstance(data, list) else [data]
                return self._raw_data
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON format in {path}: {e}")
        elif path_obj.exists():
            raise ValueError(f"Unsupported file format: {path_obj.suffix}. Expected .json")
        else:
            # TODO: Could add HuggingFace loading here
            raise FileNotFoundError(f"Dataset not found at {path}")
    
    def to_requests(self) -> List[GuardrailRequest]:
        """
        Convert raw ToxicChat data to GuardrailRequest objects.
        
        Expected raw data format:
        [
            {
                "id": int,
                "messages": [{"role": "user"/"assistant", "content": str}],
                "toxic": bool,  # or toxicity level
                ...
            },
            ...
        ]
        
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
            # Extract the conversation text
            text = self._extract_text(item)
            
            # Build metadata
            metadata = {
                'source': 'ToxicChat',
                'original_id': item.get('id'),
                'toxic': item.get('toxic', False),
            }
            
            # Add any additional fields as metadata
            for key, value in item.items():
                if key not in ['id', 'messages', 'content', 'text'] and not key.startswith('_'):
                    metadata[key] = value
            
            request = GuardrailRequest(text=text, metadata=metadata)
            requests.append(request)
        
        return requests
    
    def _extract_text(self, item: Dict[str, Any]) -> str:
        """
        Extract text content from a ToxicChat item.
        
        Handles various possible formats:
        - messages list format
        - direct text/content field
        
        Args:
            item: Raw dataset item
            
        Returns:
            Extracted text string
        """
        # Try messages format (conversation)
        if 'messages' in item and isinstance(item['messages'], list):
            messages = item['messages']
            # Extract user messages and assistant responses
            text_parts = []
            for msg in messages:
                if isinstance(msg, dict) and 'content' in msg:
                    text_parts.append(msg['content'])
            if text_parts:
                return '\n'.join(text_parts)
        
        # Try direct text/content fields
        if 'text' in item:
            return item['text']
        if 'content' in item:
            return item['content']
        if 'message' in item:
            return item['message']
        
        # Fallback: concatenate all string values
        text_parts = [str(v) for v in item.values() if isinstance(v, str)]
        if text_parts:
            return ' '.join(text_parts)
        
        raise KeyError(f"Could not extract text from item: {item}")
