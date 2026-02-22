# Data cleaning logic
# Preprocessing: Ensure dataset_models correctly transform raw CSV/JSON rows into your GuardrailRequest objects.

"""
test_ptbr_academic_dataset_preprocessing.py: Unit tests for PT-BR Academic dataset preprocessing.

This module tests the data cleaning and preprocessing logic, ensuring that raw
JSON data is correctly transformed into GuardrailRequest objects with proper
metadata extraction and categorization.
"""

import pytest
import json
import tempfile
from pathlib import Path
from typing import Dict, Any
from collections.abc import Iterator

from datasets.ptbr_academic_dataset import PTBRAcademicDataset
from core.schema import GuardrailRequest


@pytest.mark.unit
class TestPTBRAcademicDatasetLoad:
    """Tests for loading PT-BR Academic dataset from JSON."""
    
    @pytest.fixture
    def sample_dataset_dict(self) -> Dict[str, Any]:
        """Create a sample PT-BR Academic dataset structure."""
        return {
            "safe": [
                {
                    "id": "safe_001",
                    "parameters": {
                        "aspect": "technical",
                        "style": "formal"
                    },
                    "message": "Can you help me with Python programming?",
                    "explanation": "Standard technical question"
                }
            ],
            "unethical": [
                {
                    "id": "unethical_001",
                    "parameters": {
                        "aspect": "harassment",
                        "style": "aggressive"
                    },
                    "message": "You are stupid and worthless.",
                    "explanation": "Contains insulting language"
                }
            ],
            "off-topic": [
                {
                    "id": "offtopic_001",
                    "parameters": {
                        "aspect": "general",
                        "style": "casual"
                    },
                    "message": "What's your favorite pizza?",
                    "explanation": "Not related to academic discussion"
                }
            ]
        }
    
    @pytest.fixture
    def temp_json_file(self, sample_dataset_dict) -> Iterator[str]:
        """Create a temporary JSON file with sample dataset."""
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.json',
            delete=False,
            encoding='utf-8'
        ) as f:
            json.dump(sample_dataset_dict, f)
            temp_path = f.name
        
        yield temp_path
        
        # # Cleanup
        # Path(temp_path).unlink()
    
    def test_load_valid_dataset(self, temp_json_file):
        """Test loading a valid PT-BR Academic dataset from JSON."""
        dataset = PTBRAcademicDataset()
        raw_data = dataset.load(temp_json_file)
        
        assert len(raw_data) == 3
        assert dataset._raw_data is not None
        assert isinstance(raw_data, list)
    
    def test_load_file_not_found(self):
        """Test that FileNotFoundError is raised for non-existent files."""
        dataset = PTBRAcademicDataset()
        
        with pytest.raises(FileNotFoundError):
            dataset.load("/nonexistent/path/dataset.json")
    
    def test_load_invalid_file_extension(self, temp_json_file):
        """Test that ValueError is raised for non-JSON files."""
        # Rename to .txt
        txt_path = temp_json_file.replace(".json", ".txt")
        Path(temp_json_file).rename(txt_path)
        
        dataset = PTBRAcademicDataset()
        
        with pytest.raises(ValueError, match="Expected JSON file"):
            dataset.load(txt_path)
        
        # # Cleanup
        # Path(txt_path).unlink()
    
    def test_load_invalid_json_format(self):
        """Test that ValueError is raised for invalid JSON."""
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.json',
            delete=False,
            encoding='utf-8'
        ) as f:
            f.write("{ invalid json content }")
            temp_path = f.name
        
        dataset = PTBRAcademicDataset()
        
        try:
            with pytest.raises(ValueError, match="Invalid JSON format"):
                dataset.load(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_load_categories_are_flattened(self, temp_json_file):
        """Test that categorized data is correctly flattened."""
        dataset = PTBRAcademicDataset()
        raw_data = dataset.load(temp_json_file)
        
        # Check that all items have category field
        for item in raw_data:
            assert "category" in item
            assert item["category"] in ["safe", "unethical", "off-topic", "unsafe"]


@pytest.mark.unit
class TestPTBRAcademicDatasetConversion:
    """Tests for converting raw data to GuardrailRequest objects."""
    
    @pytest.fixture
    def dataset_with_data(self, sample_dataset_dict) -> Iterator[PTBRAcademicDataset]:
        """Create a dataset with loaded data."""
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.json',
            delete=False,
            encoding='utf-8'
        ) as f:
            json.dump(sample_dataset_dict, f)
            temp_path = f.name
        
        dataset = PTBRAcademicDataset()
        dataset.load(temp_path)
        
        yield dataset
        
        Path(temp_path).unlink()
    
    @pytest.fixture
    def sample_dataset_dict(self) -> Dict[str, Any]:
        """Create a sample PT-BR Academic dataset structure."""
        return {
            "safe": [
                {
                    "id": "safe_001",
                    "parameters": {
                        "aspect": "technical",
                        "style": "formal"
                    },
                    "message": "Can you help me with Python programming?",
                    "explanation": "Standard technical question"
                },
                {
                    "id": "safe_002",
                    "parameters": {
                        "aspect": "writing",
                        "style": "formal"
                    },
                    "message": "How do I improve my essay?",
                    "explanation": "Writing assistance request"
                }
            ],
            "unethical": [
                {
                    "id": "unethical_001",
                    "parameters": {
                        "aspect": "harassment",
                        "style": "aggressive"
                    },
                    "message": "You are stupid and worthless.",
                    "explanation": "Contains insulting language"
                },
                {
                    "id": "unethical_002",
                    "parameters": {
                        "aspect": "hate_speech",
                        "style": "hateful"
                    },
                    "message": "People from X should not be allowed to vote.",
                    "explanation": "Discriminatory statement"
                }
            ],
            "off-topic": [
                {
                    "id": "offtopic_001",
                    "parameters": {
                        "aspect": "general",
                        "style": "casual"
                    },
                    "message": "What's your favorite pizza?",
                    "explanation": "Not related to academic discussion"
                }
            ]
        }
    
    def test_to_requests_returns_list(self, dataset_with_data):
        """Test that to_requests returns a list of GuardrailRequest objects."""
        requests = dataset_with_data.to_requests()
        
        assert isinstance(requests, list)
        assert len(requests) == 5
        assert all(isinstance(req, GuardrailRequest) for req in requests)
    
    def test_to_requests_without_loading(self):
        """Test that ValueError is raised when calling to_requests before load."""
        dataset = PTBRAcademicDataset()
        
        with pytest.raises(ValueError, match="not loaded"):
            dataset.to_requests()
    
    def test_request_text_field(self, dataset_with_data):
        """Test that text field is correctly extracted from 'message'."""
        requests = dataset_with_data.to_requests()
        
        # Find the safe request
        safe_request = next(r for r in requests if "Can you help me" in r.text)
        assert safe_request.text == "Can you help me with Python programming?"
        
        # Find the unethical request
        unethical_request = next(r for r in requests if "stupid" in r.text)
        assert unethical_request.text == "You are stupid and worthless."
    
    def test_request_metadata_structure(self, dataset_with_data):
        """Test that metadata is correctly structured."""
        requests = dataset_with_data.to_requests()
        
        for request in requests:
            assert isinstance(request.metadata, dict)
            assert "source" in request.metadata
            assert request.metadata["source"] == "PTBRAcademicDataset"
            assert "original_id" in request.metadata
            assert "category" in request.metadata
            assert "aspect" in request.metadata
            assert "style" in request.metadata
            assert "explanation" in request.metadata
    
    def test_metadata_category_field(self, dataset_with_data):
        """Test that category field is correctly set in metadata."""
        requests = dataset_with_data.to_requests()
        
        # Check categories
        categories = set(req.metadata["category"] for req in requests)
        assert "safe" in categories
        assert "unethical" in categories
        assert "off-topic" in categories
    
    def test_metadata_parameters_extraction(self, dataset_with_data):
        """Test that parameters are correctly extracted to metadata."""
        requests = dataset_with_data.to_requests()
        
        # Find safe request
        safe_request = next(r for r in requests if "Can you help me" in r.text)
        assert safe_request.metadata["aspect"] == "technical"
        assert safe_request.metadata["style"] == "formal"
        
        # Find unethical request
        unethical_request = next(r for r in requests if "stupid" in r.text)
        assert unethical_request.metadata["aspect"] == "harassment"
        assert unethical_request.metadata["style"] == "aggressive"
    
    def test_metadata_explanation_field(self, dataset_with_data):
        """Test that explanation is correctly added to metadata."""
        requests = dataset_with_data.to_requests()
        
        safe_request = next(r for r in requests if "Can you help me" in r.text)
        assert safe_request.metadata["explanation"] == "Standard technical question"
    
    def test_request_preserves_all_samples(self, dataset_with_data):
        """Test that all samples are converted to requests."""
        requests = dataset_with_data.to_requests()
        
        # Should have 5 total (2 safe + 2 unethical + 1 off-topic)
        assert len(requests) == 5
    
    def test_conversion_does_not_modify_original_data(self, dataset_with_data):
        """Test that conversion doesn't modify the original raw data."""
        original_data = [item.copy() for item in dataset_with_data._raw_data]
        
        # requests = dataset_with_data.to_requests()
        
        # Verify raw data is unchanged
        assert len(dataset_with_data._raw_data) == len(original_data)


@pytest.mark.unit
class TestPTBRAcademicDatasetEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_missing_message_field(self):
        """Test error handling when 'message' field is missing."""
        dataset = PTBRAcademicDataset()
        dataset._raw_data = [
            {
                "id": "test_001",
                "category": "safe",
                # Missing 'message' field
                "explanation": "Test item"
            }
        ]
        
        with pytest.raises(KeyError, match="message"):
            dataset.to_requests()
    
    def test_empty_message_field(self):
        """Test error handling when 'message' field is empty."""
        dataset = PTBRAcademicDataset()
        dataset._raw_data = [
            {
                "id": "test_001",
                "message": "",  # Empty message
                "category": "safe",
                "explanation": "Test item"
            }
        ]
        
        with pytest.raises(KeyError, match="message"):
            dataset.to_requests()
    
    def test_missing_optional_metadata_fields(self):
        """Test that missing optional fields don't cause errors."""
        dataset = PTBRAcademicDataset()
        dataset._raw_data = [
            {
                "id": "test_001",
                "message": "Test message",
                "category": "safe"
                # Missing parameters and explanation
            }
        ]
        
        requests = dataset.to_requests()
        
        assert len(requests) == 1
        request = requests[0]
        assert request.text == "Test message"
        assert request.metadata["aspect"] is None
        assert request.metadata["style"] is None
        assert request.metadata["explanation"] == ""
    
    def test_dataset_with_empty_categories(self):
        """Test loading dataset with empty categories."""
        empty_data = {
            "safe": [],
            "unethical": [],
            "off-topic": []
        }
        
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.json',
            delete=False,
            encoding='utf-8'
        ) as f:
            json.dump(empty_data, f)
            temp_path = f.name
        
        try:
            dataset = PTBRAcademicDataset()
            raw_data = dataset.load(temp_path)
            requests = dataset.to_requests()
            
            assert len(raw_data) == 0
            assert len(requests) == 0
        finally:
            Path(temp_path).unlink()
    
    def test_dataset_with_special_characters(self):
        """Test handling of special characters in messages."""
        special_data = {
            "safe": [
                {
                    "id": "special_001",
                    "message": "What's the θ symbol in physics?",
                    "parameters": {"aspect": "technical", "style": "formal"},
                    "explanation": "Unicode and special characters"
                }
            ],
            "unethical": [],
            "off-topic": []
        }
        
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.json',
            delete=False,
            encoding='utf-8'
        ) as f:
            json.dump(special_data, f, ensure_ascii=False)
            temp_path = f.name
        
        try:
            dataset = PTBRAcademicDataset()
            dataset.load(temp_path)
            requests = dataset.to_requests()
            
            assert len(requests) == 1
            assert "θ" in requests[0].text
        finally:
            Path(temp_path).unlink()
    
    def test_dataset_with_multiple_unsafe_categories(self):
        """Test dataset that includes 'unsafe' category alongside others."""
        multi_category_data = {
            "safe": [
                {
                    "id": "safe_001",
                    "message": "How do I solve this equation?",
                    "parameters": {"aspect": "math", "style": "formal"},
                    "explanation": "Safe question"
                }
            ],
            "unsafe": [
                {
                    "id": "unsafe_001",
                    "message": "Harmful content here",
                    "parameters": {"aspect": "harmful", "style": "aggressive"},
                    "explanation": "Unsafe content"
                }
            ],
            "unethical": [],
            "off-topic": []
        }
        
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.json',
            delete=False,
            encoding='utf-8'
        ) as f:
            json.dump(multi_category_data, f)
            temp_path = f.name
        
        try:
            dataset = PTBRAcademicDataset()
            raw_data = dataset.load(temp_path)
            requests = dataset.to_requests()
            
            assert len(raw_data) == 2
            assert len(requests) == 2
            
            # Check that both categories are present
            categories = set(req.metadata["category"] for req in requests)
            assert "safe" in categories
            assert "unsafe" in categories
        finally:
            Path(temp_path).unlink()


@pytest.mark.unit
class TestPTBRAcademicDatasetIntegration:
    """Integration tests for complete workflow."""
    
    def test_full_workflow_load_and_convert(self):
        """Test complete workflow from loading to conversion."""
        complete_data = {
            "safe": [
                {
                    "id": "s1",
                    "message": "Safe message 1",
                    "parameters": {"aspect": "a1", "style": "s1"},
                    "explanation": "Explanation 1"
                },
                {
                    "id": "s2",
                    "message": "Safe message 2",
                    "parameters": {"aspect": "a2", "style": "s2"},
                    "explanation": "Explanation 2"
                }
            ],
            "unethical": [
                {
                    "id": "u1",
                    "message": "Unethical message",
                    "parameters": {"aspect": "harassment", "style": "aggressive"},
                    "explanation": "Harmful content"
                }
            ],
            "off-topic": [
                {
                    "id": "o1",
                    "message": "Off-topic message",
                    "parameters": {"aspect": "general", "style": "casual"},
                    "explanation": "Not relevant"
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.json',
            delete=False,
            encoding='utf-8'
        ) as f:
            json.dump(complete_data, f)
            temp_path = f.name
        
        try:
            # Create dataset and load
            dataset = PTBRAcademicDataset()
            assert dataset.name == "PTBRAcademicDataset"
            
            raw_data = dataset.load(temp_path)
            assert len(raw_data) == 4
            
            # Convert to requests
            requests = dataset.to_requests()
            assert len(requests) == 4
            
            # Verify structure
            for req in requests:
                assert isinstance(req, GuardrailRequest)
                assert len(req.text) > 0
                assert req.metadata["source"] == "PTBRAcademicDataset"
            
            # Verify categorization
            categories = {}
            for req in requests:
                cat = req.metadata["category"]
                categories[cat] = categories.get(cat, 0) + 1
            
            assert categories["safe"] == 2
            assert categories["unethical"] == 1
            assert categories["off-topic"] == 1
        finally:
            Path(temp_path).unlink()
    
    def test_dataset_can_be_reused_after_load(self):
        """Test that dataset can load multiple files sequentially."""
        data1 = {
            "safe": [{"id": "1", "message": "First", "parameters": {}, "explanation": ""}],
            "unethical": [],
            "off-topic": []
        }
        data2 = {
            "safe": [],
            "unethical": [{"id": "2", "message": "Second", "parameters": {}, "explanation": ""}],
            "off-topic": []
        }
        
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.json',
            delete=False,
            encoding='utf-8'
        ) as f1:
            json.dump(data1, f1)
            temp_path1 = f1.name
        
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.json',
            delete=False,
            encoding='utf-8'
        ) as f2:
            json.dump(data2, f2)
            temp_path2 = f2.name
        
        try:
            dataset = PTBRAcademicDataset()
            
            # Load first file
            dataset.load(temp_path1)
            requests1 = dataset.to_requests()
            assert len(requests1) == 1
            assert requests1[0].metadata["category"] == "safe"
            
            # Load second file (should replace previous data)
            dataset.load(temp_path2)
            requests2 = dataset.to_requests()
            assert len(requests2) == 1
            assert requests2[0].metadata["category"] == "unethical"
        finally:
            Path(temp_path1).unlink()
            Path(temp_path2).unlink()
