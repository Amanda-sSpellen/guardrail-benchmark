"""
save_output.py: Results serialization and experiment artifact management.

Handles saving benchmark results, experiment metadata, and visualizations
to disk for reproducibility and archival.
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from dataclasses import asdict


def save_benchmark_results(
    results: Dict[str, Any],
    experiment_name: str,
    output_dir: str = "results",
    experiment_index: int | None = None,
) -> str:
    """
    Save benchmark results to a JSON file.
    
    Args:
        results: Dictionary containing benchmark results and metrics
        experiment_name: Name of the experiment for the filename
        output_dir: Directory to save results (default: "results")
        experiment_index: Optional experiment index to prepend to filename
        
    Returns:
        Path where results were saved
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    index_prefix = f"{experiment_index:03d}_" if experiment_index is not None else ""
    filename = output_path / f"{index_prefix}{experiment_name}_{timestamp}.json"
    
    # Serialize results, handling non-serializable types
    serializable_results = _make_serializable(results)
    
    with open(filename, "w") as f:
        json.dump(serializable_results, f, indent=2, default=str)
    
    return str(filename)


def save_experiment_metadata(
    metadata: Dict[str, Any],
    experiment_name: str,
    output_dir: str = "results",
    experiment_index: int | None = None,
) -> str:
    """
    Save experiment metadata and configuration.
    
    Args:
        metadata: Dictionary containing experiment configuration and metadata
        experiment_name: Name of the experiment
        output_dir: Directory to save metadata
        experiment_index: Optional experiment index to prepend to filename
        
    Returns:
        Path where metadata was saved
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    index_prefix = f"{experiment_index:03d}_" if experiment_index is not None else ""
    filename = output_path / f"{index_prefix}{experiment_name}_metadata_{timestamp}.json"
    
    serializable_metadata = _make_serializable(metadata)
    
    with open(filename, "w") as f:
        json.dump(serializable_metadata, f, indent=2, default=str)
    
    return str(filename)


def save_visualizations(
    visualizations: Dict[str, str],
    experiment_name: str,
    output_dir: str = "results/visualizations"
) -> List[str]:
    """
    Organize and document visualization files.
    
    This function creates a manifest of generated visualizations
    and stores their metadata for easy reference.
    
    Args:
        visualizations: Dictionary mapping visualization names to file paths
        experiment_name: Name of the experiment
        output_dir: Base directory for visualizations
        
    Returns:
        List of paths to saved visualization metadata files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create manifest
    manifest = {
        "experiment": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "visualizations": visualizations
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_path = output_path / f"{experiment_name}_manifest_{timestamp}.json"
    
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    return [str(manifest_path)]


def save_detailed_report(
    report: Any,
    experiment_name: str,
    output_dir: str = "results"
) -> str:
    """
    Save a comprehensive benchmark report with all details.
    
    Args:
        report: BenchmarkReport object or similar with benchmark results
        experiment_name: Name of the experiment
        output_dir: Directory to save the report
        
    Returns:
        Path where report was saved
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Convert report to dictionary if it has a to_dict method
    if hasattr(report, "to_dict"):
        report_dict = report.to_dict()
    elif hasattr(report, "__dataclass_fields__"):
        report_dict = asdict(report)
    else:
        report_dict = report if isinstance(report, dict) else {"report": str(report)}
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_path / f"{experiment_name}_report_{timestamp}.json"
    
    serializable_report = _make_serializable(report_dict)
    
    with open(filename, "w") as f:
        json.dump(serializable_report, f, indent=2, default=str)
    
    return str(filename)


def _make_serializable(obj: Any) -> Any:
    """
    Convert non-JSON-serializable objects to serializable equivalents.
    
    Handles dataclasses, datetime objects, and complex nested structures.
    
    Args:
        obj: Object to make serializable
        
    Returns:
        Serializable version of the object
    """
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    
    elif isinstance(obj, datetime):
        return obj.isoformat()
    
    elif hasattr(obj, "__dataclass_fields__"):
        return _make_serializable(asdict(obj))
    
    elif hasattr(obj, "to_dict"):
        return _make_serializable(obj.to_dict())
    
    elif hasattr(obj, "__dict__"):
        return _make_serializable(obj.__dict__)
    
    else:
        return obj


def create_experiment_summary(
    experiment_name: str,
    models: List[str],
    dataset: str,
    metrics: Dict[str, Dict[str, float]],
    output_dir: str = "results"
) -> str:
    """
    Create a human-readable summary of experiment results.
    
    Args:
        experiment_name: Name of the experiment
        models: List of model names evaluated
        dataset: Name of the dataset used
        metrics: Dictionary of metrics by model
        output_dir: Directory to save the summary
        
    Returns:
        Path where summary was saved
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = output_path / f"{experiment_name}_summary_{timestamp}.txt"
    
    with open(summary_file, "w") as f:
        f.write(f"{'='*60}\n")
        f.write(f"Experiment Summary: {experiment_name}\n")
        f.write(f"{'='*60}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
        
        f.write("Configuration:\n")
        f.write(f"  Dataset: {dataset}\n")
        f.write(f"  Models: {', '.join(models)}\n\n")
        
        f.write("Results:\n")
        for model_name, model_metrics in metrics.items():
            f.write(f"\n  {model_name}:\n")
            for metric_name, value in model_metrics.items():
                f.write(f"    {metric_name}: {value:.4f}\n")
        
        f.write(f"\n{'='*60}\n")
    
    return str(summary_file)
