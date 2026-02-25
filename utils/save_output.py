"""
save_output.py: Results serialization and experiment artifact management.

Handles saving benchmark results, experiment metadata, and visualizations
to disk for reproducibility and archival.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from dataclasses import asdict
from loguru import logger

from core.base_model import GuardrailModel
from core.schema import GuardrailRequest, GuardrailResponse


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


def save_results(
        model_name: str, 
        experiment_name: str,
        dataset_name: str,
        requests: list, 
        responses: list, 
        metrics: dict, 
        output_dir: Path, 
        experiment_index: int, 
        detailed: bool = True,
        model: Optional[GuardrailModel] = None,
    ) -> str | tuple[str, str]:
    """
    Save benchmark results to JSON file.
    
    Args:
        model_name: Name of the model being evaluated
        requests: List of input requests
        responses: List of model responses
        metrics: Dictionary of metrics for each model
        output_dir: Output directory path
        experiment_index: Index of the experiment
        detailed: If True, include per-sample predictions; if False, only metrics
        
    Returns:
        Path to saved results file
    """
    logger.info("Saving results")
    
    results = {
        "experiment": experiment_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "dataset": dataset_name,
        "system_prompt": model.system_prompt if model is not None else "unknown",
        "categories": model.categories if model is not None else "unknown",
        "metrics": metrics,
    }
    results_path = ""
    detailed_metadata_path = ""
    
    results_path = save_benchmark_results(
        results=results,
        experiment_name=experiment_name,
        output_dir=str(output_dir / f"{experiment_index:03d}" ),
        experiment_index=experiment_index,
    )
    
    if detailed:
        predictions = [
            {
                "request_text": req.text,
                "request_metadata": req.metadata,
                "prediction": resp.model_dump(),
            }
            for req, resp in zip(requests, responses) 
        ]

        detailed_metadata_path = save_experiment_metadata(
            metadata={
                "model": model_name,
                "dataset": dataset_name,
                "total_samples": len(requests),
                "metrics": metrics,
                "predictions": predictions,
            },
            experiment_name=f"{experiment_name}_predictions",
            output_dir=str(output_dir / f"{experiment_index:03d}"),
            experiment_index=experiment_index,
        )
        logger.info(f"Saved detailed metadata to {detailed_metadata_path}")
    
    logger.info(f"Saved results to {results_path}")
    
    return results_path if not detailed else (results_path, detailed_metadata_path)

def get_experiment_index(output_dir: str) -> int:
    """Determine the next experiment index based on existing results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    existing_indices = []
    for item in output_path.iterdir():
        if item.is_dir() and item.name.isdigit():
            existing_indices.append(int(item.name))
    
    next_index = max(existing_indices, default=-1) + 1
    return next_index


def load_results(results_file: str) -> Dict[str, Any]:
    """
    Load results and return the `metrics` object saved by `save_results`.

    This function mirrors `save_results` by reading a JSON file produced
    via `save_benchmark_results` / `save_experiment_metadata` and returning
    the metrics structure saved therein. If the provided file does not
    directly contain a `metrics` key, the function will search nested
    structures for the first occurrence of a `metrics` key.

    Args:
        results_file: Path to the JSON file created by `save_results`.

    Returns:
        The metrics dictionary extracted from the file.

    Raises:
        FileNotFoundError: If `results_file` does not exist.
        ValueError: If no `metrics` key can be found in the JSON.
    """
    p = Path(results_file)
    if not p.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    with open(p, "r") as f:
        data = json.load(f)

    def _find_metrics(obj: Any) -> Optional[Any]:
        if isinstance(obj, dict):
            if "metrics" in obj:
                return obj["metrics"]
            for v in obj.values():
                res = _find_metrics(v)
                if res is not None:
                    return res
        elif isinstance(obj, list):
            for item in obj:
                res = _find_metrics(item)
                if res is not None:
                    return res
        return None
    
    metrics = _find_metrics(data)
    if metrics is None:
        raise ValueError(f"No 'metrics' key found in {results_file} and no predictions present")

    # Try to find detailed predictions 
    def _find_key(obj: Any, key: str) -> Optional[Any]:
        if isinstance(obj, dict):
            if key in obj:
                return obj[key]
            for v in obj.values():
                res = _find_key(v, key)
                if res is not None:
                    return res
        elif isinstance(obj, list):
            for item in obj:
                res = _find_key(item, key)
                if res is not None:
                    return res
        return None

    predictions = _find_key(data, "predictions")
    if predictions is not None:
        requests: list[GuardrailRequest] = []
        responses: list[GuardrailResponse] = []

        for p in predictions:
            # Reconstruct GuardrailRequest
            req_text = p.get("request_text")
            req_meta = p.get("request_metadata", {})
            try:
                requests.append(GuardrailRequest(text=req_text, metadata=req_meta))
            except Exception:
                # Fallback to a minimal dict-like request
                requests.append(GuardrailRequest(text=str(req_text), metadata=dict(req_meta)))

            # Reconstruct GuardrailResponse from the dumped prediction
            pred_obj = p.get("prediction")
            if pred_obj is None:
                # older files may use different keys
                pred_obj = p.get("response") or p.get("result")

            try:
                # GuardrailResponse expects proper types; parse gracefully
                if isinstance(pred_obj, dict):
                    responses.append(GuardrailResponse.parse_obj(pred_obj))
                else:
                    # If prediction is a primitive, wrap it into raw_response
                    responses.append(GuardrailResponse(
                        is_safe=False,
                        score=0.0,
                        latency=0.0,
                        model_name="unknown",
                        raw_response=pred_obj,
                    ))
            except Exception:
                # Best-effort fallback: create a response with raw content
                try:
                    responses.append(GuardrailResponse.parse_obj(pred_obj))
                except Exception:
                    responses.append(GuardrailResponse(
                        is_safe=False,
                        score=0.0,
                        latency=0.0,
                        model_name="unknown",
                        raw_response=pred_obj,
                    ))

        return {
            "metrics": metrics,
            "requests": requests, 
            "responses": responses,
        }    

    return {"metrics": metrics,}
