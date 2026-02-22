# Guardrail Benchmark Python Lib

A modular framework for evaluating and benchmarking LLM Guardrail models against standardized datasets. This library is designed to be **provider-agnostic**, **asynchronous**, and **extensible**.

## Contents
1. [Purpose](#purpose)
2. [Project Structure](#project-structure)
4. [Key Features](#key-features)
3. [Quick Start](#quick-start)


## 1. Purpose 

[Back to Contents](#contents)

The goal of this project is to provide a unified interface to measure the efficacy of different safety layers. By standardizing the input/output schemas, you can compare a local LLM instance against a cloud API (OpenAI/Gemini) using the same metrics, latency tracking, and visualization tools.

## 2. Project Structure

[Back to Contents](#contents)

```text
guardrail_benchmark/
├── core/
│   ├── base_model.py      # Abstract Base Class (ABC) for all Guardrail providers
│   ├── base_dataset.py    # ABC for dataset loading and standardization
│   ├── engine.py          # Async execution logic (concurrency & semaphores)
│   ├── evaluator.py       # Orchestration logic for running benchmarks
│   └── schema.py          # Pydantic models for standardizing I/O (GuardrailResponse)
├── models/
│   ├── openai_model.py    # Wrapper for OpenAI Moderation API
│   ├── gemini_model.py    # Wrapper for Google Gemini Safety API
│   ├── llama_guard_model.py     # Wrapper for HuggingFace LlamaGuard API
│   └── caramllo_model.py  # Custom/Experimental model implementation
├── datasets/
│   ├── academic_dataset.py   # Loader for standard academic CSV/JSON datasets
│   └── toxic_chat_dataset.py # Loader for the ToxicChat HuggingFace dataset
├── utils/
│   ├── logging_config.py  # Structured logging for experiment audit trails
│   ├── metrics.py         # Accuracy, Precision, Recall, F1, and Latency math
│   ├── plotting.py        # Confusion matrices and performance bar graphs
│   └── save_output.py     # Results, experiment description and images saving
├── experiments/
│   └── benchmark_v1.py    # Entry point for running specific benchmark iterations
├── tests/                 # Unit tests for core logic and dummy models
├── .env.example           # Template for API keys
└── pyproject.toml         # Dependency management with optional "extras"

```

## 3. Key Features

[Back to Contents](#contents)

* **Standardized Schema:** Every model returns a `GuardrailResponse` Pydantic object, ensuring that metrics calculation is always consistent regardless of the model provider.
* **Async Performance:** Built with `asyncio`, allowing for high-throughput testing by batching API requests without blocking.
* **Rate Limit Management:** Integrated `Semaphore` support in the `engine.py` to prevent hitting provider rate limits during large-scale benchmarks.
* **Lazy Dependencies:** Install only what you need. Use `pip install .[openai]` or `.[local]` to keep the environment lean.

## 4. Quick Start

[Back to Contents](#contents)

### 1. Prerequisites

Ensure you have [mise]() installed. This project uses it to manage the Python runtime and the `uv` binary.

```bash
# Trust the local .mise.toml and install Python + uv
mise trust
mise install

```

### 2. Setup Environment

Copy the example environment file and add your API tokens:

```bash
cp .env.example .env

```

### 3. Install Dependencies

Use `uv` to create a virtual environment and install the library with specific provider support (e.g., `openai` and `google`):

```bash
# Sync dependencies using uv
uv sync --extra openai --extra google

```

### 4. Run a Benchmark

Execute your experiment scripts within the managed environment:

```bash
# Run using the uv-managed virtualenv
uv run python experiments/benchmark_v1.py

```

### 5. TODO List
- `tests/integration/test_gemini_wrapper.py`: TODO: update mock response to dict
- `models/caramllo_model.py`: 
    - TODO: update to dynamic category classification 
    - TODO: integration and smoke tests
- `models/llama_guard_model.py`: 
    - TODO: fix implementation 
    - TODO: update to dynamic category classification 
    - TODO: integration and smoke tests
- `utils/plotting.py`
    - TODO: add timestamps
    - TODO: add multiclass confusion matrix
- `experiments/openai_ptbr_benchmark.py`: 
    - TODO: create user-friendly interface for the benchmark
    - TODO: integration and smoke tests
- `experiments/gemini_ptbr_benchmark.py`: 
    - TODO: fix implementation 
    - TODO: create user-friendly interface for the benchmark
    - TODO: integration and smoke tests
- `pyproject.toml`: TODO: Add optional dependencies for the different LLMs and datasets
- `models/openai_moderation_model.py`: 
    - TODO: fix implementation 
    - TODO: update to dynamic category classification 
    - TODO: integration and smoke tests
- `models/gemini_moderation_model.py`: 
    - TODO: fix implementation 
    - TODO: update to dynamic category classification 
    - TODO: integration and smoke tests
- `datasets/toxic_chat_dataset.py`: 
    - TODO: Could add HuggingFace loading 
    - TODO: integration and smoke tests

