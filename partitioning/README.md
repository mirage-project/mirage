# Mirage Program Partitioning

This directory contains the program partitioning system for Mirage, which automatically partitions PyTorch models into subgraphs and executes them using a hybrid approach: Mirage-optimized kernels for supported operations and PyTorch fallback for others.

## Overview

The partitioning system converts PyTorch models to ONNX, partitions the computation graph into optimal subgraphs, and creates a hybrid execution model that combines:
- **Mirage-optimized kernels**: For supported operations that can benefit from kernel fusion and optimization
- **PyTorch fallback**: For unsupported or less frequently used operations

## Key Components

### Core Modules

- **`runtime.py`**: Main execution engine
  - `HybridModel`: Hybrid execution wrapper that routes operations to Mirage or PyTorch
  - `partition_graph_with_dp`: Dynamic programming-based partitioning using GNN-XGBoost cost model
  
- **`gnn_xgboost.py`**: GNN-XGBoost cost model for predicting kernel execution time
  - Combines Graph Neural Networks for feature extraction with XGBoost for cost prediction

- **`build_computation_graph.py`**: Converts PyTorch models to ONNX computation graphs

- **`graph_splitter.py`**: Processes and splits operator graphs into partitions

- **`generate_dag.py`**: Generates execution DAG and solves partition ordering

- **`testing_models.py`**: Test models (TestMLP, TestTransformer) for validation

- **`utils.py`**: Utility functions for graph conversion and manipulation

### Cost Model

The partitioning system uses a learned cost model located in `cost_model/models/`:
- `11_25_exec_time_gine_best_full_lr3e-03.pt`: GNN encoder weights
- `11_25_exec_time_xgb_best_xgb.json`: XGBoost model for cost prediction

## Installation

### Prerequisites

```bash
# Core dependencies
pip install torch torchvision onnx onnxruntime
pip install torch-geometric xgboost scikit-learn
```

### Environment Setup

The code automatically handles module imports from the parent directory. Ensure you have:
- CUDA-capable GPU (required for testing and benchmarking)
- Python 3.10+
- PyTorch with CUDA support

## Running Tests

### Basic Test Execution

Navigate to Mirage root directory and run:

```bash
# Test with MLP model using dynamic programming partitioning with GNN+XGBoost model
python -m partitioning.tests.test_partition --cost-model gnn-xgboost --test-model-name test-mlp

# Test with Transformer model
python -m partitioning.tests.test_partition --cost-model gnn-xgboost --test-model-name test-transformer
```

### Command-Line Arguments

```bash
python test_partition.py [OPTIONS]

Required:
  --cost-model {gnn-xgboost}              Cost model for partitioning ('gnn-xgboost' uses GNN-XGBoost)
  --test-model-name {test-mlp,test-transformer}
                                 Model to test

Optional:
  --max-nodes-per-partition N    Maximum nodes per partition (default: 4)
  --max-mirage-ops N            Maximum ops for Mirage optimization (default: 9)
  --dry-run                     Skip superoptimization (for testing)
```

### What the Tests Do

1. **Graph Partitioning**: Converts the model to ONNX and partitions it using the specified cost model
2. **Execution Test**: Runs both original PyTorch and hybrid models to verify correctness
3. **Numerical Validation**: Compares outputs to ensure they match within tolerance
4. **Benchmarking**: Measures average latency over 100 iterations with warmup
5. **Profiling**: Generates detailed CPU/CUDA profiling data

## Test Models

### TestMLP
- Simple 5-layer MLP with tanh and sigmoid activations
- Input: `[batch_size=32, 1024]`
- Uses manual weight parameters (not `nn.Linear`) for better ONNX export
- Good for testing basic operator fusion

### TestTransformer
- Single transformer block with multi-head attention and feedforward
- Input: `[batch_size=1, seq_len=1024]` (token IDs)
- Includes RMSNorm, attention mechanism, and MLP
- Tests complex graph partitioning with multiple operation types

## Configuration

### Supported Operations
The partitioning system supports most PyTorch operations. Unsupported operations automatically fall back to PyTorch:
- Unsupported: `Constant`, `Identity`, `Unsqueeze`, `Abs`, `Gemm`, `Expand`, `Gather`, `Reshape`, `Transpose`, `Cast`, `CastLike`, `Tanh`, `ReduceSum`

### Tuning Parameters

- **`max_nodes_per_partition`**: Controls partition granularity (smaller = more partitions)
- **`max_mirage_ops`**: Maximum operations in a single Mirage kernel (affects optimization time)
- **Batch sizes and sequence lengths**: Defined in test files, can be modified for different workloads

## Development

### Adding New Test Models

1. Define your model in `testing_models.py`:
```python
class YourModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Define layers
    
    def forward(self, x):
        # Forward pass
        return x
```

2. Add to `test_partition.py`:
```python
model_name_to_class = {
    "your-model": YourModel,
}

def _get_input_tensor(model_name):
    if model_name == "your-model":
        return torch.randn(batch, dims, device="cuda", dtype=torch.float16)
```

### Debugging Partitions

Enable debug output in test execution:
```python
hybrid_output = hybrid_model(test_input, debug=True)  # Prints execution trace
```
