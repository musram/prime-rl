# JAX Integration Strategy for PRIME-RL

## Overview

This document outlines the strategy for adding JAX support to PRIME-RL as a separate backend alongside the existing PyTorch implementation. This dual-framework approach leverages the strengths of both ecosystems while maintaining clean separation and user choice.

## Motivation

### Why JAX for RL Training?

1. **Performance Advantages**
   - 20-40% speedup for large batch training through XLA compilation
   - 15-30% memory reduction via kernel fusion
   - Superior vectorization with `vmap` for parallel trajectory processing
   - Whole-program optimization vs operation-by-operation execution

2. **Functional Programming Benefits**
   - Pure functions enable aggressive compiler optimizations
   - Deterministic execution and easier debugging
   - Natural parallelization patterns
   - Immutable state management

3. **Advanced Parallelization**
   - Built-in data parallelism with `pmap`
   - Model parallelism with `shard_map` 
   - Pipeline parallelism support
   - Efficient collective operations

4. **Research Flexibility**
   - Faster algorithm prototyping and experimentation
   - Easier implementation of mathematical operations
   - Better support for offline RL batch processing

### Framework Specialization Strategy

| Use Case | Recommended Framework | Rationale |
|----------|----------------------|-----------|
| Online RL (PPO, GRPO) | PyTorch | Better inference ecosystem (vLLM), established patterns |
| Offline RL (DPO, CQL) | JAX | Superior batch processing, mathematical operations |
| Preference Learning | JAX | Complex loss functions, gradient computations |
| SFT Training | PyTorch | Mature ecosystem, proven workflows |
| Inference Serving | PyTorch | vLLM integration, production stability |

## Architecture Design

### Proposed Directory Structure

```
prime-rl/
├── src/prime_rl/
│   ├── backends/
│   │   ├── pytorch/          # Existing PyTorch implementation
│   │   │   ├── trainer/
│   │   │   ├── inference/    # vLLM backend
│   │   │   ├── orchestrator/
│   │   │   └── algorithms/   # PyTorch-specific algorithms
│   │   │
│   │   ├── jax/             # New JAX implementation  
│   │   │   ├── trainer/     # JAX/Flax trainer
│   │   │   ├── inference/   # JAX inference or bridge
│   │   │   ├── orchestrator/
│   │   │   └── algorithms/  # JAX-specific algorithms
│   │   │
│   │   └── __init__.py
│   │
│   ├── core/               # Shared abstractions
│   │   ├── config/         # Framework-agnostic configs
│   │   ├── data/           # Common data structures
│   │   ├── algorithms/     # Algorithm interfaces
│   │   ├── evaluation/     # Shared eval logic
│   │   └── utils/          # Common utilities
│   │
│   └── bridges/            # Framework interoperability
│       ├── torch_to_jax.py
│       ├── jax_to_torch.py
│       └── data_conversion.py
```

### Core Abstractions

#### Algorithm Interface
```python
from abc import ABC, abstractmethod
from typing import Any, Dict

class RLAlgorithm(ABC):
    """Base class for all RL algorithms across frameworks."""
    
    @abstractmethod
    def compute_loss(self, rollouts: Any, advantages: Any) -> Dict[str, Any]:
        """Compute loss given rollouts and advantages."""
        pass
    
    @abstractmethod
    def train_step(self, params: Any, batch: Any) -> tuple[Any, Dict[str, Any]]:
        """Single training step."""
        pass

class OfflineRLAlgorithm(RLAlgorithm):
    """Base class for offline RL algorithms."""
    
    @abstractmethod
    def process_dataset(self, dataset: Any) -> Any:
        """Process offline dataset for training."""
        pass
```

#### Universal Data Structures
```python
@dataclass
class UniversalRollout:
    """Framework-agnostic rollout representation."""
    prompts: List[str]
    completions: List[str]
    rewards: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_pytorch(self) -> "TorchRollout":
        """Convert to PyTorch format."""
        pass
    
    def to_jax(self) -> "JAXRollout":
        """Convert to JAX format."""
        pass

@dataclass
class TrainingMetrics:
    """Common metrics across frameworks."""
    loss: float
    grad_norm: float
    learning_rate: float
    step: int
    framework_specific: Dict[str, Any] = field(default_factory=dict)
```

#### Configuration System
```python
class BackendConfig(BaseConfig):
    """Configuration for framework backend."""
    framework: Literal["pytorch", "jax"] = "pytorch"
    algorithm: str
    
    # Framework-specific settings
    pytorch_settings: Optional[PyTorchSettings] = None
    jax_settings: Optional[JAXSettings] = None

class JAXSettings(BaseConfig):
    """JAX-specific configuration."""
    enable_x64: bool = False
    jit_compile: bool = True
    device_count: Optional[int] = None
    parallel_strategy: Literal["data", "model", "pipeline"] = "data"

class PyTorchSettings(BaseConfig):
    """PyTorch-specific configuration."""
    compile_mode: Optional[str] = None
    fsdp_strategy: str = "full_shard"
    memory_efficient_attention: bool = True
```

## Implementation Phases

### Phase 1: Core Infrastructure (Weeks 1-2)

#### Objectives
- Establish shared abstractions and interfaces
- Create framework-agnostic data structures
- Set up configuration system for dual backends

#### Deliverables
1. **Core Interfaces** (`src/prime_rl/core/algorithms/`)
   ```python
   # base.py - Algorithm base classes
   # interfaces.py - Common interfaces
   # registry.py - Algorithm registry system
   ```

2. **Data Structures** (`src/prime_rl/core/data/`)
   ```python
   # rollouts.py - Universal rollout formats
   # metrics.py - Common metrics
   # datasets.py - Dataset abstractions
   ```

3. **Configuration** (`src/prime_rl/core/config/`)
   ```python
   # backend.py - Backend selection
   # jax_config.py - JAX-specific settings
   # pytorch_config.py - PyTorch-specific settings
   ```

### Phase 2: JAX Trainer Implementation (Weeks 3-4)

#### Objectives
- Implement core JAX trainer infrastructure
- Create JAX versions of key algorithms
- Establish model loading and saving patterns

#### Deliverables
1. **JAX Trainer** (`src/prime_rl/backends/jax/trainer/`)
   ```python
   # trainer.py - Main JAX trainer class
   # model.py - Flax model implementations
   # optim.py - JAX optimizers (Optax integration)
   # parallel.py - JAX parallelization utilities
   ```

2. **JAX Algorithms** (`src/prime_rl/backends/jax/algorithms/`)
   ```python
   # dpo.py - Direct Preference Optimization
   # offline_ppo.py - Offline PPO variant
   # rejection_sampling.py - Rejection sampling
   # conservative_q.py - Conservative Q-Learning
   ```

3. **Model Integration**
   ```python
   # Support for HuggingFace -> Flax conversion
   # Checkpoint compatibility between frameworks
   # Weight initialization strategies
   ```

### Phase 3: Offline RL Algorithms (Weeks 5-6)

#### Objectives
- Implement offline RL algorithms specifically in JAX
- Create dataset processing pipelines
- Add preference learning methods

#### Deliverables
1. **Offline Algorithms**
   - Direct Preference Optimization (DPO)
   - Conservative Q-Learning (CQL) 
   - Advantage Weighted Regression (AWR)
   - Behavior Cloning variants

2. **Dataset Processing**
   ```python
   # Large-scale dataset loading
   # Batch processing with vmap
   # Data augmentation and filtering
   # Quality scoring and ranking
   ```

3. **Preference Learning**
   ```python
   # Pairwise preference processing
   # Reward model integration
   # Constitutional AI methods
   ```

### Phase 4: Bridge Components (Weeks 7-8)

#### Objectives
- Enable interoperability between PyTorch and JAX
- Create conversion utilities
- Implement hybrid workflows

#### Deliverables
1. **Framework Bridges** (`src/prime_rl/bridges/`)
   ```python
   # conversion.py - Tensor/array conversions
   # checkpoint.py - Cross-framework checkpointing
   # inference.py - Bridge JAX models to PyTorch inference
   ```

2. **Hybrid Workflows**
   ```python
   # JAX training -> PyTorch inference
   # Cross-framework evaluation
   # Model format conversion
   ```

## Algorithm Implementation Priority

### Tier 1: Core Offline RL (Immediate)
1. **Direct Preference Optimization (DPO)**
   - Most requested for mathematical reasoning
   - Clear performance benefits in JAX
   - Foundation for other preference methods

2. **Rejection Sampling**
   - Simple to implement and validate
   - Significant for solution quality improvement
   - Good JAX vectorization example

3. **Supervised Fine-tuning (JAX variant)**
   - Baseline for offline methods
   - Easy migration from PyTorch version
   - Testing framework integration

### Tier 2: Advanced Offline Methods (Next)
1. **Conservative Q-Learning (CQL)**
   - Important for offline RL research
   - Demonstrates JAX mathematical capabilities
   - Novel for language model training

2. **Constitutional AI**
   - Multi-stage training pipeline
   - Complex preference optimization
   - Research differentiator

3. **Self-Training Pipelines**
   - Iterative improvement methods
   - Data generation and filtering
   - DeepSeek-Math-V2 style approaches

### Tier 3: Hybrid Methods (Future)
1. **Online-to-Offline Transitions**
   - Collect data with PyTorch, train with JAX
   - Best of both worlds approach
   - Production-research bridge

2. **Multi-Algorithm Training**
   - Simultaneous SFT + DPO
   - Algorithm scheduling and switching
   - Advanced research capabilities

## Performance Expectations

### Training Speed Improvements
| Algorithm Type | Expected Speedup | Memory Savings | Notes |
|---------------|------------------|----------------|-------|
| DPO Training | 30-50% | 20-30% | Large batch benefits |
| Rejection Sampling | 40-60% | 15-25% | Vectorization wins |
| Dataset Processing | 50-80% | 10-20% | vmap optimization |
| Gradient Computation | 25-40% | 20-35% | XLA fusion |

### Scaling Characteristics
- **Single GPU**: 20-30% improvement
- **Multi-GPU**: 40-50% improvement (better collective ops)
- **Large Batches**: 50-70% improvement (XLA shines)
- **Mathematical Operations**: 60-80% improvement (native JAX)

## Configuration Examples

### JAX DPO Training
```toml
[trainer]
backend = "jax"
algorithm = "dpo"

[trainer.jax_settings]
enable_x64 = true
jit_compile = true
parallel_strategy = "data"

[trainer.model]
name = "meta-llama/Llama-3.1-7B"
load_in_8bit = false

[trainer.algorithm.dpo]
beta = 0.1
loss_type = "sigmoid"
reference_free = false

[trainer.data]
dataset_path = "offline_preferences.jsonl"
batch_size = 32
max_length = 2048
```

### PyTorch Online RL (Existing)
```toml
[trainer]
backend = "pytorch"
algorithm = "grpo"

[trainer.pytorch_settings]
compile_mode = "max-autotune"
fsdp_strategy = "full_shard"

[trainer.model]
name = "meta-llama/Llama-3.1-7B"
use_lora = true

[orchestrator]
max_async_level = 2
batch_size = 16
```

### Hybrid Workflow
```toml
[trainer]
backend = "jax"
algorithm = "dpo"
inference_bridge = "pytorch"  # Use PyTorch for inference

[inference]
backend = "pytorch"  # Keep vLLM
server_type = "vllm"

[orchestrator]
cross_framework = true
conversion_strategy = "checkpoint"
```

## Testing Strategy

### Unit Tests
- Algorithm correctness across frameworks
- Data conversion accuracy
- Configuration validation
- Performance regression detection

### Integration Tests
- End-to-end training workflows
- Cross-framework checkpoint loading
- Hybrid training pipelines
- Multi-GPU scaling

### Performance Benchmarks
- Training speed comparisons
- Memory usage profiling
- Convergence rate analysis
- Scaling characteristics

### Validation Studies
- Mathematical reasoning benchmarks (GSM8K, MATH)
- Preference learning accuracy
- Solution quality metrics
- Generalization testing

## Migration Guide

### For Existing Users
1. **No Breaking Changes**: Existing PyTorch configs continue to work
2. **Opt-in JAX**: Add `backend = "jax"` to try new implementation
3. **Gradual Migration**: Test specific algorithms before full switch
4. **Fallback Support**: Automatic fallback to PyTorch if JAX unavailable

### For New Algorithms
1. **Framework Selection**: Choose based on algorithm characteristics
2. **Implementation Guide**: Templates and examples for each framework
3. **Performance Testing**: Benchmarking tools and guidelines
4. **Documentation**: Algorithm-specific setup instructions

## Future Roadmap

### Short-term (3 months)
- Core JAX infrastructure
- DPO and rejection sampling
- Basic offline RL capabilities
- Performance validation

### Medium-term (6 months)
- Full offline RL algorithm suite
- Advanced preference learning
- Multi-algorithm training
- Production deployment guides

### Long-term (12 months)
- Novel RL algorithms research
- Advanced mathematical reasoning methods
- Hybrid online-offline workflows
- Community ecosystem development

## Getting Started

### Prerequisites
```bash
# Install JAX dependencies
pip install jax[cuda12] flax optax
# or for CPU-only
pip install jax flax optax

# Verify installation
python -c "import jax; print(jax.devices())"
```

### Quick Start
```bash
# Clone and setup
git clone https://github.com/PrimeIntellect-ai/prime-rl.git
cd prime-rl
uv sync

# Try JAX DPO training
uv run trainer @ configs/jax/dpo/train.toml

# Compare with PyTorch baseline
uv run trainer @ configs/pytorch/dpo/train.toml
```

### Example Workflow
```python
# Select backend in config
config = TrainerConfig(
    backend="jax",
    algorithm="dpo",
    model=ModelConfig(name="meta-llama/Llama-3.1-7B"),
    data=DataConfig(path="preferences.jsonl")
)

# Train with JAX
trainer = create_trainer(config)
trainer.train()

# Convert for PyTorch inference if needed
convert_checkpoint(
    "outputs/jax_checkpoint",
    "outputs/pytorch_checkpoint", 
    target_framework="pytorch"
)
```

## Contributing

### Adding New JAX Algorithms
1. Implement algorithm interface
2. Add configuration schema  
3. Create unit and integration tests
4. Update documentation and examples
5. Submit PR with performance benchmarks

### Framework Bridge Development
1. Identify conversion requirements
2. Implement tensor/array conversion utilities
3. Add checkpoint compatibility
4. Test cross-framework workflows
5. Document usage patterns

---

This JAX integration strategy provides a clear path forward for extending PRIME-RL with high-performance offline RL capabilities while maintaining the strengths of the existing PyTorch implementation.