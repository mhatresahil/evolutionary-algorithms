# Evolutionary Algorithms for Neural Network Optimization

This repository implements various evolutionary algorithms to optimize neural network weights for a hotel booking prediction problem. The implemented algorithms include:

- Genetic Algorithm (GA) with masked weights and parallel processing
- Evolutionary Strategy (μ,λ) with adaptive mutation rates
- Evolutionary Strategy (μ+λ) with parent survival
- Multi-mutation rate Evolutionary Strategy

## Upcoming Algorithms

The following algorithms are planned to be added to the repository:
- Differential Evolution (DE)
- Particle Swarm Optimization (PSO)
- Firefly Algorithm (FA)
- More variants of existing algorithms

Stay tuned for updates!

## Project Structure

```
├── genetic_algo.py      # Genetic Algorithm implementation
├── evols.py            # Basic Evolutionary Strategies (μ,λ) and (μ+λ)
├── multi_mutation.py   # Enhanced ES with multiple mutation rates
├── environment.yaml    # Conda environment specification
├── requirements.txt    # Python package requirements
```

## Algorithms Overview

### Genetic Algorithm
- Uses masked weight initialization with 10% probability of keeping weights
- Implements uniform crossover with 90% probability
- Features targeted mutation of masked weights
- Utilizes parallel processing for fitness evaluation
- Includes elitism to preserve best solutions

### Evolutionary Strategies
Two variants are implemented:
1. (μ,λ) Strategy:
   - Parents are replaced by offspring each generation
   - Adaptive mutation step size based on success rate
   - Features generational selection pressure

2. (μ+λ) Strategy:
   - Parents compete with offspring for survival
   - Maintains higher selection pressure
   - Better at preserving good solutions

### Multi-mutation ES
- Layer-specific mutation rates
- Automatic adaptation of mutation parameters
- Enhanced exploration capabilities

## Setup and Installation

### Using Conda
```bash
# Create environment
conda env create -f environment.yaml

# Activate environment
conda activate evolve
```

### Using pip
```bash
# Create virtual environment
python -m venv venv

# Activate environment
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install requirements
pip install -r requirements.txt
```

## Usage

### Running Genetic Algorithm
```bash
python genetic_algo.py
```

### Running Basic Evolutionary Strategies
```bash
python evols.py
```

### Running Multi-mutation Evolution
```bash
python multi_mutation.py
```

## Model Architecture

All implementations use a simple neural network with:
- Input layer based on feature dimensions
- Hidden layer with 28 neurons and sigmoid activation
- Output layer with sigmoid activation for binary classification
- Binary cross-entropy loss and Adam optimizer

## Performance Visualization

Each algorithm implementation includes visualization capabilities:
- Fitness progression plots
- Confusion matrix for final model evaluation
- Performance metrics including accuracy scores

## Requirements

Key dependencies include:
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- scikit-learn

For complete dependencies, refer to `requirements.txt` or `environment.yaml`.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

