# e-AutoMFIS

**Ensemble Automatic Takagi-Sugeno-Kang Fuzzy Inference System** for interpretable time series forecasting.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

> The NumPy implementation of AutoMFIS used for the article *"Automatic generation of fuzzy
> inference systems for multivariate time series forecasting"* (Fuzzy Sets and Systems, 470,
> 108657, 2023) is preserved at tag [`article-v1`](https://github.com/mdrs-thiago/eautomfis/tree/article-v1).

## Overview

e-AutoMFIS is a machine learning model that combines the interpretability of fuzzy logic with the accuracy of ensemble methods. Unlike black-box models, e-AutoMFIS produces human-readable IF-THEN rules while achieving competitive prediction accuracy.

### Key Features

- **Interpretable Rules**: Generates linguistic rules like "IF X1 is High AND X2 is Low THEN y = 0.5"
- **TSK or Mamdani**: `system_type="tsk"` (default) or `"mamdani"`
- **TSK Consequents**: Polynomial consequents of any order (TSK-0 constant, TSK-1 linear, TSK-2+ with cross terms)
- **Mamdani Consequents**: Linguistic output terms ("THEN y is High"), chosen per rule by mean compatibility (CM) or fuzzy confidence degree (GCF)
- **Built on torch-fis**: Each ensemble member is a [torch-fis](https://github.com/mdrs-thiago/fuzzy_torch) `FuzzyModule`, so it is a differentiable `nn.Module`
- **Automatic Rule Learning**: Apriori-style mining with min-support pruning
- **Ensemble Learning**: Multiple members with feature/temporal subsampling
- **GPU Compatible**: PyTorch-based for GPU acceleration
- **Explainability**: Rule attribution for each prediction

## Installation

```bash
# Clone with the torch-fis submodule
git clone --recurse-submodules git@github.com:mdrs-thiago/eautomfis.git
cd eautomfis

# Install dependencies (torch-fis is the fuzzy_torch submodule)
pip install -r requirements.txt
pip install scikit-learn  # For benchmarking only
```

### How e-AutoMFIS uses torch-fis

| e-AutoMFIS | torch-fis |
|------------|-----------|
| `FuzzyPartition` | wraps a `FuzzyVariable` of `TriangularMF`/`TrapezoidalMF` terms |
| TSK consequents | `TSKConsequent`, fitted in closed form by weighted ridge regression on its polynomial basis (inputs rescaled to [-1, 1]) |
| Mamdani consequents | a term of the target's own `FuzzyVariable` (partitioned like the inputs), chosen by CM or GCF |
| Member inference | one `FuzzyModule` per horizon, `Product` t-norm, torch-fis rule weights (rescaled so the strongest rule is 1); Mamdani uses min implication, max aggregation and centroid defuzzification |

```python
config = EAutoMFISConfig(system_type="mamdani", mamdani_association="cm")
```

Rule mining (Apriori), filtering and the ensemble stay in e-AutoMFIS and work on the
membership tensor directly.

## Quick Start

```python
from eautomfis import EAutoMFIS, EAutoMFISConfig

# 1. Configure the model
config = EAutoMFISConfig(
    num_members=5,          # Ensemble size
    num_terms=5,            # Fuzzy terms per variable (e.g., Very Low, Low, Medium, High, Very High)
    tsk_order=1,            # 0=constant, 1=linear, 2+=polynomial consequents
    min_support=0.05,       # Minimum rule support
    max_antecedents=3,      # Maximum rule complexity
    random_seed=42,         # Reproducibility
)

# 2. Create and train the model
model = EAutoMFIS(config, n_jobs=4, verbose=True)
model.fit(X_train, y_train)

# 3. Make predictions
predictions = model.predict(X_test)

# 4. Evaluate performance
metrics = model.score(X_test, y_test)
print(metrics)
# ForecastMetrics:
#   MAE:  0.0951
#   RMSE: 0.1234
#   Rules: 150
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_members` | 10 | Number of ensemble members |
| `num_terms` | 5 | Fuzzy terms per variable |
| `partition_method` | "percentile" | How to create fuzzy partitions: "uniform", "percentile", "clustering" |
| `system_type` | "tsk" | Inference system: "tsk" or "mamdani" |
| `tsk_order` | 0 | TSK consequent order (0=constant, 1=linear, 2+=polynomial) |
| `mamdani_association` | "cm" | Mamdani output-term selection: "cm" or "gcf" |
| `min_support` | 0.05 | Minimum fuzzy support for rules |
| `max_antecedents` | 4 | Maximum antecedents per rule |
| `top_k_rules` | 50 | Maximum rules per member after filtering |
| `similarity_threshold` | 0.8 | Threshold for redundancy detection |
| `random_seed` | 42 | For reproducibility |

## Example: Explaining Predictions

```python
# Get explanation for a single sample
explanation = model.explain(X_test[0], top_k=5)

print(f"Prediction: {explanation['prediction']:.4f}")
print(f"\nTop contributing rules:")
for rule in explanation['member_contributions'][0]['top_rules']:
    print(f"  {rule['premise']}")
    print(f"    Firing strength: {rule['firing_strength']:.4f}")
    print(f"    Consequent: {rule['consequent']:.4f}")
```

Output:
```
Prediction: 0.4521

Top contributing rules:
  Premise((F0, T2) AND (F1, T3), supp=0.456)
    Firing strength: 0.7823
    Consequent: 0.4812
  Premise((F2, T1), supp=0.632)
    Firing strength: 0.6541
    Consequent: 0.3256
```

## Benchmark Results

Comparison on synthetic datasets (500 samples, 80/20 train/test split):

### Linear Data
| Model | MAE | RMSE | Time |
|-------|-----|------|------|
| Ridge Regression | **0.0715** | 0.0922 | 0.01s |
| Linear Regression | 0.0717 | 0.0924 | 0.02s |
| e-AutoMFIS (TSK-1) | 0.1626 | 0.2039 | 8.51s |
| Random Forest | 0.1335 | 0.1640 | 0.26s |

### Nonlinear Data (interactions, polynomials)
| Model | MAE | RMSE | Time |
|-------|-----|------|------|
| Random Forest | **0.3690** | 0.5235 | 0.16s |
| e-AutoMFIS (TSK-1) | 0.5229 | 0.7421 | 7.62s |
| Linear Regression | 0.5324 | 0.7436 | 0.00s |

### AR(5) Time Series
| Model | MAE | RMSE | Time |
|-------|-----|------|------|
| Linear Regression | **0.0871** | 0.1071 | 0.00s |
| e-AutoMFIS (TSK-1) | 0.0879 | 0.1078 | 8.04s |
| Ridge Regression | 0.0892 | 0.1088 | 0.00s |

**Key Insight**: e-AutoMFIS with TSK-1 achieves accuracy comparable to linear models on linear/AR data while providing **interpretable rules** and **explainable predictions**.

## Project Structure

```
fuzzy_torch/             # torch-fis (git submodule)
eautomfis/
├── __init__.py          # Package exports
├── config.py            # EAutoMFISConfig dataclass
├── utils.py             # Seeds, Timer, metrics, Logger
├── validation.py        # Walk-forward validation, temporal splits
├── partition.py         # Strong fuzzy partitions (3 methods)
├── mining.py            # Apriori-style premise mining
├── association.py       # TSK consequent fitting, Mamdani term association (CM/GCF)
├── filtering.py         # Similarity detection, top-K selection
├── inference.py         # torch-fis TSK / Mamdani systems per member
├── ensemble.py          # Parallel member training
├── forecaster.py        # Main EAutoMFIS class
└── wang_mendel.py       # Wang-Mendel rule extraction
test_eautomfis.py        # Test suite
benchmark_eautomfis.py   # Comparison against linear models and random forests
```

## API Reference

### EAutoMFIS

```python
class EAutoMFIS:
    def __init__(config: EAutoMFISConfig = None, n_jobs: int = 1, 
                 device: str = "auto", verbose: bool = True)
    
    def fit(X: np.ndarray, y: np.ndarray, 
            X_val: np.ndarray = None, y_val: np.ndarray = None) -> EAutoMFIS
    
    def predict(X: np.ndarray, horizon: int = 1, 
                return_std: bool = False) -> np.ndarray
    
    def score(X: np.ndarray, y: np.ndarray) -> ForecastMetrics
    
    def explain(x: np.ndarray, top_k: int = 5) -> dict
    
    def get_rules() -> List[str]
    
    def summary() -> str
```

### EAutoMFISConfig

```python
@dataclass
class EAutoMFISConfig:
    # Data
    max_lag: int = 5
    forecast_horizon: int = 1
    
    # Partitioning
    num_terms: int = 5
    partition_method: str = "percentile"  # uniform, percentile, clustering
    
    # Mining
    min_support: float = 0.05
    max_antecedents: int = 4
    
    # Consequent
    system_type: str = "tsk"  # tsk, mamdani
    tsk_order: int = 0  # 0=constant, 1=linear, 2+=polynomial
    tsk_regularization: float = 1e-3
    mamdani_association: str = "cm"  # cm, gcf (Mamdani only)
    
    # Ensemble
    num_members: int = 10
    feature_subset_size: int = 5
    
    # Reproducibility
    random_seed: int = 42
```

## Running Tests

```bash
# Run all tests
python test_eautomfis.py

# Run benchmark comparison
python benchmark_eautomfis.py
```

## How It Works

### 1. Fuzzy Partitioning
Each input variable is partitioned into fuzzy sets (e.g., "Low", "Medium", "High") using:
- **Uniform**: Evenly spaced terms
- **Percentile**: Terms at data percentiles (default)
- **Clustering**: K-means based terms

### 2. Rule Mining
Premises are discovered using Apriori-style level-wise search:
- Start with single antecedents (e.g., "X1 is High")
- Combine to form complex premises (e.g., "X1 is High AND X2 is Low")
- Prune by min-support (anti-monotone property)

### 3. TSK Consequent Fitting
For each premise, fit a TSK consequent:
- **TSK-0**: `y = constant` (weighted mean)
- **TSK-1**: `y = c0 + c1*x1 + c2*x2 + ...` (weighted ridge regression)

### 4. Rule Filtering
Remove redundant/low-quality rules:
- Jaccard similarity filtering
- Evidence-based scoring
- Top-K selection

### 5. Ensemble Aggregation
Multiple members with:
- Feature subsampling (diversity)
- Temporal subsampling
- OOF error-based weighting

### 6. Inference
Prediction via weighted average:
```
y_hat = Σ_r (w_r × f_r(x)) / Σ_r w_r
```

## Citation

If you use e-AutoMFIS in your research, please cite:

```bibtex
@article{carvalho2023automatic,
  title={Automatic generation of fuzzy inference systems for multivariate time series forecasting},
  author={Carvalho, Thiago and Vellasco, Marley and Amaral, Jos{\'e} Franco},
  journal={Fuzzy Sets and Systems},
  volume={470},
  pages={108657},
  year={2023},
  publisher={Elsevier}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
