"""
Fuzzy inference module for e-AutoMFIS.

Builds torch-fis ``FuzzyModule``s from mined rules, TSK or Mamdani:
- Product t-norm firing strengths, scaled by per-rule weights
- TSK: weighted average of consequent outputs
- Mamdani: min implication, max aggregation, centroid defuzzification
- Multi-horizon support (one system per horizon, sharing the input variables)
"""

from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn
import numpy as np

from fuzzy_torch import FuzzyModule, FuzzyVariable, Product, MAMDANI, TSK
from fuzzy_torch import FuzzyRule as FISRule

from .association import FuzzyRule
from .partition import FuzzyPartition


class FuzzyInferenceEngine(nn.Module):
    """
    Fuzzy inference engine for rule-based prediction.

    TSK (``output_partitions`` is None) computes:
        y_hat = Σ_r (w_r * f_r(x)) / Σ_r w_r

    Mamdani (``output_partitions`` given) clips each rule's output term at w_r,
    aggregates with max and returns the centroid.

    where:
        w_r = firing_strength_r * rule_weight_r
        f_r(x) = TSK consequent function

    Rule weights are rescaled so the strongest rule has weight 1. Mined weights
    sum to 1, so unscaled they would be ~1/R and a Mamdani rule's clip height
    would depend on the rule count; TSK outputs are unaffected by the rescale.
    Samples where no rule fires are predicted as 0.
    """

    def __init__(
        self,
        rules: List[FuzzyRule],
        partitions: List[FuzzyPartition],
        max_antecedents: int = 4,
        device: str = "cpu",
        output_partitions: Optional[List[FuzzyPartition]] = None,
    ):
        """
        Initialize inference engine.

        Args:
            rules: List of fitted FuzzyRule objects
            partitions: List of FuzzyPartition objects (one per feature)
            max_antecedents: Unused; kept for API compatibility
            device: Computation device
            output_partitions: Mamdani only: one partition of the target per
                horizon; rule consequents are then term indices into it
        """
        super().__init__()
        if not rules:
            raise ValueError("cannot build an inference engine from an empty rule base")

        self.rules = rules
        self.partitions = partitions
        self.output_partitions = output_partitions
        self.input_names = [p.name for p in partitions]
        self.mode = MAMDANI if output_partitions is not None else TSK

        # Check if multi-horizon
        self.is_multi_horizon = isinstance(rules[0].consequent, list)
        self.num_horizons = len(rules[0].consequent) if self.is_multi_horizon else 1

        variables = [p.variable for p in partitions]
        max_weight = max(r.weight for r in rules)
        weights = [r.weight / max_weight if max_weight > 0 else 1.0 for r in rules]

        systems = []
        for h in range(self.num_horizons):
            if self.mode == MAMDANI:
                output = output_partitions[h].variable
            else:
                # TSK consequents are already bound and crisp, so the output range is unused
                output = FuzzyVariable("y", 0.0, 1.0)

            fis_rules = []
            for rule, weight in zip(rules, weights):
                ants = sorted(rule.premise.antecedents, key=lambda a: a.feature_idx)
                antecedent = [
                    (variables[a.feature_idx], partitions[a.feature_idx].term_names[a.term_idx])
                    for a in ants
                ]
                consequent = rule.consequent[h] if self.is_multi_horizon else rule.consequent
                if self.mode == MAMDANI:
                    consequent = (output, output_partitions[h].term_names[consequent])
                fis_rules.append(FISRule(antecedent, consequent, t_norm=Product(), weight=weight))

            systems.append(FuzzyModule(variables, output, fis_rules, no_rule_value=0.0))

        self.systems = nn.ModuleList(systems)
        self.to(torch.device(device))

    @property
    def device(self) -> torch.device:
        """Device the engine lives on; follows ``.to(...)``."""
        return self.systems[0].rules[0]._weight.device

    def _as_inputs(self, X: Union[np.ndarray, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Split a [N, D] matrix into the {variable name: [N]} dict torch-fis expects."""
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        if X.dim() == 1:
            X = X.unsqueeze(0)
        X = X.to(self.device)
        return {name: X[:, i] for i, name in enumerate(self.input_names)}

    def compute_firing_strengths(self, X: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Compute (unweighted) firing strengths for all rules.

        Args:
            X: Input features [N, D]

        Returns:
            Firing strengths [N, R]
        """
        with torch.no_grad():
            return self.systems[0].get_rule_activations(self._as_inputs(X), weighted=False)

    def infer(
        self,
        X: Union[np.ndarray, torch.Tensor],
        horizon: int = 1,
    ) -> torch.Tensor:
        """
        Perform fuzzy inference.

        Args:
            X: Input features [N, D]
            horizon: Forecast horizon (for multi-horizon models)

        Returns:
            Predictions [N] or [N, H]
        """
        with torch.no_grad():
            inputs = self._as_inputs(X)

            if self.is_multi_horizon:
                H = min(horizon, self.num_horizons)
                return torch.stack([system(inputs) for system in self.systems[:H]], dim=1)

            return self.systems[0](inputs)

    def explain(
        self,
        x: Union[np.ndarray, torch.Tensor],
        top_k: int = 5,
    ) -> dict:
        """
        Explain prediction for a single sample.

        Args:
            x: Single input [D] or [1, D]
            top_k: Number of top rules to show

        Returns:
            Explanation dictionary
        """
        inputs = self._as_inputs(x)
        system = self.systems[0]

        with torch.no_grad():
            firing = system.get_rule_activations(inputs, weighted=False).squeeze(0)  # [R]
            X_row = torch.stack([inputs[n] for n in system.input_order], dim=1)

            # Sort by firing strength
            sorted_idx = firing.argsort(descending=True)

            explanation = {
                "prediction": self.infer(x).item(),
                "top_rules": [],
            }

            for i in range(min(top_k, len(self.rules))):
                idx = sorted_idx[i].item()
                rule = self.rules[idx]
                strength = firing[idx].item()

                if strength > 1e-6:
                    fis_rule = system.rules[idx]
                    explanation["top_rules"].append({
                        "rule_id": idx,
                        "premise": str(rule.premise),
                        "firing_strength": strength,
                        "weight": rule.weight,
                        "contribution": strength * rule.weight,
                        # Horizon-1 consequent: output term (Mamdani) or f(x) at this input (TSK)
                        "consequent": (fis_rule.consequent[1] if self.mode == MAMDANI
                                       else fis_rule.consequent_fn(X_row).item()),
                    })

        return explanation


# Backwards-compatible name from when only TSK was supported
TSKInferenceEngine = FuzzyInferenceEngine


def create_inference_engine(
    rules: List[FuzzyRule],
    partitions: List[FuzzyPartition],
    max_antecedents: int = 4,
    device: str = "cpu",
    output_partitions: Optional[List[FuzzyPartition]] = None,
) -> FuzzyInferenceEngine:
    """
    Factory function to create an inference engine.

    Args:
        rules: List of FuzzyRule objects
        partitions: List of FuzzyPartition objects
        max_antecedents: Max antecedents per rule
        device: Computation device
        output_partitions: Target partitions (one per horizon) for a Mamdani engine

    Returns:
        FuzzyInferenceEngine
    """
    return FuzzyInferenceEngine(rules, partitions, max_antecedents, device, output_partitions)


def predict_multistep_direct(
    engine: FuzzyInferenceEngine,
    X: Union[np.ndarray, torch.Tensor],
    horizon: int,
) -> torch.Tensor:
    """
    Multi-step direct prediction.

    Each horizon has its own consequent parameters.

    Args:
        engine: Fuzzy inference engine
        X: Input features [N, D]
        horizon: Forecast horizon

    Returns:
        Predictions [N, H]
    """
    return engine.infer(X, horizon=horizon)


def predict_multistep_recursive(
    engine: FuzzyInferenceEngine,
    X: Union[np.ndarray, torch.Tensor],
    horizon: int,
    target_col: int = 0,
) -> torch.Tensor:
    """
    Multi-step recursive prediction.

    Uses previous predictions as input for future steps.

    Args:
        engine: Fuzzy inference engine (single-horizon)
        X: Initial input features [N, D]
        horizon: Forecast horizon
        target_col: Column index for target variable

    Returns:
        Predictions [N, H]
    """
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).float()
    X = X.to(engine.device)

    N, D = X.shape
    predictions = torch.zeros(N, horizon, device=engine.device)

    current_X = X.clone()

    for h in range(horizon):
        # Predict one step
        y_h = engine.infer(current_X)
        predictions[:, h] = y_h

        # Update input for next step (shift and insert prediction)
        # This assumes a specific lag structure
        if h < horizon - 1:
            # Shift lagged values and insert new prediction
            # This is a simplified version - actual implementation depends on feature structure
            current_X = current_X.roll(-1, dims=1)
            current_X[:, target_col] = y_h

    return predictions
