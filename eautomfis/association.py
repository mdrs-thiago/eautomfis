"""
TSK consequent association module for e-AutoMFIS.

Implements Takagi-Sugeno-Kang (TSK) consequent computation on top of
torch-fis ``TSKConsequent`` (full multivariate polynomial of any order):
- TSK-0: Constant consequent (weighted mean)
- TSK-1+: Polynomial consequent (weighted least squares with ridge)

torch-fis trains consequents by gradient descent; here they are fitted in
closed form, one weighted ridge regression per rule, and the solution is
written into the consequent's ``coefficients`` parameter.

Also includes classical association metrics:
- Mean Compatibility (CM)
- Fuzzy Confidence Degree (GCF)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from fuzzy_torch import TSKConsequent


def fit_tsk_consequent(
    X: torch.Tensor,
    y: torch.Tensor,
    weights: torch.Tensor,
    order: int = 0,
    regularization: float = 1e-3,
    input_names: Optional[Sequence[str]] = None,
    input_ranges: Optional[Sequence[Tuple[float, float]]] = None,
) -> TSKConsequent:
    """
    Fit a TSK consequent using weighted least squares.

    The design matrix is the consequent's own monomial basis, with inputs
    rescaled from ``input_ranges`` onto [-1, 1], so any order is supported
    and the ridge penalty is independent of the feature scale.

    Args:
        X: Input features [N, D]
        y: Target values [N] or [N, 1]
        weights: Sample weights (firing strengths) [N]
        order: TSK order (polynomial degree)
        regularization: Ridge regularization for order > 0
        input_names: Name of each column of X (default X0..X{D-1})
        input_ranges: (min, max) of each column of X (default: from X)

    Returns:
        Fitted torch-fis TSKConsequent, bound to ``input_names``
    """
    y = y.reshape(-1)
    weights = weights.reshape(-1)

    if input_names is None:
        input_names = [f"X{i}" for i in range(X.shape[1])]
    if input_ranges is None:
        input_ranges = list(zip(X.min(dim=0).values.tolist(), X.max(dim=0).values.tolist()))

    consequent = TSKConsequent(order, input_names=input_names, input_ranges=input_ranges)

    # Filter to samples with non-zero weight; no active samples leaves zeros
    mask = weights > 1e-8
    if mask.sum() == 0:
        return consequent

    w_active = weights[mask]
    y_active = y[mask]
    Phi = consequent.basis(X[mask])  # [n, P], column 0 is the constant

    # Weighted normal equations with ridge: (Φ'WΦ + λI)β = Φ'Wy
    PhitW = Phi.T * w_active
    PhitWPhi = PhitW @ Phi
    PhitWy = PhitW @ y_active

    # Add ridge regularization (not to bias term)
    reg_matrix = regularization * torch.eye(PhitWPhi.shape[0], device=PhitWPhi.device)
    reg_matrix[0, 0] = 0

    try:
        coef = torch.linalg.solve(PhitWPhi + reg_matrix, PhitWy)
    except RuntimeError:
        # Singular matrix, fallback to least squares
        coef = torch.linalg.lstsq(PhitWPhi + reg_matrix, PhitWy.unsqueeze(-1)).solution.squeeze(-1)

    with torch.no_grad():
        consequent.coefficients.copy_(coef)

    return consequent


def fit_tsk_consequents_batch(
    X: torch.Tensor,
    y: torch.Tensor,
    firing_strengths: torch.Tensor,
    order: int = 0,
    regularization: float = 1e-3,
    input_names: Optional[Sequence[str]] = None,
    input_ranges: Optional[Sequence[Tuple[float, float]]] = None,
) -> List[Union[TSKConsequent, List[TSKConsequent]]]:
    """
    Fit TSK consequents for multiple rules.

    Args:
        X: Input features [N, D]
        y: Target values [N] or [N, H] for multi-horizon
        firing_strengths: Firing strengths [N, R] (R rules)
        order: TSK order
        regularization: Ridge regularization
        input_names: Name of each column of X
        input_ranges: (min, max) of each column of X

    Returns:
        List of R TSKConsequent objects (a list of H per rule if multi-horizon)
    """
    N, R = firing_strengths.shape
    y = y.squeeze() if y.dim() > 1 and y.shape[1] == 1 else y

    consequents = []
    for r in range(R):
        weights = firing_strengths[:, r]

        # Handle multi-horizon
        if y.dim() == 2:
            # Fit separate consequent for each horizon
            horizon_conseqs = []
            for h in range(y.shape[1]):
                conseq = fit_tsk_consequent(X, y[:, h], weights, order, regularization,
                                            input_names, input_ranges)
                horizon_conseqs.append(conseq)
            consequents.append(horizon_conseqs)
        else:
            conseq = fit_tsk_consequent(X, y, weights, order, regularization,
                                        input_names, input_ranges)
            consequents.append(conseq)

    return consequents


# =============================================================================
# Classical Association Metrics (for comparison/interpretability)
# =============================================================================

def mean_compatibility(
    premise_activations: torch.Tensor,
    consequent_memberships: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Mean Compatibility (CM) between premises and consequent terms.

    CM = (1/N) * Σ μ_premise(x_i) * μ_consequent(y_i)

    Args:
        premise_activations: [N] or [N, R] premise firing strengths
        consequent_memberships: [N, T] memberships for each consequent term

    Returns:
        CM values [T] or [R, T]
    """
    if premise_activations.dim() == 1:
        premise_activations = premise_activations.unsqueeze(1)

    # CM for each premise-consequent pair: [R, T]
    # (N, R).T @ (N, T) / N = (R, N) @ (N, T) / N = (R, T)
    N = premise_activations.shape[0]
    cm = (premise_activations.T @ consequent_memberships) / N

    return cm


def fuzzy_confidence_degree(
    premise_activations: torch.Tensor,
    consequent_memberships: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Fuzzy Confidence Degree (GCF) using cosine similarity.

    GCF = (μ_P · μ_C) / (||μ_P|| × ||μ_C||)

    Args:
        premise_activations: [N] or [N, R]
        consequent_memberships: [N, T]

    Returns:
        GCF values [T] or [R, T]
    """
    if premise_activations.dim() == 1:
        premise_activations = premise_activations.unsqueeze(1)

    # Normalize
    p_norm = premise_activations / (premise_activations.norm(dim=0, keepdim=True) + 1e-8)
    c_norm = consequent_memberships / (consequent_memberships.norm(dim=0, keepdim=True) + 1e-8)

    # Cosine similarity: [R, T]
    gcf = p_norm.T @ c_norm

    return gcf


def select_best_consequent_term(
    premise_activations: torch.Tensor,
    consequent_memberships: torch.Tensor,
    method: str = "cm",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Select the best consequent term for each premise.

    Args:
        premise_activations: [N, R] premise firing strengths
        consequent_memberships: [N, T] memberships for each term
        method: 'cm' or 'gcf'

    Returns:
        (best_term_indices, scores) - both [R]
    """
    if method == "cm":
        scores = mean_compatibility(premise_activations, consequent_memberships)
    else:
        scores = fuzzy_confidence_degree(premise_activations, consequent_memberships)

    best_idx = scores.argmax(dim=-1)
    best_scores = scores.gather(-1, best_idx.unsqueeze(-1)).squeeze(-1)

    return best_idx, best_scores



# =============================================================================
# Rule Representation
# =============================================================================

@dataclass
class FuzzyRule:
    """
    Complete fuzzy rule with premise and consequent.

    For TSK systems:
        IF (x1 IS A1) AND (x2 IS A2) THEN y = f(x)

    This is the mining-side record (premise as feature/term indices plus
    scores); ``inference.TSKInferenceEngine`` turns a list of them into
    torch-fis ``FuzzyRule``s inside a ``TSKSystem``.

    Attributes:
        premise: Premise object (antecedents)
        consequent: torch-fis TSKConsequent, one per horizon, or term index (for Mamdani)
        weight: Rule weight/importance
        quality_score: Quality metric (support, coverage, etc.)
    """
    premise: "Premise"  # Forward reference
    consequent: Union[TSKConsequent, int, List[TSKConsequent]]
    weight: float = 1.0
    quality_score: float = 0.0

    def __repr__(self) -> str:
        if isinstance(self.consequent, TSKConsequent):
            if self.consequent.order == 0:
                conseq_str = f"y = {self.consequent.coefficients.item():.4f}"
            else:
                conseq_str = f"y = f(x) [order={self.consequent.order}]"
        elif isinstance(self.consequent, list):
            conseq_str = f"[{len(self.consequent)} horizons]"
        else:
            conseq_str = f"T{self.consequent}"

        return f"Rule({self.premise} => {conseq_str}, w={self.weight:.3f})"


def create_rules_from_premises(
    premises: List["Premise"],
    X: torch.Tensor,
    y: torch.Tensor,
    firing_strengths: torch.Tensor,
    tsk_order: int = 0,
    regularization: float = 1e-3,
    input_names: Optional[Sequence[str]] = None,
    input_ranges: Optional[Sequence[Tuple[float, float]]] = None,
) -> List[FuzzyRule]:
    """
    Create complete fuzzy rules by fitting TSK consequents.

    Args:
        premises: List of Premise objects
        X: Input features [N, D]
        y: Target values [N] or [N, H]
        firing_strengths: [N, R] firing strengths
        tsk_order: TSK order
        regularization: Ridge regularization
        input_names: Name of each column of X (must match the inference system's inputs)
        input_ranges: (min, max) of each column of X

    Returns:
        List of FuzzyRule objects
    """
    consequents = fit_tsk_consequents_batch(
        X, y, firing_strengths,
        order=tsk_order,
        regularization=regularization,
        input_names=input_names,
        input_ranges=input_ranges,
    )

    rules = []
    for i, (premise, consequent) in enumerate(zip(premises, consequents)):
        rule = FuzzyRule(
            premise=premise,
            consequent=consequent,
            weight=1.0,
            quality_score=premise.support,
        )
        rules.append(rule)

    return rules


def create_mamdani_rules_from_premises(
    premises: List["Premise"],
    y: torch.Tensor,
    firing_strengths: torch.Tensor,
    output_partitions: List["FuzzyPartition"],
    method: str = "cm",
) -> List[FuzzyRule]:
    """
    Create Mamdani rules by associating each premise with an output term.

    For each horizon, the output term is the one with the highest association
    (CM or GCF) between the premise's firing strengths and the target's
    memberships in that term.

    Args:
        premises: List of Premise objects
        y: Target values [N] or [N, H]
        firing_strengths: [N, R] firing strengths
        output_partitions: One FuzzyPartition of the target per horizon
        method: 'cm' (mean compatibility) or 'gcf' (fuzzy confidence degree)

    Returns:
        List of FuzzyRule objects whose consequent is a term index
        (a list of H indices if multi-horizon)
    """
    y = y.reshape(y.shape[0], -1)

    best_terms = []  # one list of R term indices per horizon
    for h, partition in enumerate(output_partitions):
        with torch.no_grad():
            y_memberships = partition.fuzzify(y[:, h]).cpu()  # [N, T]
        best_idx, _ = select_best_consequent_term(firing_strengths, y_memberships, method)
        best_terms.append(best_idx.tolist())

    multi_horizon = len(output_partitions) > 1

    rules = []
    for r, premise in enumerate(premises):
        terms = [best_terms[h][r] for h in range(len(output_partitions))]
        rules.append(FuzzyRule(
            premise=premise,
            consequent=terms if multi_horizon else terms[0],
            weight=1.0,
            quality_score=premise.support,
        ))

    return rules
