"""Turning churn probabilities into a decision: who should get a retention offer?

The default numbers are assumptions, not something in the dataset. Change them to fit a real company.
"""

import numpy as np

OFFER_COST = 50.0       # cost of one retention offer (discount, call, gift), per targeted customer
ACCEPT_RATE = 0.30      # share of real churners who stay because of the offer
MONTHS_KEPT = 12        # if they stay, we keep roughly a year of their monthly bill


def customer_value(monthly_charges, months_kept=MONTHS_KEPT) -> np.ndarray:
    return np.asarray(monthly_charges, dtype=float) * months_kept


def expected_gain(proba, monthly_charges, accept_rate=ACCEPT_RATE, months_kept=MONTHS_KEPT) -> np.ndarray:
    """What we expect to save by making the offer, before paying for it."""
    return np.asarray(proba) * accept_rate * customer_value(monthly_charges, months_kept)


def profit(y_true, target, monthly_charges, offer_cost=OFFER_COST, accept_rate=ACCEPT_RATE,
           months_kept=MONTHS_KEPT) -> float:
    """Money saved by offering to everyone in `target`, minus what the offers cost."""
    y_true = np.asarray(y_true)
    target = np.asarray(target, dtype=bool)
    value = customer_value(monthly_charges, months_kept)
    saved = (y_true[target] * accept_rate * value[target]).sum()
    return float(saved - offer_cost * target.sum())


def best_threshold(y_true, proba, monthly_charges, grid=None) -> tuple[float, float]:
    grid = np.linspace(0.05, 0.95, 91) if grid is None else grid
    results = [(t, profit(y_true, proba >= t, monthly_charges)) for t in grid]
    return max(results, key=lambda r: r[1])
