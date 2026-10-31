"""Finite-family risk calibration; established binomial bounds, not new theory.

Each calibration observation must be an independent frame/channel draw. A
preselected random user per frame supplies one Bernoulli observation; pooling
the co-scheduled users as independent Bernoulli trials is not allowed.
"""
from dataclasses import dataclass
import numpy as np
from scipy.stats import beta


def binomial_upper(errors: int, trials: int, failure_probability: float) -> float:
    """One-sided exact Clopper--Pearson upper confidence limit."""
    if (not isinstance(errors, (int, np.integer)) or
            not isinstance(trials, (int, np.integer)) or
            trials < 1 or not 0 <= errors <= trials or
            not 0 < failure_probability < 1):
        raise ValueError("Invalid binomial observations or confidence budget")
    if errors == trials:
        return 1.0
    return float(beta.ppf(1 - failure_probability, errors + 1, trials - errors))


@dataclass(frozen=True)
class RiskSelection:
    selected: str
    upper_limits: dict
    eligible: tuple
    alpha: float
    delta: float
    family_size: int


def select_policy(harm_events: dict, development_costs: dict,
                  alpha: float, delta: float = 0.05,
                  reference: str = "exact") -> RiskSelection:
    """Select the cheapest certified finite candidate, or the reference.

    Policies and the condition mixture must be frozen before calibration.
    Cost ordering is fixed from independent development data. Bonferroni
    simultaneously bounds every candidate, so selecting among them is valid
    without assuming risk is monotone in iteration or interpolation stride.
    The reference has identically zero *excess* loss, not zero absolute BLER.
    The guarantee concerns repeated calibration samples under the same
    independent sampling distribution; distribution shift voids that premise.
    """
    if not 0 < alpha < 1 or not 0 < delta < 1 or reference in harm_events:
        raise ValueError("Invalid risk budgets or candidate/reference names")
    if set(harm_events) | {reference} != set(development_costs) or not harm_events:
        raise ValueError("Every candidate and the reference need development cost")
    if not np.isfinite(development_costs[reference]) or development_costs[reference] < 0:
        raise ValueError("Invalid reference development cost")
    bounds = {}
    count = len(harm_events)
    for name, events in harm_events.items():
        events = np.asarray(events)
        if events.ndim != 1 or not len(events) or np.any((events != 0) & (events != 1)):
            raise ValueError("One binary event per independent frame is required")
        if not np.isfinite(development_costs[name]) or development_costs[name] < 0:
            raise ValueError("Invalid development cost")
        bounds[name] = binomial_upper(int(events.sum()), len(events), delta / count)
    eligible = tuple(sorted((k for k, v in bounds.items() if v <= alpha),
                            key=lambda k: (development_costs[k], k)))
    selected = min((*eligible, reference), key=lambda k: (development_costs[k], k))
    return RiskSelection(selected, bounds,
                         eligible, alpha, delta, count)
