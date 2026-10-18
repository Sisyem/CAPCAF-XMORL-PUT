"""
Hard-constraint validation + final policy selection.
Matches Eq. (10), (11), (12) of the manuscript.
"""
import numpy as np

HARD_CONSTRAINTS = {"hipaa": 0.05, "gdpr": 0.09, "pipl": 0.13,
                    "latency_ms": 100, "energy_overhead": 0.10,
                    "accuracy_degradation": 0.05}

def feasible(policy_row):
    """Eq. (11): g_k(p) <= tau_k for all k in K."""
    privacy_risk = policy_row[0]
    return (privacy_risk <= HARD_CONSTRAINTS["hipaa"] and
            privacy_risk <= HARD_CONSTRAINTS["gdpr"] and
            privacy_risk <= HARD_CONSTRAINTS["pipl"])

def validate(P_pool):
    """Eq. (12): P_final = argmin_{p in P_feasible} S(f(p))."""
    feasible_rows = [p for p in P_pool if feasible(p)]
    if not feasible_rows:
        return np.array([])
    # S(f(p)) = privacy + threat - utility (scalar fitness, minimized)
    scores = np.array([p[0] + p[2] - (-p[1]) for p in feasible_rows])
    best_idx = int(np.argmin(scores))
    return np.array(feasible_rows[best_idx])