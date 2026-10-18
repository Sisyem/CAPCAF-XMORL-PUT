"""All metrics reported in Tables XI–XVI."""
import numpy as np

def csc(domains_results):
    """Eq. (12): context-sensitive coverage averaged over domains."""
    return float(np.mean([r["csc"] for r in domains_results]))

def violation_rate(domains_results):
    return float(np.mean([r["violation"] for r in domains_results]))

def relative_improvement(v_baseline, v_capcaf, lower_is_better=True):
    """Eqs. (17)–(18)."""
    if lower_is_better:
        return (v_baseline - v_capcaf) / v_baseline * 100
    return (v_capcaf - v_baseline) / v_baseline * 100

def igd(reference, obtained):
    """Eq. (26): IGD = (1/|R|) * sum_r min_a d(r,a)."""
    dists = [np.min(np.linalg.norm(reference - a, axis=1)) for a in obtained]
    return float(np.mean(dists))

def spread(front):
    """Eq. (27): Spread metric."""
    d = np.linalg.norm(np.diff(front, axis=0), axis=1)
    d_mean = np.mean(d)
    df = np.linalg.norm(front[0] - front[-1])
    dl = df
    n = len(front)
    num = df + dl + np.sum(np.abs(d - d_mean))
    den = df + dl + (n - 1) * d_mean
    return float(num / den) if den else 0.0