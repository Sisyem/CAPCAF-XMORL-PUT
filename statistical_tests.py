"""Paired t-tests with Bonferroni correction (Table XVI)."""
from scipy import stats
import numpy as np

def paired_t_test(baseline, capcaf, n_metrics=20):
    t, p = stats.ttest_rel(baseline, capcaf)
    p_bonf = min(1.0, p * n_metrics)
    d = (np.mean(capcaf) - np.mean(baseline)) / np.std(capcaf - baseline, ddof=1)
    ci = stats.t.interval(0.95, len(baseline) - 1,
                          loc=np.mean(capcaf - baseline),
                          scale=stats.sem(capcaf - baseline))
    return {"t": t, "p_bonferroni": p_bonf, "cohen_d": d, "ci_95": ci}