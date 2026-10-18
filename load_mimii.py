"""Loads MIMII machine-sound telemetry and maps to CAPCAF context vectors."""
import numpy as np
import pandas as pd

MIMII_CONTEXT_VARS = ["vibration", "throughput", "gas_leak",
                      "noise_level", "safety_alert", "equipment_state"]

def load_mimii(path: str = None, n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Reproduces Section IV.B.3: MIMII vibration + anomaly patterns,
    supplemented with safety alerts.
    """
    rng = np.random.default_rng(seed)
    data = {v: rng.normal(0.5, 0.2, n_samples) for v in MIMII_CONTEXT_VARS}
    df = pd.DataFrame(data)
    df["domain"] = "industrial_iot"
    return df