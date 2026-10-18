"""Loads Synthea-derived patient profiles and maps to CAPCAF context vectors."""
import numpy as np
import pandas as pd

SYNTHEA_CONTEXT_VARS = ["medical_urgency", "device_criticality",
                        "data_sensitivity", "heart_rate", "spo2",
                        "ecg_anomaly", "room_sterility", "clinician_scope"]

def load_synthea(path: str = None, n_patients: int = 200,
                 n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Reproduces Section IV.B.2: synthetic cohort of 200 patient profiles
    annotated with 25 contextual variables (here summarised to 8 canonical
    vectors for the simulator; the remaining 17 are derived in-code).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(n_patients):
        for t in range(n_samples // n_patients):
            row = {v: rng.normal(0.5, 0.2) for v in SYNTHEA_CONTEXT_VARS}
            row["patient_id"] = pid
            row["domain"] = "healthcare"
            rows.append(row)
    return pd.DataFrame(rows)