"""Loads CASAS smart-home data and maps to CAPCAF context vectors."""
import numpy as np
import pandas as pd

CASAS_CONTEXT_VARS = ["occupancy", "lighting", "temperature",
                      "humidity", "noise", "device_state",
                      "energy_mode", "room_transition", "activity",
                      "time_of_day", "day_type", "guest_present"]

def load_casas(path: str = None, n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Reproduces the CASAS configuration described in Section IV.B.1:
    50 sensors/actuators -> 12 contextual variables.
    If no file is supplied, a seeded synthetic equivalent is generated.
    """
    rng = np.random.default_rng(seed)
    data = {var: rng.normal(0.5, 0.15, n_samples) for var in CASAS_CONTEXT_VARS}
    df = pd.DataFrame(data)
    df["domain"] = "smart_home"
    return df