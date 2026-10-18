"""
Edge latency benchmark. Reproduces Section IV.D.1 / new Section IV.E.
Run inside a throttled container to emulate edge hardware.
"""
import json, os, time
import numpy as np
import pandas as pd

# Import runtime modules
from runtime.rtca import RTCA
from runtime.putc_engine import PUTCEngine
from runtime.ram import RAM
from runtime.rpee import RPEE

ARTIFACTS = "artifacts"
LOGS = os.path.join(ARTIFACTS, "logs")
os.makedirs(LOGS, exist_ok=True)

# Load a domain config (default: smart_home)
import yaml
with open("config/smart_home.yaml") as f:
    cfg = yaml.safe_load(f)

# Load (or generate) the policy matrix
matrix_path = os.path.join(ARTIFACTS, "policy_matrix.json")
if not os.path.exists(matrix_path):
    with open(matrix_path, "w") as f:
        json.dump({
            "default": {"policy_id": 0, "privacy_risk": 0.049,
                        "utility": 0.91, "threat": 0.02},
            "context_0": {"policy_id": 17, "privacy_risk": 0.031,
                          "utility": 0.88, "threat": 0.01},
            "context_1": {"policy_id": 42, "privacy_risk": 0.047,
                          "utility": 0.93, "threat": 0.03},
        }, f, indent=2)

rtca = RTCA(cfg)
putc = PUTCEngine()
ram = RAM(matrix_path, "smart_home")
rpee = RPEE()

N = 500
records = []
rng = np.random.default_rng(42)

for i in range(N):
    sensor_row = {sig: float(rng.normal(0.5, 0.2)) for sig in cfg["context_signals"]}
    context, t_detect = rtca.detect(sensor_row)
    ctx_id = f"context_{i % 2}"
    policy, _ = ram.lookup(ctx_id)
    result = rpee.enforce(policy, sensor_row)
    t_enforce = result["t_enforce"]
    latency = t_enforce - t_detect
    records.append({"i": i, "t_detect": t_detect,
                    "t_enforce": t_enforce, "latency_ms": latency})

df = pd.DataFrame(records)
df.to_csv(os.path.join(LOGS, "hil_latency.csv"), index=False)

mean_ms = float(df.latency_ms.mean())
p95_ms = float(np.percentile(df.latency_ms, 95))
print(f"Edge latency over N={N} invocations:")
print(f"  Mean    : {mean_ms:.2f} ms")
print(f"  95th pct: {p95_ms:.2f} ms")
print(f"  Log     : {os.path.join(LOGS, 'hil_latency.csv')}")