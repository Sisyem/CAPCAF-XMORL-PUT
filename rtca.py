"""Real-Time Context Acquisition (Section III.D.1)."""
import time

class RTCA:
    def __init__(self, domain_cfg):
        self.cfg = domain_cfg

    def detect(self, sensor_row):
        """Returns (context_vector, t_detect)."""
        t_detect = time.perf_counter_ns() / 1e6   # ms
        context = [sensor_row.get(sig, 0.0) for sig in self.cfg["context_signals"]]
        return context, t_detect