from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional
import time


class Logger:
    def __init__(
        self,
        log_dir: str,
        run_name: Optional[str] = None,
    ):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        name = f"{run_name}_{timestamp}" if run_name else timestamp
        self.writer = SummaryWriter(log_dir=f"{log_dir}/{name}")
        self.global_step = 0

    # -------------------------
    # Scalar logging
    # -------------------------
    def log_scalar(self, key: str, value: float, step: Optional[int] = None):
        step = self.global_step if step is None else step
        self.writer.add_scalar(key, value, step)

    def log_scalars(self, metrics: Dict[str, float], step: Optional[int] = None):
        for k, v in metrics.items():
            self.log_scalar(k, v, step)

    # -------------------------
    # Histogram logging
    # -------------------------
    def log_histogram(self, key: str, values, step: Optional[int] = None):
        step = self.global_step if step is None else step
        self.writer.add_histogram(key, values, step)

    # -------------------------
    # Step management
    # -------------------------
    def step(self, n: int = 1):
        self.global_step += n

    def close(self):
        self.writer.close()
