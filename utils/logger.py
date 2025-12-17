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
        self.update_step = 0
        self.total_step = 0
        self.metrics = {}

    # -------------------------
    # Scalar logging
    # -------------------------
    def log_scalar(self, key: str, value: float, step: Optional[int] = None):
        step = self.update_step if step is None else step
        self.writer.add_scalar(key, value, step)

    def log_scalars(self, metrics: Dict[str, float], step: Optional[int] = None):
        for k, v in metrics.items():
            self.log_scalar(k, v, step)

    def save_metrics(self, metrics: Dict[str, float]):
        for k, v in metrics.items():
            if k not in self.metrics:
                self.metrics[k] = []
            self.metrics[k].append(v)
        
    # -------------------------
    # Histogram logging
    # -------------------------
    def log_histogram(self, key: str, values, step: Optional[int] = None):
        step = self.update_step if step is None else step
        self.writer.add_histogram(key, values, step)

    def close(self):
        self.writer.close()
