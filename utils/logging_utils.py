import os
import json

class Logger:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.log_file = os.path.join(out_dir, 'log.txt')
        
    def log(self, metrics, step=None):
        msg = f"Step {step}: " if step is not None else ""
        msg += ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()])
        print(msg)
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps({'step': step, **metrics}) + "\n")
