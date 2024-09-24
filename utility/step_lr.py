import math
import numpy as np

class StepLR:
    def __init__(self, optimizer, learning_rate: float, total_epochs: int):
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.base = learning_rate
        self.type = 'cos'
        self.min_value = 0
        self.max_value = learning_rate

    def __call__(self, epoch):
        if self.type == 'step':
            if epoch < self.total_epochs * 3/10:
                lr = self.base
            elif epoch < self.total_epochs * 6/10:
                lr = self.base * 0.2
            elif epoch < self.total_epochs * 8/10:
                lr = self.base * 0.2 ** 2
            else:
                lr = self.base * 0.2 ** 3

        elif self.type == 'cos':
            phase = (epoch) / (self.total_epochs) * math.pi
            lr = self.min_value + (self.max_value - self.min_value) * (np.cos(phase) + 1.) / 2.0

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]
