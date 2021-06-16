from torch.optim.lr_scheduler import LambdaLR
import math


class LinearWarmUpScheduler(LambdaLR):
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(LinearWarmUpScheduler, self).__init__(optimizer=optimizer, lr_lambda=self.lr_lambda,
                                                    last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))

        return max(0.0, float(step - self.warmup_steps) / float(max(1.0, self.t_total - self.warmup_steps)))


class CosineWarmUpScheduler(LambdaLR):
    def __init__(self, optimizer, warmup_steps, t_total, cycle=0.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycle = cycle
        super(CosineWarmUpScheduler, self).__init__(optimizer=optimizer, lr_lambda=self.lr_lambda,
                                                    last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))

        progress = float(step - self.warmup_steps) / float(max(1.0, self.t_total - step))

        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(self.cycle) * 2.0 * progress)))
