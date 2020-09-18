

class WarmupOptimizer(object):
    def __init__(self, lr_base, optimizer, epoch_steps, warmup):
        self.optimizer = optimizer
        self._step = 0
        self.lr_base = lr_base
        self._rate = 0
        self.epoch_steps = epoch_steps
        self.warmup = warmup
        self.eta_min = 0.
        self.T_max = 20

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def rate(self, step=None):
        if step is None:
            step = self._step
        epoch_ = int(step / self.epoch_steps) - 3

        if self.warmup:
            if step <= int(self.epoch_steps * 1):
                r = self.lr_base * 1/4.
            elif step <= int(self.epoch_steps * 2):
                r = self.lr_base * 2/4.
            elif step <= int(self.epoch_steps * 3):
                r = self.lr_base * 3/4.
            else:
                r = self.lr_base
                # r = self.eta_min + (self.lr_base - self.eta_min) * (1 + math.cos(math.pi * epoch_ / self.T_max)) / 2
        else:
            r = self.lr_base
            # r = self.eta_min + (self.lr_base - self.eta_min) * (1 + math.cos(math.pi * epoch_ / self.T_max)) / 2

        return r

    def decay(self, decay_r):
        self.lr_base *= decay_r

    def set_start_step(self, step):
        self._step = step