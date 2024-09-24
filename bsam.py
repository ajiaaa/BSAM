import torch
from utility.save_file import write_to_file


class BSAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho_max_sharp, rho_min_sharp, rho_scheduler, adaptive=False, **kwargs):

        defaults = dict(rho_max_sharp=rho_max_sharp, adaptive=adaptive, **kwargs)
        super(BSAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer
        self.rho_scheduler = rho_scheduler
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.rho_max_sharp = rho_max_sharp
        self.rho_min_sharp = rho_min_sharp
        self.update_rho_t()
        self.adaptive = False
        self.max_norm = 0
        self.sgd_norm = 0
        self.min_norm = 0

    @torch.no_grad()
    def update_rho_t(self):
        self.rho_min_sharp = self.rho_scheduler.step()
        return self.rho_min_sharp

    @torch.no_grad()
    def to_max_point(self, zero_grad=False):
        self.sgd_norm = self._grad_norm(weight_adaptive=self.adaptive)
        # self.sgd_norm = self._grad_norm()
        scale = self.rho_max_sharp / (self.sgd_norm + 1e-12)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["sgd_g"] = p.grad.clone()
                self.state[p]["old_p"] = p.data.clone()
                e_w = p.grad * scale.to(p)

                if self.adaptive:
                    e_w *= torch.pow(p, 2)
                p.add_(e_w)

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def to_min_point(self, zero_grad=False):
        self.max_norm = self._grad_norm(weight_adaptive=self.adaptive)# self._grad_norm()
        scale = self.rho_min_sharp / (self.sgd_norm + 1e-12)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["max_g"] = p.grad.clone()
                p.data = self.state[p]["old_p"]
                e_w = self.state[p]["sgd_g"] * scale.to(p)

                if self.adaptive:
                    e_w *= torch.pow(p, 2)
                p.sub_(e_w)

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def opt_max_min_step(self, epoch, zero_grad=False):
        self.min_norm = self._grad_norm(weight_adaptive=self.adaptive)# self._grad_norm()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]
                p.grad = (self.state[p]["max_g"]) + (self.state[p]["sgd_g"] - (self.max_norm / self.min_norm) * p.grad)

        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def _grad_norm(self, by=None, weight_adaptive=False):
        # shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        if not by:
            norm = torch.norm(
                torch.stack([
                    ((torch.abs(p.data) if weight_adaptive else 1.0) * p.grad).norm(p=2)
                    for group in self.param_groups for p in group["params"]
                    if p.grad is not None
                ]),
                p=2
            )
        else:
            norm = torch.norm(
                torch.stack([
                    ((torch.abs(p.data) if weight_adaptive else 1.0) * self.state[p][by]).norm(p=2)
                    for group in self.param_groups for p in group["params"]
                    if p.grad is not None
                ]),
                p=2
            )
        return norm

    @torch.no_grad()
    def _grad_norm_by(self, by=None, weight_adaptive=False):
        # shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        if not by:
            norm = torch.norm(
                torch.stack([
                    ((torch.abs(p.data) if weight_adaptive else 1.0) * p.grad).norm(p=2)
                    for group in self.param_groups for p in group["params"]
                    if p.grad is not None
                ]),
                p=2
            )
        else:
            norm = torch.norm(
                torch.stack([
                    ((torch.abs(p.data) if weight_adaptive else 1.0) * self.state[p][by]).norm(p=2)
                    for group in self.param_groups for p in group["params"]
                    if p.grad is not None
                ]),
                p=2
            )
        return norm

    @torch.no_grad()
    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
