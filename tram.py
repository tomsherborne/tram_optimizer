import torch
import logging
from torch.optim.optimizer import Optimizer

from tram import enable_running_stats, disable_running_stats

logger = logging.getLogger(__name__)

class TRAM(Optimizer):
    def __init__(self, params, base_optimizer, sam_args, **kwargs):
        self.sam_args = sam_args
        rho = sam_args['rho']
        adaptive = sam_args['adaptive']
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(TRAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, logit_divergence, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = logit_divergence / (grad_norm + 1e-12)
            
            # Imitate Adam._init_group() but simpler
            params_with_grad = []
            grads = []
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    grads.append(p.grad)
                    self.state[p]["p_old"] = p.data.clone()

            grouped_tensors = torch.utils._foreach_utils._group_tensors_by_device_and_dtype(
                [params_with_grad, grads]
            )

            # torch signature is ((device_params, device_grads), _).
            for (device_params, device_grads) in grouped_tensors.values():
                # Handle complex parameters
                device_grads = [torch.view_as_real(x) if torch.is_complex(x) else x for x in device_grads]
                device_params = [torch.view_as_real(x) if torch.is_complex(x) else x for x in device_params]
                device_scale = scale.clone().to(device_params[0])
                if group["adaptive"]:
                    e_w = torch._foreach_mul(device_params, device_params)
                else:
                    e_w = [torch.ones_like(p.grad) for p in device_params]

                torch._foreach_mul_(e_w, device_scale)
                torch._foreach_add_(device_params, e_w)
                
                del e_w, device_grads, device_scale

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["p_old"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def both_step(self, closure=None):
        """
        This is the step() functionality in the original implementation
        """
        assert (
            closure is not None
        ), "TRAM requires closure, but it was not provided"
        closure = torch.enable_grad()(
            closure
        )  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)

        closure()
        
        self.second_step(zero_grad=True)

    @torch.no_grad()
    def step(self, closure=None):
        """
        This is the second_step() call outside `training_step` that HF will call
        """
        self.second_step(zero_grad=True)

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][
            0
        ].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack(
                [
                    ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad)
                    .norm(p=2)
                    .to(shared_device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2,
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
