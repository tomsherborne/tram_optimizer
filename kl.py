"""
Forward pass KL divergence class
"""
import torch
import torch.nn.functional as F

class L2Distance:
    def __init__(self):
        pass

    def _l2(x: torch.Tensor, y:  torch.Tensor) -> torch.Tensor:
        return (y-x).pow(2).sum(-1).sqrt().mean()

    def get_divergence(self,
                          x: torch.Tensor, 
                          y: torch.Tensor
    ) -> torch.Tensor:
        return self._l2(x, y) / x.size(0)


class MaximumMeanDiscrepancy:
    def __init__(self, kernel_type: str = "imq2"):
        if kernel_type not in ['imq', 'imq2', 'imq3', 'rbf']:
            raise NotImplementedError(f"Kernel type {kernel_type} not recognized!")
        self.kernel_type = kernel_type

    def _mmd(self,
            x: torch.Tensor, 
            y: torch.Tensor
    ) -> torch.Tensor: 
        # https://www.kaggle.com/code/onurtunali/maximum-mean-discrepancy
        # https://github.com/schelotto/Wasserstein-AutoEncoders/blob/master/wae_mmd.py
        kernel_bandwidth=[0.1, 0.2, 0.5, 1., 2., 5., 10.]
        xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))
        dxx = rx.t() + rx - 2 * xx
        dyy = ry.t() + ry - 2 * yy
        dxy = rx.t() + ry - 2 * zz

        XX, YY, XY = (torch.zeros(xx.shape, device=x.device),
                    torch.zeros(xx.shape, device=x.device),
                    torch.zeros(xx.shape, device=x.device))

        if self.kernel_type == "imq":
            for kb in kernel_bandwidth:
                XX += kb * (kb + dxx)**-1
                YY += kb * (kb + dyy)**-1
                XY += kb * (kb + dxy)**-1  
        elif self.kernel_type == "imq2":
            for kb in kernel_bandwidth:
                XX +=  (kb**2 + dxx)**-0.5
                YY +=  (kb**2 + dyy)**-0.5
                XY +=  (kb**2 + dxy)**-0.5
        elif self.kernel_type == "imq3":
            for kb in kernel_bandwidth:
                XX += kb**2 * (kb**2 + dxx)**-1
                YY += kb**2 * (kb**2 + dyy)**-1
                XY += kb**2 * (kb**2 + dxy)**-1
        elif self.kernel_type == "rbf":
            for kb in kernel_bandwidth:
                XX += torch.exp(-0.5*dxx / kb)
                YY += torch.exp(-0.5*dyy / kb)
                XY += torch.exp(-0.5*dxy / kb)

        return torch.mean(XX + YY - 2 * XY)
    
    def get_divergence(self,
                          x: torch.Tensor, 
                          y: torch.Tensor
    ) -> torch.Tensor:
        return self._mmd(x, y) / x.size(0)


class KLDivergence:
    """
    KL Divergence with options for forward, reverse and symmetric KL.
    Assume that X is the target and Y is the model observations in all cases
    i.e. X||Y is forward Y||X is reverse and X||Y + Y||X is symmetric.

    kl_type = ['symmetric', 'forward', 'reverse']
    """
    def __init__(self, kl_type: str = "forward"):
        self.kl_type = kl_type

        if self.kl_type == "forward":
            self.klfn = lambda x, y: self._kl(x, y)
        elif self.kl_type == "reverse":
            self.klfn = lambda x, y: self._kl(y, x)
        elif self.kl_type == "symmetric":
            self.klfn = lambda x, y: self._kl(x, y) + self._kl(y, x)
             
    def _kl(self,
            x: torch.Tensor, 
            y: torch.Tensor
    ) -> torch.Tensor:
        return F.kl_div(
            input=F.log_softmax(y, dim=-1, dtype=torch.float32),
            target=F.log_softmax(x, dim=-1, dtype=torch.float32),
            log_target=True,
            reduction="mean"
        )

    def get_divergence(self,
                          x: torch.Tensor, 
                          y: torch.Tensor
    ) -> torch.Tensor:
        return self.klfn(x, y) / x.size(0)
