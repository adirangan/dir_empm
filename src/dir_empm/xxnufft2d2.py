import numpy as np ; import torch
from finufft import nufft2d2 as nufft2d2_cpu

def xxnufft2d2(
        nj,
        xj,
        yj,
        iflag,
        eps,
        ms,
        mt,
        fk,
        ):
    eps_scalar = float(eps) if np.isscalar(eps) else float(eps.item())
    iflag_scalar = int(iflag) if np.isscalar(iflag) else int(iflag.item())
    ms_scalar = int(ms) if np.isscalar(ms) else int(ms.item())
    mt_scalar = int(mt) if np.isscalar(mt) else int(mt.item())
    return torch.tensor(nufft2d2_cpu(
        x=np.asarray(xj, dtype=np.float64),
        y=np.asarray(yj, dtype=np.float64),
        isign=iflag_scalar,
        eps=eps_scalar,
        f=np.asarray(fk, dtype=np.complex128),
    ));
