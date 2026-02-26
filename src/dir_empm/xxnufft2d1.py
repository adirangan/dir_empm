from dir_empm.dir_matlab_macros import * ;
from finufft import nufft2d1 as nufft2d1_cpu

def xxnufft2d1(
        nj,
        xj,
        yj,
        cj,
        iflag,
        eps,
        ms,
        mt,
        ):
    eps_scalar = float(eps) if np.isscalar(eps) else float(eps.item())
    iflag_scalar = int(iflag) if np.isscalar(iflag) else int(iflag.item())
    ms_scalar = int(ms) if np.isscalar(ms) else int(ms.item())
    mt_scalar = int(mt) if np.isscalar(mt) else int(mt.item())
    N_ = ( ms_scalar , mt_scalar )
    return torch.tensor(nufft2d1_cpu(
        x=np.asarray(xj, dtype=np.float64),
        y=np.asarray(yj, dtype=np.float64),
        c=np.asarray(cj, dtype=np.complex128),
        isign=iflag_scalar,
        eps=eps_scalar,
        n_modes = N_,
    ) / max(1,nj) );
