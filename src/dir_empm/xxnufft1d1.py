from dir_empm.dir_matlab_macros import * ;
from finufft import nufft1d1 as nufft1d1_cpu

def xxnufft1d1(
        nj,
        xj,
        cj,
        iflag,
        eps,
        ms,
        ):
    eps_scalar = float(eps) if np.isscalar(eps) else float(eps.item())
    iflag_scalar = int(iflag) if np.isscalar(iflag) else int(iflag.item())
    ms_scalar = int(ms) if np.isscalar(ms) else int(ms.item())
    return torch.tensor(nufft1d1_cpu(
        x=np.asarray(xj, dtype=np.float64),
        c=np.asarray(cj, dtype=np.complex128),
        isign=iflag_scalar,
        eps=eps_scalar,
        ms=ms_scalar,
    ) / max(1,nj) ) ;
