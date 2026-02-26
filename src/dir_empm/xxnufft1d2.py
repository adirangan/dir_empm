from dir_empm.dir_matlab_macros import * ;
from finufft import nufft1d2 as nufft1d2_cpu

def xxnufft1d2(
        nj,
        xj,
        iflag,
        eps,
        ms,
        fk,
        ):
    eps_scalar = float(eps) if np.isscalar(eps) else float(eps.item())
    iflag_scalar = int(iflag) if np.isscalar(iflag) else int(iflag.item())
    ms_scalar = int(ms) if np.isscalar(ms) else int(ms.item())
    return torch.tensor(nufft1d2_cpu(
        x=np.asarray(xj, dtype=np.float64),
        isign=iflag_scalar,
        eps=eps_scalar,
        f=np.asarray(fk, dtype=np.complex128),
    ));
