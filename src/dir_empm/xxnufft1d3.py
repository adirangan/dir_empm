from dir_empm.dir_matlab_macros import * ;
from finufft import nufft1d3 as nufft1d3_cpu

def xxnufft1d3(
        nj,
        xj,
        fc,
        iflag,
        eps,
        nk,
        sk,
        ):
    eps_scalar = float(eps) if np.isscalar(eps) else float(eps.item())
    iflag_scalar = int(iflag) if np.isscalar(iflag) else int(iflag.item())
    return torch.tensor(nufft1d3_cpu(
        x=np.asarray(xj, dtype=np.float64),
        f=np.asarray(fc, dtype=np.complex128),
        isign=iflag_scalar,
        eps=eps_scalar,
        s=np.asarray(sk, dtype=np.float64),
    ));
