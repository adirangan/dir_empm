from dir_empm.dir_matlab_macros import * ;
from finufft import nufft3d3 as nufft3d3_cpu ;

def xxnufft3d3(
        nj,
        xj,
        yj,
        zj,
        cj,
        iflag,
        eps,
        nk,
        sk,
        tk,
        uk,
        ):
    eps_scalar = float(eps) if np.isscalar(eps) else float(eps.item()) ;
    iflag_scalar = int(iflag) if np.isscalar(iflag) else int(iflag.item()) ;
    return torch.tensor(nufft3d3_cpu(
        x=np.asarray(xj, dtype=np.float64),
        y=np.asarray(yj, dtype=np.float64),
        z=np.asarray(zj, dtype=np.float64),
        c=np.asarray(cj, dtype=np.complex128),
        isign=iflag_scalar,
        eps=eps_scalar,
        s=np.asarray(sk, dtype=np.float64),
        t=np.asarray(tk, dtype=np.float64),
        u=np.asarray(uk, dtype=np.float64),
    ));
