from dir_empm.dir_matlab_macros import * ;
from finufft import nufft2d3 as nufft2d3_cpu

def xxnufft2d3(
        nj=None,
        xj=None,
        yj=None,
        cj=None,
        iflag=None,
        eps=None,
        nk=None,
        sk=None,
        tk=None,
        finufft_opts=None,
        ):
    eps_scalar = float(eps) if np.isscalar(eps) else float(eps.item())
    iflag_scalar = int(iflag) if np.isscalar(iflag) else int(iflag.item())
    if finufft_opts is not None:
        output_ = torch.tensor(nufft2d3_cpu(
            x=np.asarray(xj, dtype=np.float64),
            y=np.asarray(yj, dtype=np.float64),
            c=np.asarray(cj, dtype=np.complex128),
            isign=iflag_scalar,
            eps=eps_scalar,
            s=np.asarray(sk, dtype=np.float64),
            t=np.asarray(tk, dtype=np.float64),
            **finufft_opts,
        ));
    #end;%if finufft_opts is None:
    if finufft_opts is None:
        output_ = torch.tensor(nufft2d3_cpu(
            x=np.asarray(xj, dtype=np.float64),
            y=np.asarray(yj, dtype=np.float64),
            c=np.asarray(cj, dtype=np.complex128),
            isign=iflag_scalar,
            eps=eps_scalar,
            s=np.asarray(sk, dtype=np.float64),
            t=np.asarray(tk, dtype=np.float64),
        ));
    #end;%if finufft_opts is None:
    return output_ ;
