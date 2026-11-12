import numpy as np ; import torch ;
from scipy.special import jv ;

def besselj(nu,x):
    if isinstance(x,torch.Tensor):
        tmp_dtype = x.dtype if (x.is_floating_point() or x.is_complex()) else torch.float64 ;
        return torch.tensor(jv(nu,x.detach().cpu().numpy())).to(dtype=tmp_dtype,device=x.device) ;
    #end;%if isinstance(x,torch.Tensor);
    return jv(nu,x) ;
