import numpy as np ; import torch ;
isempty = lambda t: (t is None) or (isinstance(t, torch.Tensor) and t.numel() == 0) ;

def matlab_chebpts_0(
        n_gamma_z=None,
        width_=None,
):
    
    if isempty(width_):
        width_=torch.zeros(2).to(dtype=torch.float32);
        width_[0]=-1.0; width_[1]=-1.0;
    #end;%if isempty(width_):

    tsch_node_, tsch_weight_ = np.polynomial.chebyshev.chebgauss(n_gamma_z) #<-- note these range from high to low, which is the opposite of matlab chebpts. ;
    tsch_node_   = torch.tensor(  tsch_node_).to(dtype=torch.float32) ;
    tsch_weight_ = torch.tensor(tsch_weight_).to(dtype=torch.float32) ; #<-- np.polynomial.chebyshev.chebgauss corresponds to inverse-sqrt-weighted quadrature');
    tsch_weight_ = tsch_weight_ * torch.sqrt(1.0 - tsch_node_**2) ;
    cheb_node_ = torch.flip(tsch_node_,(0,))*torch.mean(torch.diff(width_)).item()/2.0 + torch.mean(width_).item();
    cheb_weight_ = tsch_weight_ * torch.mean(torch.diff(width_)).item() / 2.0 ;

    return(
        cheb_node_,
        cheb_weight_,
    );
