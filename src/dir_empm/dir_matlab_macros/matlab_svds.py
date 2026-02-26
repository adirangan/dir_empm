import numpy as np ; import torch ;
ndims = lambda a : a.ndim ;
size = lambda a , d : a.shape[ndims(a)-1-d] ;

def matlab_svd(
        X_rc__=None,
):
    n_r = size(X_rc__,0); n_c = size(X_rc__,1);
    tmp_UX_rn__ , tmp_SX_n_ , tmp_VX_cn__ = torch.linalg.svd(X_rc__.T,full_matrices=False); 
    tmp_UX_rn__ = tmp_UX_rn__.T; #%<-- extra transposes to match matlab. ;
    return(
        tmp_UX_rn__,
        tmp_SX_n_,
        tmp_VX_cn__,
    );

def matlab_svds(
        X_rc__=None,
        n_svd=None,
):
    n_r = size(X_rc__,0); n_c = size(X_rc__,1);
    if n_svd is None: n_svd = int(np.minimum(n_r,n_c)); #end;
    tmp_UX_rn__ , tmp_SX_n_ , tmp_VX_cn__ = matlab_svd(X_rc__);
    tmp_UX_rn__ = tmp_UX_rn__[0:n_svd,:];
    tmp_SX_n_ = tmp_SX_n_[0:n_svd];
    tmp_VX_cn__ = tmp_VX_cn__[0:n_svd,:];
    return(
        tmp_UX_rn__,
        tmp_SX_n_,
        tmp_VX_cn__,
    );

