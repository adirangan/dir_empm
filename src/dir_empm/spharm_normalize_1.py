from dir_empm.dir_matlab_macros import * ;

def spharm_normalize_1(
        n_k_p_r=None,
        k_p_r_=None,
        weight_3d_k_p_r_=None,
        l_max_=None,
        a_k_Y_=None,
):
    #% Normalizes the spherical-harmonic expansion to have norm 1, but does not center. ;
    n_y_ = (l_max_+1)**2;
    n_y_sum = int(torch.sum(n_y_).item());
    n_y_csum_ = cumsum_0(n_y_);
    l_max_max = int(torch.max(l_max_).item());
    m_max_ = torch.arange(-l_max_max,+l_max_max+1).to(dtype=torch.int32);
    n_m_max = numel(m_max_);
    weight_Y_val_ = torch.zeros(n_y_sum).to(dtype=torch.float32);
    e_k_Y_ = torch.zeros(n_y_sum).to(dtype=torch.float32);
    na=0;
    for nk_p_r in range(n_k_p_r):
        n_y = int(n_y_[nk_p_r].item());
        tmp_index_ = na + torch.arange(n_y).to(dtype=torch.int32);
        weight_Y_val_[tmp_index_] = weight_3d_k_p_r_[nk_p_r].item()*torch.ones(n_y).to(dtype=torch.float32);
        e_k_Y_[na] = 1;
        na=na+n_y;
    #end;%for nk_p_r=0:n_k_p_r-1;
    e_avg = torch.sum(torch.conj(e_k_Y_)*weight_Y_val_*e_k_Y_).item();
    u_k_Y_ = e_k_Y_/np.maximum(1e-12,np.sqrt(e_avg));
    a_avg = torch.sum(torch.conj(u_k_Y_)*weight_Y_val_*a_k_Y_).item();
    a_k_Y_norm_ = (a_k_Y_ - 0.0*a_avg*u_k_Y_); #%<-- do not center. ;
    a_std = np.sqrt(torch.sum(torch.conj(a_k_Y_norm_)*weight_Y_val_*a_k_Y_norm_).item());
    a_k_Y_norm_ = a_k_Y_norm_/np.maximum(1e-12,a_std);
    
    return(
        a_k_Y_norm_,
        a_avg,
        a_std,
        u_k_Y_,
    );
