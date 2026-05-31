from dir_empm.dir_matlab_macros import * ;
from dir_empm.interp_x_c_to_k_p_xxnufft import interp_x_c_to_k_p_xxnufft ;

def interp_x_c_xxM___to_k_p_wkM__xxnufft(
    n_x1=None, 
    diameter_x1_c=None, 
    n_x2=None, 
    diameter_x2_c=None, 
    n_M=None,
    M_x_c_xxM___=None, 
    n_r=None, 
    grid_k_p_=None, 
    n_w_=None,
    flag_u_vs_c=None,
    ):
    
    n_w_sum = int(torch.sum(n_w_).item());
    M_k_p_wkM__ = torch.zeros(mtr((n_w_sum,n_M))).to(dtype=torch.complex64);
    for nM in range(n_M):
        M_x_c_xx__ = M_x_c_xxM___[nM,:,:];
        M_k_p_wk_ = interp_x_c_to_k_p_xxnufft(
            n_x1, 
            diameter_x1_c, 
            n_x2, 
            diameter_x2_c, 
            M_x_c_xx__, 
            n_r, 
            grid_k_p_, 
            n_w_,
            flag_u_vs_c,
        );
        M_k_p_wkM__[nM,:] = M_k_p_wk_;
    #end;%for nM=0:n_M-1;
    
    return M_k_p_wkM__ ;
