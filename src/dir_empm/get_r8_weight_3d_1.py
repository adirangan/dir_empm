from dir_empm.dir_matlab_macros import * ;
from scipy.special import roots_jacobi ;

def get_r8_weight_3d_1(flag_verbose=None, r8_k_p_r_max=None, r8_k_eq_d=None, str_T_vs_L=None):
    if flag_verbose is None: flag_verbose = 0 ;
    if r8_k_p_r_max is None: r8_k_p_r_max = 1.0 ;
    if r8_k_eq_d is None: r8_k_eq_d = 1.0 / (2 * pi) ;
    if str_T_vs_L is None: str_T_vs_L = 'L' ;

    n_k_p_r = 1 + int(np.ceil(r8_k_p_r_max / r8_k_eq_d)) ;
    np_a_jx_, np_a_jw_ = roots_jacobi(n_k_p_r, 0, 2) ; #%<-- assume returns float64. ;
    r8_a_jx_ = torch.tensor(np_a_jx_).to(dtype=torch.float64);
    r8_a_jw_ = torch.tensor(np_a_jw_).to(dtype=torch.float64);
    r8_k_p_r_ = torch.reshape((r8_a_jx_ + 1.0) * r8_k_p_r_max / 2.0, (n_k_p_r,) ).to(dtype=torch.float64) ;
    r8_weight_3d_k_p_r_ = torch.reshape(r8_a_jw_ * (r8_k_p_r_max / 2.0) ** 3, (n_k_p_r,) ).to(dtype=torch.float64) ;

    return (
        n_k_p_r,
        r8_k_p_r_, 
        r8_weight_3d_k_p_r_,
    ) ;
