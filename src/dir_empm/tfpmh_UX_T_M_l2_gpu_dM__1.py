from dir_empm.dir_matlab_macros import * ;

def tfpmh_UX_T_M_l2_gpu_dM__1(
        device_use=None,
        n_delta_v=None,
        n_svd_l=None,
        c16_svd_U_d_expiw_s_gpu__=None,
        n_w_max=None,
        n_M=None,
        pm_n_UX_rank=None,
        svd_VUXM_gpu_lwnM____=None,
):
    flag_verbose=0;
    str_thisfunction = 'tfpmh_UX_T_M_l2_gpu_dM__1';
    if flag_verbose > 0: print(f' %% [entering {str_thisfunction}]');
    svd_VUXM_gpu_lwnM____ = svd_VUXM_gpu_lwnM____.to(device=device_use);
    c16_svd_U_d_expiw_s_gpu__ = c16_svd_U_d_expiw_s_gpu__.to(device=device_use);

    UX_T_M_k_q_gpu_dwnM____ = torch.reshape( mmmm( torch.reshape(c16_svd_U_d_expiw_s_gpu__.to(dtype=torch.complex64,device=device_use),mtr((n_delta_v,n_svd_l))) , torch.reshape(svd_VUXM_gpu_lwnM____.to(dtype=torch.complex64,device=device_use),mtr((n_svd_l,n_w_max*pm_n_UX_rank*n_M))) ) , mtr((n_delta_v,n_w_max,pm_n_UX_rank,n_M)));
    UX_T_M_l2_gpu_dM__ = torch.zeros(mtr((n_delta_v,n_M))).to(dtype=torch.float32,device=device_use);
    UX_T_M_l2_gpu_dM__ = torch.reshape(torch.real(torch.sum(torch.reshape(torch.abs(torch.permute(UX_T_M_k_q_gpu_dwnM____,mtr(mts((1,2,0,3)))))**2,mtr((n_w_max*pm_n_UX_rank,n_delta_v,n_M))),2-0) * n_w_max),mtr((n_delta_v,n_M)));
    if flag_verbose > 0: print(f' %% [finished {str_thisfunction}]');
    return(UX_T_M_l2_gpu_dM__);
