from dir_empm.dir_matlab_macros import * ;

def tfpmh_UX_T_M_l2_dM__1(
        FTK=None,
        n_w_=None,
        n_M=None,
        pm_n_UX_rank=None,
        svd_V_UX_M_lwnM____=None,
):
    flag_verbose=0;
    str_thisfunction = 'tfpmh_UX_T_M_l2_dM__1';
    if flag_verbose > 0: print(f' %% [entering {str_thisfunction}]');
    UX_T_M_l2_dM__ = torch.zeros(mtr((FTK['n_delta_v'],n_M))).to(dtype=torch.float32);
    n_w_max = int(torch.max(n_w_).item());
    UX_T_M_k_q_dwnM____ = torch.reshape( mmmm( torch.reshape(FTK['c16_svd_U_d_expiw_s__'].to(dtype=torch.complex64),mtr((FTK['n_delta_v'],FTK['n_svd_l']))) , torch.reshape(svd_V_UX_M_lwnM____.to(dtype=torch.complex64),mtr((FTK['n_svd_l'],n_w_max*pm_n_UX_rank*n_M))) ) , mtr((FTK['n_delta_v'],n_w_max,pm_n_UX_rank,n_M)));
    UX_T_M_l2_dM__ = torch.reshape(torch.real(torch.sum(torch.reshape(torch.abs(torch.permute(UX_T_M_k_q_dwnM____,mtr(mts((1,2,0,3)))))**2,mtr((n_w_max*pm_n_UX_rank,FTK['n_delta_v'],n_M))),2-0) * n_w_max),mtr((FTK['n_delta_v'],n_M)));
    if flag_verbose > 0: print(f' %% [finished {str_thisfunction}]');
    return(UX_T_M_l2_dM__);
