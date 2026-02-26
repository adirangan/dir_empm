from dir_empm.dir_matlab_macros import * ;

def tfpmh_UX_T_M_l2_dM__0(
        FTK=None,
        n_w_=None,
        n_M=None,
        pm_n_UX_rank=None,
        UX_T_M_k_q_dwnM____=None,
):
    flag_verbose=0;
    str_thisfunction = 'tfpmh_UX_T_M_l2_dM__0';
    if (flag_verbose>0): disp(sprintf(' %% [entering %s]',str_thisfunction)); #end;
    UX_T_M_l2_dM__ = torch.zeros(mtr((FTK['n_delta_v'],n_M))).to(dtype=torch.float32);
    n_w_max = int(torch.max(n_w_).item()); n_k_p_r = numel(n_w_); n_w_sum = int(torch.sum(n_w_).item());
    if (n_w_sum!=n_w_max*n_k_p_r): disp(sprintf(' %% Warning, n_w_ not uniform in %s',str_thisfunction)); #end;
    UX_T_M_l2_dM__ = torch.reshape(torch.sum(torch.abs(UX_T_M_k_q_dwnM____)**2,dim=(3-1,3-2),keepdim=True),mtr((FTK['n_delta_v'],n_M))) / np.maximum(1,n_w_max) ;
    return(UX_T_M_l2_dM__);
