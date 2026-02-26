from dir_empm.dir_matlab_macros import * ;

def tfpmh_UX_T_M_l2_gpu_dM__0(
        device_use=None,
        n_delta_v=None,
        n_k_p_r=None,
        n_w_max=None,
        n_M=None,
        pm_n_UX_rank=None,
        UX_T_M_k_q_gpu_dwnM____=None,
):
    flag_verbose=0;
    str_thisfunction = 'tfpmh_UX_T_M_l2_gpu_dM__0';
    if (flag_verbose>0): disp(sprintf(' %% [entering %s]',str_thisfunction)); #end;
    UX_T_M_k_q_gpu_dwnM____ = UX_T_M_k_q_gpu_dwnM____.to(device=device_use);
    UX_T_M_l2_gpu_dM__ = torch.zeros(mtr((n_delta_v,n_M))).to(dtype=torch.float32,device=device_use);
    UX_T_M_l2_gpu_dM__ = torch.reshape(torch.real(torch.sum(torch.abs(UX_T_M_k_q_gpu_dwnM____)**2,dim=(3-1,3-2),keepdim=True)),mtr((n_delta_v,n_M))).to(dtype=torch.float32,device=device_use) / np.maximum(1,n_w_max);
    if (flag_verbose>0): disp(sprintf(' %% [finished %s]',str_thisfunction)); #end;
    return(UX_T_M_l2_gpu_dM__);
