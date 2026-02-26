from dir_empm.dir_matlab_macros import * ;
from dir_empm.transf_p_to_p_uniform_over_n_k_p_r_gpu_0 import transf_p_to_p_uniform_over_n_k_p_r_gpu_0 ;
from dir_empm.interp_p_to_q import interp_p_to_q ; #<-- should apply on device_use=='cuda'. ;

def tpmh_UXTM_gpu_dwnM____0(
        device_use=None,
        FTK=None,
        n_k_p_r=None,
        k_p_gpu_r_=None,
        n_w_=None,
        n_M=None,
        M_k_p_gpu_wkM__=None,
        pm_n_UX_rank=None,
        pm_UX_gpu_kn__=None,
        pm_X_weight_gpu_r_=None,
):
    flag_verbose=0;
    str_thisfunction = 'tpmh_UXTM_gpu_dwnM____0';

    if (flag_verbose>0): disp(sprint(' %% [entering %s]',str_thisfunction)); #end;

    n_w_ = n_w_.ravel(); n_w_max = int(torch.max(n_w_).item()); n_w_sum = int(torch.sum(n_w_).item()); n_w_csum_ = cumsum_0(n_w_);
    if numel_unique(n_w_)> 1: disp(sprintf(' %% Warning, n_w_ not uniform in %s',str_thisfunction)); end;

    k_p_gpu_r_ = k_p_gpu_r_.to(dtype=torch.float32,device=device_use);
    M_k_p_gpu_wkM__ = M_k_p_gpu_wkM__.to(dtype=torch.complex64,device=device_use);
    pm_UX_gpu_kn__ = pm_UX_gpu_kn__.to(dtype=torch.float32,device=device_use);
    pm_X_weight_gpu_r_ = pm_X_weight_gpu_r_.to(dtype=torch.float32,device=device_use);

    n_delta_v = FTK['n_delta_v'];
    delta_x_gpu_dM__ = torch.tile(torch.reshape(FTK['r8_delta_x_'],mtr((n_delta_v,n_1))),mtr((n_1,n_M))).to(dtype=torch.float32,device=device_use);
    delta_y_gpu_dM__ = torch.tile(torch.reshape(FTK['r8_delta_y_'],mtr((n_delta_v,n_1))),mtr((n_1,n_M))).to(dtype=torch.float32,device=device_use);
    T_M_k_p_gpu_wkdM___ = transf_p_to_p_uniform_over_n_k_p_r_gpu_0(
        device_use,
        n_k_p_r,
        k_p_gpu_r_,
        n_w_,
        n_M,
        M_k_p_gpu_wkM__,
        n_delta_v,
        delta_x_gpu_dM__,
        delta_y_gpu_dM__,
    );

    pm_wUX_gpu_kn__ = torch.reshape(pm_X_weight_gpu_r_,mtr((n_k_p_r,n_1))) * torch.reshape(pm_UX_gpu_kn__,mtr((n_k_p_r,pm_n_UX_rank))) ;
    str_einsum = msr('kn') + ',' + msr('wkdM') + '->' + msr('wndM');
    UX_T_M_k_p_gpu_wndM____ = torch.reshape(torch.einsum( str_einsum , pm_wUX_gpu_kn__.to(dtype=torch.complex64,device=device_use) , torch.reshape(T_M_k_p_gpu_wkdM___.to(dtype=torch.complex64,device=device_use),mtr((n_w_max,n_k_p_r,n_delta_v,n_M))) ),mtr((n_w_max,pm_n_UX_rank,n_delta_v,n_M)));
    pm_n_w_ = n_w_max*torch.ones(pm_n_UX_rank).to(dtype=torch.int32); pm_n_w_sum = int(torch.sum(pm_n_w_).item());
    UX_T_M_k_q_gpu_wndM____ = torch.reshape(interp_p_to_q(pm_n_UX_rank,pm_n_w_,pm_n_w_sum,UX_T_M_k_p_gpu_wndM____),mtr((n_w_max,pm_n_UX_rank,n_delta_v,n_M)));
    UX_T_M_k_q_gpu_dwnM____ = torch.permute(UX_T_M_k_q_gpu_wndM____,mtr(mts((2,0,1,3))));

    if (flag_verbose>0): disp(sprint(' %% [finished %s]',str_thisfunction)); #end;
    return(UX_T_M_k_q_gpu_dwnM____);
