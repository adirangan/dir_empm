from dir_empm.dir_matlab_macros import * ;
from dir_empm.transf_p_to_p_uniform_over_n_k_p_r_0 import transf_p_to_p_uniform_over_n_k_p_r_0 ;
from dir_empm.interp_p_to_q import interp_p_to_q ;

def tpmh_UXTM_dwnM____0(
        FTK=None,
        n_k_p_r=None,
        k_p_r_=None,
        n_w_=None,
        n_M=None,
        M_k_p_wkM__=None,
        pm_n_UX_rank=None,
        pm_UX_kn__=None,
        pm_X_weight_r_=None,
):
    flag_verbose=0;
    str_thisfunction = 'tpmh_UXTM_dwnM____0';

    if (flag_verbose>0): disp(sprint(' %% [entering %s]',str_thisfunction)); #end;

    n_w_ = n_w_.ravel(); n_w_max = int(torch.max(n_w_).item()); n_w_sum = int(torch.sum(n_w_).item()); n_w_csum_ = cumsum_0(n_w_);
    if numel_unique(n_w_)> 1: disp(sprintf(' %% Warning, n_w_ not uniform in %s',str_thisfunction)); end;

    n_delta_v = FTK['n_delta_v'];
    delta_x_dM__ = torch.tile(torch.reshape(FTK['r8_delta_x_'],mtr((n_delta_v,n_1))),mtr((n_1,n_M)));
    delta_y_dM__ = torch.tile(torch.reshape(FTK['r8_delta_y_'],mtr((n_delta_v,n_1))),mtr((n_1,n_M)));
    T_M_k_p_wkdM___ = transf_p_to_p_uniform_over_n_k_p_r_0(
        n_k_p_r,
        k_p_r_,
        n_w_,
        n_M,
        M_k_p_wkM__,
        n_delta_v,
        delta_x_dM__,
        delta_y_dM__,
    );

    pm_wUX_kn__ = torch.reshape(pm_X_weight_r_,mtr((n_k_p_r,n_1))) * torch.reshape(pm_UX_kn__,mtr((n_k_p_r,pm_n_UX_rank))) ;
    str_einsum = msr('kn') + ',' + msr('wkdM') + '->' + msr('wndM');
    UX_T_M_k_p_wndM____ = torch.reshape(torch.einsum( str_einsum , pm_wUX_kn__.to(dtype=torch.complex64) , torch.reshape(T_M_k_p_wkdM___.to(dtype=torch.complex64),mtr((n_w_max,n_k_p_r,n_delta_v,n_M))) ),mtr((n_w_max,pm_n_UX_rank,n_delta_v,n_M)));
    pm_n_w_ = n_w_max*torch.ones(pm_n_UX_rank).to(dtype=torch.int32); pm_n_w_sum = int(torch.sum(pm_n_w_).item());
    UX_T_M_k_q_wndM____ = torch.reshape(interp_p_to_q(pm_n_UX_rank,pm_n_w_,pm_n_w_sum,UX_T_M_k_p_wndM____),mtr((n_w_max,pm_n_UX_rank,n_delta_v,n_M)));
    UX_T_M_k_q_dwnM____ = torch.permute(UX_T_M_k_q_wndM____,mtr(mts((2,0,1,3))));

    if (flag_verbose>0): disp(sprint(' %% [finished %s]',str_thisfunction)); #end;
    return(UX_T_M_k_q_dwnM____);
