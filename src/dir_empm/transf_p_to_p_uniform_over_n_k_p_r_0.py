from dir_empm.dir_matlab_macros import * ;

#%%%%%%%%;
#% Assumes that n_w_ = n_w_max*ones(n_k_p_r,1);
#% This 'per-template' translation is appropriate ;
#% for the construction of an array of the form M_k_p_wkdM___ ;
#% (as might be required for the FTK). ;
#%%%%%%%%;

def transf_p_to_p_uniform_over_n_k_p_r_0(
        n_k_p_r=None,
        k_p_r_=None,
        n_w_=None,
        n_S=None,
        S_k_p_wkS__=None,
        n_delta_v=None,
        r8_delta_x_dS__=None,
        r8_delta_y_dS__=None,
):
    flag_verbose=0;
    str_thisfunction = 'transf_p_to_p_uniform_over_n_k_p_r_0' ;

    #%%%%%%%%;
    if (flag_verbose>0): disp(sprintf(' %% [finished %s]',str_thisfunction)); #end;
    #%%%%%%%%;
    
    n_w_ = n_w_.ravel(); n_w_max = int(torch.max(n_w_).item()); n_w_sum = int(torch.sum(n_w_).item()); n_w_csum_ = cumsum_0(n_w_);
    if numel_unique(n_w_)> 1: disp(sprintf(' %% Warning, n_w_ not uniform in %s',str_thisfunction)); #end;

    #%%%%%%%%%%%%%%%%;
    gamma_z_ = torch.linspace(0,2*pi,n_w_max+1).to(dtype=torch.float32); gamma_z_ = gamma_z_[0:n_w_max].ravel();
    k_c_0_wk_ = (torch.reshape(+torch.cos(gamma_z_),mtr((n_w_max,n_1)))*torch.reshape(k_p_r_,mtr((n_1,n_k_p_r)))).ravel(); assert(numel(k_c_0_wk_)==n_w_sum);
    k_c_1_wk_ = (torch.reshape(+torch.sin(gamma_z_),mtr((n_w_max,n_1)))*torch.reshape(k_p_r_,mtr((n_1,n_k_p_r)))).ravel(); assert(numel(k_c_1_wk_)==n_w_sum);
    L_c_wkdS___ = torch.reshape(k_c_0_wk_,mtr((n_w_sum,n_1,n_1))) * torch.reshape(r8_delta_x_dS__,mtr((n_1,n_delta_v,n_S))) + torch.reshape(k_c_1_wk_,mtr((n_w_sum,n_1,n_1))) * torch.reshape(r8_delta_y_dS__,mtr((n_1,n_delta_v,n_S))) ;
    C_c_wkdS___ = torch.exp(-i*2*pi*L_c_wkdS___);
    T_M_k_p_wkdS___ = torch.reshape( C_c_wkdS___ * torch.reshape(S_k_p_wkS__,mtr((n_w_sum,n_1,n_S))),mtr((n_w_sum,n_delta_v,n_S)));
    #%%%%%%%%%%%%%%%%;

    #%%%%%%%%;
    if (flag_verbose>0): disp(sprintf(' %% [finished %s]',str_thisfunction)); #end;
    #%%%%%%%%;
    return(T_M_k_p_wkdS___);


