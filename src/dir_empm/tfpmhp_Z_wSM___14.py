from dir_empm.dir_matlab_macros import * ;
from dir_empm.interp_p_to_q import interp_p_to_q ;
from dir_empm.tpmh_VUXM_lwnM____3 import tpmh_VUXM_lwnM____3 ;
from dir_empm.tpmh_UXTM_dwnM____0 import tpmh_UXTM_dwnM____0 ;
from dir_empm.tfpmh_UX_T_M_l2_dM__1 import tfpmh_UX_T_M_l2_dM__1 ;
from dir_empm.tfpmh_UX_T_M_l2_dM__0 import tfpmh_UX_T_M_l2_dM__0 ;
from dir_empm.tfpmhh_pm_wUX_0 import tfpmhh_pm_wUX_0 ;

def tfpmhp_Z_wSM___14(
        parameter=None,
        n_k_p_r=None,
        k_p_r_=None,
        k_p_r_max=None,
        n_w_=None,
        weight_2d_k_p_r_=None,
        weight_2d_k_p_wk_=None,
        n_M=None,
        M_k_p_wkM__=None,
        CTF_k_p_r_k_=None,
        n_S=None,
        S_k_p_wkS__=None,
        pm_n_UX_rank=None,
        pm_UX_kn__=None,
        pm_X_weight_r_=None,
        FTK=None,
        index_nM_=None,
        M_k_q_wkM__=None,
        UX_T_M_l2_dM__=None,
        UX_M_l2_M_=None,
        svd_V_UX_M_lwnM____=None,
        index_nS_=None,
        UX_CTF_S_k_q_wnS__=None,
        UX_CTF_S_l2_S_=None,
):

    #%%%%%%%%;
    #% precomputation for tfpmh_Z_wSM___14. ;
    #% Note that the radial compression is applied here, and is structured to align with the canonical innerproduct. ;
    #% (i.e., can use radial compression even when the loss is high). ;
    #% Note that this takes in a single (isotropic) CTF_k_p_r_k_, ;
    #% Assumes n_w_ = n_w_max*ones(n_k_p_r,1);
    #%%%%%%%%;

    str_thisfunction = 'tfpmhp_Z_wSM___14' ;

    if not isinstance(parameter, dict): parameter = {'type': 'parameter'};
    if 'flag_verbose' not in parameter: parameter['flag_verbose'] = 0 ;
    flag_verbose = parameter['flag_verbose'];
    if 'tolerance_master' not in parameter: parameter['tolerance_master'] = 1e-2 ;
    tolerance_master = parameter['tolerance_master'];
    if 'memory_limit_GB' not in parameter: parameter['memory_limit_GB'] = 4.0 ;
    memory_limit_GB = parameter['memory_limit_GB'];
    #%%%%%%%%;    
    if 'flag_precompute_M_k_q_wkM__' not in parameter: parameter['flag_precompute_M_k_q_wkM__'] = 0 ;
    flag_precompute_M_k_q_wkM__ = parameter['flag_precompute_M_k_q_wkM__'];
    if 'flag_precompute_UX_T_M_l2_dM__' not in parameter: parameter['flag_precompute_UX_T_M_l2_dM__'] = 0 ;
    flag_precompute_UX_T_M_l2_dM__ = parameter['flag_precompute_UX_T_M_l2_dM__'];
    if 'flag_precompute_UX_M_l2_M_' not in parameter: parameter['flag_precompute_UX_M_l2_M_'] = 0 ;
    flag_precompute_UX_M_l2_M_ = parameter['flag_precompute_UX_M_l2_M_'];
    if 'flag_precompute_svd_V_UX_M_lwnM____' not in parameter: parameter['flag_precompute_svd_V_UX_M_lwnM____'] = 0 ;
    flag_precompute_svd_V_UX_M_lwnM____ = parameter['flag_precompute_svd_V_UX_M_lwnM____'];
    if 'flag_precompute_UX_CTF_S_k_q_wnS__' not in parameter: parameter['flag_precompute_UX_CTF_S_k_q_wnS__'] = 0 ;
    flag_precompute_UX_CTF_S_k_q_wnS__ = parameter['flag_precompute_UX_CTF_S_k_q_wnS__'];
    if 'flag_precompute_UX_CTF_S_l2_S_' not in parameter: parameter['flag_precompute_UX_CTF_S_l2_S_'] = 0 ;
    flag_precompute_UX_CTF_S_l2_S_ = parameter['flag_precompute_UX_CTF_S_l2_S_'];
    #%%%%%%%%;    
    if 'device_use' not in parameter: parameter['device_use'] = 'cpu' ; #<-- need to control batch sizes below in order to push onto cuda. ;
    device_use = parameter['device_use'];
    if 'flag_output_ravel' not in parameter: parameter['flag_output_ravel'] = 0 ;
    flag_output_ravel = parameter['flag_output_ravel'];

    #%%%%%%%%;
    if (flag_verbose): print(f' %% [entering {str_thisfunction}]');
    #%%%%%%%%;

    tolerance_machine = 1e-6;

    n_w_ = n_w_.ravel();
    n_w_max = int(torch.max(n_w_).item());
    n_w_sum = int(torch.sum(n_w_).item());
    n_w_csum_ = cumsum_0(n_w_);
    if (n_w_sum!=n_w_max*n_k_p_r): disp(sprintf(' %% Warning, n_w_sum %d ~= n_w_max*n_k_p_r %d*%d in %s',n_w_sum,n_w_max,n_k_p_r,str_thisfunction)); #end;
    if np.mod(n_w_max,2)!=0:  disp(sprintf(' %% Warning, n_w_max %d in %s',n_w_max,str_thisfunction)); #end;
    n_delta_v = FTK['n_delta_v'];
    n_svd_l = FTK['n_svd_l'];
    #tmp_index_d0 = intersect_0(efind(FTK['r8_delta_x_']==0),efind(FTK['r8_delta_y_']==0))[0]; assert(numel(tmp_index_d0)==1); #%<-- should be a single index corresponding to zero-displacement. ;
    tmp_index_d0 = intersect_0(efind(torch.abs(FTK['r8_delta_x_'])<tolerance_machine),efind(torch.abs(FTK['r8_delta_y_'])<tolerance_machine))[0]; assert(numel(tmp_index_d0)>=1);
    if numel(tmp_index_d0)> 1: tmp_index_d0 = tmp_index_d0[0]; #end;
    pm_n_k_p_r = pm_n_UX_rank; pm_n_w_max = n_w_max;
    pm_n_w_ = (pm_n_w_max*torch.ones(pm_n_k_p_r)).to(dtype=torch.int32);
    pm_n_w_sum = int(pm_n_k_p_r*pm_n_w_max);
    pm_wUX_kn__ = mmmm( torch.diagflat(pm_X_weight_r_).to(dtype=torch.float32) , pm_UX_kn__ );

    #%%%%%%%%;
    #% allocate memory for output. ;
    #%%%%%%%%;
    if not isempty(index_nM_):
        if flag_precompute_M_k_q_wkM__==1: 
            if isempty(M_k_q_wkM__): M_k_q_wkM__ = torch.zeros(mtr((n_w_sum,n_M))).to(dtype=torch.complex64); #end; end;
        if flag_precompute_UX_T_M_l2_dM__==1: 
            if isempty(UX_T_M_l2_dM__): UX_T_M_l2_dM__ = torch.zeros(mtr((n_delta_v,n_M))).to(dtype=torch.float32); #end; end;
        if flag_precompute_UX_M_l2_M_==1: 
            if isempty(UX_M_l2_M_): UX_M_l2_M_ = torch.zeros(n_M).to(dtype=torch.float32); #end; end;
        if flag_precompute_svd_V_UX_M_lwnM____==1: 
            if isempty(svd_V_UX_M_lwnM____): 
                if FTK['flag_tf_vs_bf']==1: svd_V_UX_M_lwnM____ = torch.zeros(mtr((n_svd_l,n_w_max,pm_n_k_p_r,n_M))).to(dtype=torch.complex64); #end;
                if FTK['flag_tf_vs_bf']==0: svd_V_UX_M_lwnM____ = torch.zeros(mtr((n_delta_v,n_w_max,pm_n_k_p_r,n_M))).to(dtype=torch.complex64); #end; %<-- UX_T_M_k_q_dwnM____. ;
            #end;%if isempty(svd_V_UX_M_lwnM____);
        #end;%if flag_precompute_svd_V_UX_M_lwnM____==1;
    #end;%if ~isempty(index_nM_);
    #%%%%%%%%;
    if not isempty(index_nS_):
        if flag_precompute_UX_CTF_S_k_q_wnS__==1: 
            if isempty(UX_CTF_S_k_q_wnS__): UX_CTF_S_k_q_wnS__ = torch.zeros(mtr((pm_n_w_sum,n_S))).to(dtype=torch.complex64); #end; end;
        if flag_precompute_UX_CTF_S_l2_S_==1: 
            if isempty(UX_CTF_S_l2_S_): UX_CTF_S_l2_S_ = torch.zeros(mtr((n_S,1))).to(dtype=torch.float32); #end; end;
    #end;%if ~isempty(index_nS_);
    #%%%%%%%%;

    if (flag_verbose>1):
        n_wkM_GB = n_w_max*n_k_p_r*n_M*n_byte_per_complex64/1e9;
        n_dM_GB = n_delta_v*n_M*n_byte_per_float32/1e9;
        n_M_GB = n_M*n_byte_per_float32/1e9;
        n_lwnM_GB = n_svd_l*n_w_max*pm_n_UX_rank*n_M*n_byte_per_complex64/1e9;
        n_dwnM_GB = n_delta_v*n_w_max*pm_n_UX_rank*n_M*n_byte_per_complex64/1e9;
        n_wnS_GB = n_w_max*pm_n_UX_rank*n_S*n_byte_per_complex64/1e9;
        n_S_GB = n_S*n_byte_per_float32/1e9;
        if flag_precompute_M_k_q_wkM__==1:
            tmp_str = 'M_k_q_wkM__'; disp(sprintf(' %% memory: %s --> %0.6f GB',tmp_str,n_wkM_GB)); #end;
        if flag_precompute_UX_T_M_l2_dM__==1:
            tmp_str = 'UX_T_M_l2_dM__'; disp(sprintf(' %% memory: %s --> %0.6f GB',tmp_str,n_dM_GB)); #end;
        if flag_precompute_UX_M_l2_M_==1:
            tmp_str = 'UX_M_l2_M_'; disp(sprintf(' %% memory: %s --> %0.6f GB',tmp_str,n_M_GB)); #end;
        if flag_precompute_svd_V_UX_M_lwnM____==1:
            if FTK['flag_tf_vs_bf']==1:
                tmp_str = 'svd_V_UX_M_lwnM____'; disp(sprintf(' %% memory: %s --> %0.6f GB',tmp_str,n_lwnM_GB));  #end;#end;
            if FTK['flag_tf_vs_bf']==0:
                tmp_str = 'svd_V_UX_M_lwnM____'; disp(sprintf(' %% memory: %s --> %0.6f GB',tmp_str,n_dwnM_GB));  #end;#end;
        #end;%if flag_precompute_svd_V_UX_M_lwnM____==1;
        if flag_precompute_UX_CTF_S_k_q_wnS__==1:
            tmp_str = 'UX_CTF_S_k_q_wnS__'; disp(sprintf(' %% memory: %s --> %0.6f GB',tmp_str,n_wnS_GB)); #end;
        if flag_precompute_UX_CTF_S_l2_S_==1:
            tmp_str = 'UX_CTF_S_l2_S_'; disp(sprintf(' %% memory: %s --> %0.6f GB',tmp_str,n_S_GB)); #end;
    #end;%if (flag_verbose>1); 
    
    #%%%%%%%%%%%%%%%%;
    if not isempty(index_nM_):
    #%%%%%%%%%%%%%%%%;
        n_M_sub = numel(index_nM_);
        #%%%%;
        #% Prepare M_k_q_wnM__. ;
        #%%%%;
        if flag_precompute_M_k_q_wkM__==1:
            tmp_t = tic();
            tmp_i8_index_lhs_ = matlab_index_2d_0(n_w_sum,':',n_M,index_nM_);
            tmp_i8_index_rhs_ = matlab_index_2d_0(n_w_sum,':',n_M,index_nM_);
            M_k_q_wkM__.ravel()[tmp_i8_index_lhs_] = interp_p_to_q(n_k_p_r,n_w_,n_w_sum,torch.reshape(M_k_p_wkM__.ravel()[tmp_i8_index_rhs_],mtr((n_w_sum,n_M_sub))));
            tmp_t = toc(tmp_t); 
            if (flag_verbose>1): disp(sprintf(' %% M_k_q_wnM__: %0.6fs',tmp_t)); end;
            parameter = parameter_timing_update(parameter,sprintf('%s: precompute M_k_q_wnM__',str_thisfunction),tmp_t);
        #end;%if flag_precompute_M_k_q_wkM__==1;
        #%%%%;
        #% Prepare quasi-images. ;
        #%%%%;
        if flag_precompute_svd_V_UX_M_lwnM____==1:
            if FTK['flag_tf_vs_bf']==1:
                tmp_t = tic();
                tmp_i8_index_lhs_ = matlab_index_4d_0(n_svd_l,':',n_w_max,':',pm_n_UX_rank,':',n_M,index_nM_);
                tmp_i8_index_rhs_ = matlab_index_2d_0(n_w_sum,':',n_M,index_nM_);
                svd_V_UX_M_lwnM____.ravel()[tmp_i8_index_lhs_] = tpmh_VUXM_lwnM____3(FTK,n_k_p_r,n_w_,n_M_sub,torch.reshape(M_k_q_wkM__.ravel()[tmp_i8_index_rhs_],mtr((n_w_sum,n_M_sub))),pm_n_UX_rank,pm_UX_kn__,pm_X_weight_r_).ravel();
                tmp_t = toc(tmp_t);
                if (flag_verbose>1): disp(sprintf(' %% svd_V_UX_M_lwnM____: %0.6fs',tmp_t)); end;
                parameter = parameter_timing_update(parameter,sprintf('%s: precompute svd_V_UX_M_lwnM____',str_thisfunction),tmp_t);
            #end;%if FTK['flag_tf_vs_bf']==1;
            if FTK['flag_tf_vs_bf']==0:
                tmp_t = tic();
                tmp_i8_index_lhs_ = matlab_index_4d_0(n_delta_v,':',n_w_max,':',pm_n_UX_rank,':',n_M,index_nM_);
                tmp_i8_index_rhs_ = matlab_index_2d_0(n_w_sum,':',n_M,index_nM_);
                svd_V_UX_M_lwnM____.ravel()[tmp_i8_index_lhs_] = tpmh_UXTM_dwnM____0(FTK,n_k_p_r,k_p_r_,n_w_,n_M_sub,torch.reshape(M_k_p_wkM__.ravel()[tmp_i8_index_rhs_],mtr((n_w_sum,n_M_sub))),pm_n_UX_rank,pm_UX_kn__,pm_X_weight_r_).ravel();
                tmp_t = toc(tmp_t);
                if (flag_verbose>1): disp(sprintf(' %% UX_T_M_k_q_dwnM____: %0.6fs',tmp_t)); end;
                parameter = parameter_timing_update(parameter,sprintf('%s: precompute UX_T_M_k_q_dwnM____',str_thisfunction),tmp_t);
            #end;%if FTK['flag_tf_vs_bf']==0;
        #end;%if flag_precompute_svd_V_UX_M_lwnM____==1;
        #%%%%;
        #% Now calculate norms of the translated images. ;
        #%%%%;
        if flag_precompute_UX_T_M_l2_dM__==1:
            if FTK['flag_tf_vs_bf']==1:
                tmp_t = tic();
                tmp_i8_index_lhs_ = matlab_index_2d_0(n_delta_v,':',n_M,index_nM_);
                tmp_i8_index_rhs_ = matlab_index_4d_0(n_svd_l,':',n_w_max,':',pm_n_UX_rank,':',n_M,index_nM_);
                UX_T_M_l2_dM__.ravel()[tmp_i8_index_lhs_] = tfpmh_UX_T_M_l2_dM__1(FTK,n_w_,n_M_sub,pm_n_UX_rank,torch.reshape(svd_V_UX_M_lwnM____.ravel()[tmp_i8_index_rhs_],mtr((n_svd_l,n_w_max,pm_n_UX_rank,n_M_sub)))).ravel();
                tmp_t = toc(tmp_t);
                if (flag_verbose>1): disp(sprintf(' %% tfpmh_UX_T_M_l2_dM__1: %0.6fs',tmp_t)); end;
                parameter = parameter_timing_update(parameter,sprintf('%s: precompute tfpmh_UX_T_M_l2_dM__1',str_thisfunction),tmp_t);
            #end;%if FTK['flag_tf_vs_bf']==1;
            if FTK['flag_tf_vs_bf']==0:
                tmp_t = tic();
                tmp_i8_index_lhs_ = matlab_index_2d_0(n_delta_v,':',n_M,index_nM_);
                tmp_i8_index_rhs_ = matlab_index_4d_0(n_delta_v,':',n_w_max,':',pm_n_UX_rank,':',n_M,index_nM_);
                UX_T_M_l2_dM__.ravel()[tmp_i8_index_lhs_] = tfpmh_UX_T_M_l2_dM__0(FTK,n_w_,n_M_sub,pm_n_UX_rank,torch.reshape(svd_V_UX_M_lwnM____.ravel()[tmp_i8_index_rhs_],mtr((n_delta_v,n_w_max,pm_n_UX_rank,n_M_sub)))).ravel();
                tmp_t = toc(tmp_t);
                if (flag_verbose>1): disp(sprintf(' %% tfpmh_UX_T_M_l2_dM__0: %0.6fs',tmp_t)); end;
                parameter = parameter_timing_update(parameter,sprintf('%s: precompute tfpmh_UX_T_M_l2_dM__0',str_thisfunction),tmp_t);
            #end;%if FTK['flag_tf_vs_bf']==0;
        #end;%if flag_precompute_UX_T_M_l2_dM__==1;
        if flag_precompute_UX_M_l2_M_==1:
            tmp_i8_index_rhs_ = matlab_index_2d_0(n_delta_v,tmp_index_d0,n_M,index_nM_);
            UX_M_l2_M_[index_nM_] = UX_T_M_l2_dM__.ravel()[tmp_i8_index_rhs_].ravel();
        #end;%if flag_precompute_UX_M_l2_M_==1;
    #%%%%%%%%%%%%%%%%;
    #end;%if ~isempty(index_nM_);
    #%%%%%%%%%%%%%%%%;

    #%%%%%%%%%%%%%%%%;
    if not isempty(index_nS_):
    #%%%%%%%%%%%%%%%%;
        n_S_sub = numel(index_nS_);
        #%%%%;
        #% Prepare UX_CTF_S_k_q_wnS__. ;
        #%%%%;
        if flag_precompute_UX_CTF_S_k_q_wnS__==1:
            tmp_t = tic();
            tmp_i8_index_rhs_ = matlab_index_2d_0(n_w_sum,':',n_S,index_nS_);
            CTF_S_sub_k_p_wkS__ = torch.reshape(torch.reshape(CTF_k_p_r_k_,mtr((1,n_k_p_r))) * torch.reshape(S_k_p_wkS__.ravel()[tmp_i8_index_rhs_],mtr((n_w_max,n_k_p_r,n_S_sub))),mtr((n_w_sum,n_S_sub)));
            _,UX_CTF_S_sub_k_p_wnS__ = tfpmhh_pm_wUX_0(parameter,n_k_p_r,pm_n_k_p_r,pm_wUX_kn__,n_w_max,n_S_sub,CTF_S_sub_k_p_wkS__);
            tmp_i8_index_lhs_ = matlab_index_2d_0(pm_n_w_sum,':',n_S,index_nS_);
            UX_CTF_S_k_q_wnS__.ravel()[tmp_i8_index_lhs_] = interp_p_to_q(pm_n_k_p_r,pm_n_w_,pm_n_w_sum,UX_CTF_S_sub_k_p_wnS__).ravel();
            tmp_t = toc(tmp_t);
            if (flag_verbose>1): disp(sprintf(' %% UX_CTF_S_k_q_wnS__: %0.6fs',tmp_t)); end;
            parameter = parameter_timing_update(parameter,sprintf('%s: precompute UX_CTF_S_k_q_wnS__',str_thisfunction),tmp_t);
        #end;%if flag_precompute_UX_CTF_S_k_q_wnS__==1;
        #%%%%;
        #% Prepare UX_CTF_S_l2_S_. ;
        #%%%%;
        if flag_precompute_UX_CTF_S_l2_S_==1:
            tmp_t = tic();
            tmp_i8_index_rhs_ = matlab_index_2d_0(pm_n_w_sum,':',n_S,index_nS_);
            UX_CTF_S_l2_S_[index_nS_] = torch.real(torch.sum(torch.reshape(torch.conj(UX_CTF_S_k_q_wnS__).ravel()[tmp_i8_index_rhs_] * UX_CTF_S_k_q_wnS__.ravel()[tmp_i8_index_rhs_],mtr((pm_n_w_sum,n_S_sub))),dim=1-0) / np.maximum(1,pm_n_w_max)).ravel();
            tmp_t = toc(tmp_t);
            if (flag_verbose>1): disp(sprintf(' %% UX_CTF_S_l2_S_: %0.6fs',tmp_t)); end;
            parameter = parameter_timing_update(parameter,sprintf('%s: precompute UX_CTF_S_l2_S_',str_thisfunction),tmp_t);
        #end;%if flag_precompute_UX_CTF_S_l2_S_==1;
    #%%%%%%%%%%%%%%%%;
    #end;%if ~isempty(index_nS_);
    #%%%%%%%%%%%%%%%%;

    #%%%%%%%%;
    if (flag_verbose): print(f' %% [finished {str_thisfunction}]');
    #%%%%%%%%;
    if flag_output_ravel:
        if not isempty(M_k_q_wkM__): M_k_q_wkM__ = M_k_q_wkM__.ravel(); #end;
        if not isempty(UX_T_M_l2_dM__): UX_T_M_l2_dM__ = UX_T_M_l2_dM__.ravel(); #end;
        if not isempty(UX_M_l2_M_): UX_M_l2_M_ = UX_M_l2_M_.ravel(); #end;
        if not isempty(svd_V_UX_M_lwnM____): svd_V_UX_M_lwnM____ = svd_V_UX_M_lwnM____.ravel(); #end;
        if not isempty(UX_CTF_S_k_q_wnS__): UX_CTF_S_k_q_wnS__ = UX_CTF_S_k_q_wnS__.ravel(); #end;
        if not isempty(UX_CTF_S_l2_S_): UX_CTF_S_l2_S_ = UX_CTF_S_l2_S_.ravel(); #end;
    #end;%if flag_output_ravel;
    return(
        parameter,
        M_k_q_wkM__,
        UX_T_M_l2_dM__,
        UX_M_l2_M_,
        svd_V_UX_M_lwnM____,
        UX_CTF_S_k_q_wnS__,
        UX_CTF_S_l2_S_,
    );

