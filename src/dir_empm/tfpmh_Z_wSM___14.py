from dir_empm.dir_matlab_macros import * ;
from dir_empm.interp_p_to_q import interp_p_to_q ;
from dir_empm.tpmh_VUXM_lwnM____3 import tpmh_VUXM_lwnM____3 ;
from dir_empm.tpmh_UXTM_dwnM____0 import tpmh_UXTM_dwnM____0 ;
from dir_empm.tfpmh_UX_T_M_l2_dM__0 import tfpmh_UX_T_M_l2_dM__0 ;
from dir_empm.tfpmh_UX_T_M_l2_dM__1 import tfpmh_UX_T_M_l2_dM__1 ;
from dir_empm.tfpmhh_pm_wUX_0 import tfpmhh_pm_wUX_0 ;

def tfpmh_Z_wSM___14(
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
        M_k_q_wkM__=None,
        UX_T_M_l2_dM__=None,
        UX_M_l2_M_=None,
        svd_V_UX_M_lwnM____=None,  #%<-- or UX_T_M_k_q_dwnM____ ;
        UX_CTF_S_k_q_wnS__=None,
        UX_CTF_S_l2_S_=None,
):
    str_thisfunction = 'tfpmh_Z_wSM___14' ;

    if not isinstance(parameter, dict): parameter = {'type': 'parameter'};
    if 'flag_verbose' not in parameter: parameter['flag_verbose'] = 0 ;
    flag_verbose = parameter['flag_verbose'];
    if 'tolerance_master' not in parameter: parameter['tolerance_master'] = 1e-2 ;
    tolerance_master = parameter['tolerance_master'];
    if 'n_M_per_Mbatch' not in parameter: parameter['n_M_per_Mbatch'] = 24 ;
    n_M_per_Mbatch = parameter['n_M_per_Mbatch'];
    if 'n_S_per_Sbatch' not in parameter: parameter['n_S_per_Sbatch'] = 24 ;
    n_S_per_Sbatch = parameter['n_S_per_Sbatch'];
    if 'flag_optimize_over_gamma_z' not in parameter: parameter['flag_optimize_over_gamma_z'] = 0 ;
    flag_optimize_over_gamma_z = parameter['flag_optimize_over_gamma_z'];
    if 'flag_dwSM' not in parameter: parameter['flag_dwSM'] = 0 ;
    flag_dwSM = parameter['flag_dwSM'];
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
    if (flag_verbose): disp(sprintf(' %% [entering %s]',str_thisfunction)); #end;
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
    pm_wUX_kn__ = mmmm( torch.diagflat(pm_X_weight_r_).to(dtype=torch.float32) , pm_UX_kn__.to(dtype=torch.float32) );

    #%%%%%%%%;
    #% allocate memory for output. ;
    #%%%%%%%%;
    if flag_precompute_UX_T_M_l2_dM__==0: UX_T_M_l2_dM__ = torch.zeros(mtr((n_delta_v,n_M))).to(dtype=torch.float32); #end;
    if flag_precompute_UX_M_l2_M_==0: UX_M_l2_M_ = torch.zeros(n_M).to(dtype=torch.float32); #end;
    if flag_precompute_UX_CTF_S_k_q_wnS__==0: UX_CTF_S_k_q_wnS__ = torch.zeros(mtr((pm_n_w_sum,n_S))).to(dtype=torch.complex64); #end;
    if flag_precompute_UX_CTF_S_l2_S_==0: UX_CTF_S_l2_S_ = torch.zeros(n_S).to(dtype=torch.float32); #end;
    if flag_optimize_over_gamma_z==0:
        Z_wSM___ = torch.zeros(mtr((n_w_max,n_S,n_M))).to(dtype=torch.float32);
        X_wSM___ = torch.zeros(mtr((n_w_max,n_S,n_M))).to(dtype=torch.float32);
        delta_x_wSM___ = torch.zeros(mtr((n_w_max,n_S,n_M))).to(dtype=torch.float32);
        delta_y_wSM___ = torch.zeros(mtr((n_w_max,n_S,n_M))).to(dtype=torch.float32);
        gamma_z_wSM___ = torch.zeros(mtr((n_w_max,n_S,n_M))).to(dtype=torch.float32);
        index_sub_wSM___ = torch.zeros(mtr((n_w_max,n_S,n_M))).to(dtype=torch.int32);
    #end;%if flag_optimize_over_gamma_z==0;
    if flag_optimize_over_gamma_z==1:
        Z_SM__ = torch.zeros(mtr((n_S,n_M))).to(dtype=torch.float32);
        X_SM__ = torch.zeros(mtr((n_S,n_M))).to(dtype=torch.float32);
        delta_x_SM__ = torch.zeros(mtr((n_S,n_M))).to(dtype=torch.float32);
        delta_y_SM__ = torch.zeros(mtr((n_S,n_M))).to(dtype=torch.float32);
        gamma_z_SM__ = torch.zeros(mtr((n_S,n_M))).to(dtype=torch.float32);
        index_sub_SM__ = torch.zeros(mtr((n_S,n_M))).to(dtype=torch.int32);
    #end;%if flag_optimize_over_gamma_z==1;

    Z_dwSM____=None;
    X_dwSM____=None;
    if flag_dwSM:
        n_dwSM = n_delta_v*n_w_max*n_S*n_M;
        n_dwSM_GB = n_dwSM*n_byte_per_float32/1e9;
        if (flag_verbose): tmp_str = 'X_dwSM____'; print(f' %% memory: {tmp_str} --> {n_dwSM_GB:.6f} GB');
        if (n_dwSM_GB> memory_limit_GB):
            print(f' %% Warning, n_dwSM_GB {n_dwSM_GB:.2f} in {str_thisfunction}');
            flag_dwSM = 0;
        #end;%if (n_dwSM_GB> memory_limit_GB);
        if (n_dwSM_GB<=memory_limit_GB):
            Z_dwSM____ = torch.zeros(mtr((n_delta_v,n_w_max,n_S,n_M))).to(dtype=torch.float32);
            X_dwSM____ = torch.zeros(mtr((n_delta_v,n_w_max,n_S,n_M))).to(dtype=torch.float32);
            flag_dwSM = 1;
        #end;%if (n_dwSM_GB<=memory_limit_GB);
    #end;%if flag_dwSM;

    if (flag_verbose>0): 
        n_dwSM_GB = n_delta_v*n_w_max*n_S*n_M*n_byte_per_float32/1e9;
        n_wSM_GB = n_w_max*n_S*n_M*n_byte_per_float32/1e9;
        n_wS_GB = n_w_max*n_S*n_byte_per_float32/1e9;
        n_dM_GB = n_delta_v*n_M*n_byte_per_float32/1e9;
        n_M_GB = n_M*n_byte_per_float32/1e9;
        tmp_str = 'UX_CTF_S_l2_S_'; print(f' %% memory: {tmp_str} --> {n_wS_GB} GB');
        tmp_str = 'CTF_S_l2_S_'; print(f' %% memory: {tmp_str} --> {n_wS_GB} GB');
        tmp_str = 'UX_T_M_l2_dM__'; print(f' %% memory: {tmp_str} --> {n_dM_GB} GB');
        tmp_str = 'UX_M_l2_M_'; print(f' %% memory: {tmp_str} --> {n_M_GB} GB');
        if flag_optimize_over_gamma_z==0:
            tmp_str = 'Z_wSM___'; print(f' %% memory: {tmp_str} --> {n_wSM_GB} GB');
            tmp_str = 'X_wSM___'; print(f' %% memory: {tmp_str} --> {n_wSM_GB} GB');
            tmp_str = 'delta_x_wSM___'; print(f' %% memory: {tmp_str} --> {n_wSM_GB} GB');
            tmp_str = 'delta_y_wSM___'; print(f' %% memory: {tmp_str} --> {n_wSM_GB} GB');
            tmp_str = 'gamma_z_wSM___'; print(f' %% memory: {tmp_str} --> {n_wSM_GB} GB');
        #end;%flag_optimize_over_gamma_z==0;
        if flag_optimize_over_gamma_z==1:
            tmp_str = 'Z_SM__'; print(f' %% memory: {tmp_str} --> {n_wSM_GB} GB');
            tmp_str = 'X_SM__'; print(f' %% memory: {tmp_str} --> {n_wSM_GB} GB');
            tmp_str = 'delta_x_SM__'; print(f' %% memory: {tmp_str} --> {n_wSM_GB} GB');
            tmp_str = 'delta_y_SM__'; print(f' %% memory: {tmp_str} --> {n_wSM_GB} GB');
            tmp_str = 'gamma_z_SM__'; print(f' %% memory: {tmp_str} --> {n_wSM_GB} GB');
        #end;%if flag_optimize_over_gamma_z==1;
        if (flag_dwSM): tmp_str = 'Z_dwSM____'; print(f' %% memory: {tmp_str} --> {n_dwSM_GB} GB');
        if (flag_dwSM): tmp_str = 'X_dwSM____'; print(f' %% memory: {tmp_str} --> {n_dwSM_GB} GB');
    #end;%if (flag_verbose>0); 

    if isempty(M_k_q_wkM__): M_k_q_wkM__=torch.zeros(mtr((n_w_sum,n_M))).to(dtype=torch.complex64); #end;
    if isempty(UX_T_M_l2_dM__): UX_T_M_l2_dM__=torch.zeros(mtr((n_delta_v,n_M))).to(dtype=torch.float32); #end;
    if isempty(UX_M_l2_M_): UX_M_l2_M_=torch.zeros(n_M).to(dtype=torch.float32); #end;
    if isempty(svd_V_UX_M_lwnM____):
        if FTK['flag_tf_vs_bf']==1:
            svd_V_UX_M_lwnM____=torch.zeros(mtr((n_svd_l,n_w_max,pm_n_UX_rank,n_M))).to(dtype=torch.complex64);
        #end;%if FTK['flag_tf_vs_bf']==1;
        if FTK['flag_tf_vs_bf']==0:
            svd_V_UX_M_lwnM____=torch.zeros(mtr((n_delta_v,n_w_max,pm_n_UX_rank,n_M))).to(dtype=torch.complex64);
        #end;%if FTK['flag_tf_vs_bf']==0;
    #end;%if isempty(svd_V_UX_M_lwnM____);
    if isempty(UX_CTF_S_k_q_wnS__): UX_CTF_S_k_q_wnS__=torch.zeros(mtr((pm_n_w_sum,n_S))).to(dtype=torch.complex64); #end;
    if isempty(UX_CTF_S_l2_S_): UX_CTF_S_l2_S_=torch.zeros(n_S).to(dtype=torch.float32); #end;

    #%%%%;
    #% Prepare UX_M_k_q_wnM__. ;
    #%%%%;
    if flag_precompute_M_k_q_wkM__==0:
        tmp_t = tic();
        M_k_q_wkM__ = interp_p_to_q(n_k_p_r,n_w_,n_w_sum,M_k_p_wkM__);
        tmp_t = toc(tmp_t);
        if (flag_verbose>1): disp(sprintf(' %% UX_M_k_q_wnM__: %0.6fs',tmp_t)); #end;
        parameter = parameter_timing_update(parameter,sprintf('%s: UX_M_k_q_wnM__',str_thisfunction),tmp_t);
    #end;%if flag_precompute_M_k_q_wkM__==0;

    #%%%%;
    #% Prepare UX_CTF_S_k_q_wnS__ and UX_CTF_S_l2_S_. ;
    #%%%%;
    if flag_precompute_UX_CTF_S_k_q_wnS__==0:
        tmp_t = tic();
        CTF_S_k_p_wkS__ = torch.reshape(torch.reshape(CTF_k_p_r_k_,mtr((1,n_k_p_r))) * torch.reshape(S_k_p_wkS__,mtr((n_w_max,n_k_p_r,n_S))),mtr((n_w_sum,n_S)));
        _,UX_CTF_S_k_p_wnS__ = tfpmhh_pm_wUX_0([],n_k_p_r,pm_n_k_p_r,pm_wUX_kn__,n_w_max,n_S,CTF_S_k_p_wkS__);
        UX_CTF_S_k_q_wnS__ = interp_p_to_q(pm_n_k_p_r,pm_n_w_,pm_n_w_sum,UX_CTF_S_k_p_wnS__);
        tmp_t = toc(tmp_t);
        if (flag_verbose>1): disp(sprintf(' %% UX_CTF_S_k_q_wnS__: %0.6fs',tmp_t)); end;
        parameter = parameter_timing_update(parameter,sprintf('%s: UX_CTF_S_k_q_wnS__',str_thisfunction),tmp_t);
    #end;%if flag_precompute_UX_CTF_S_k_q_wnS__==0;
    if flag_precompute_UX_CTF_S_l2_S_==0:
        UX_CTF_S_l2_S_ = torch.squeeze(torch.real(torch.sum(torch.reshape(torch.conj(UX_CTF_S_k_p_wnS__),mtr((pm_n_w_sum,n_S))) * torch.reshape(UX_CTF_S_k_p_wnS__,mtr((pm_n_w_sum,n_S))),1-0))) / np.maximum(1,pm_n_w_max);
    #end;%if flag_precompute_UX_CTF_S_l2_S_==0;


    n_M_per_Mbatch = int(np.maximum(1,n_M_per_Mbatch));
    n_S_per_Sbatch = int(np.maximum(1,n_S_per_Sbatch));
    n_dwSM = n_delta_v*n_w_max*n_S_per_Sbatch*n_M_per_Mbatch;
    n_dwSM_GB = n_dwSM*n_byte_per_float32/1e9;
    if (n_dwSM_GB> memory_limit_GB): disp(sprintf(' %% Warning, n_dwSM_GB %0.2f > %0.2f in %s',n_dwSM_GB,memory_limit_GB,str_thisfunction)); #end;
    n_Mbatch = int(np.ceil(n_M/np.maximum(1,n_M_per_Mbatch)));
    if (flag_verbose>1): disp(sprintf(' %% n_Mbatch %d',n_Mbatch)); #end;
    n_Sbatch = int(np.ceil(n_S/np.maximum(1,n_S_per_Sbatch)));
    if (flag_verbose>1): disp(sprintf(' %% n_Sbatch %d',n_Sbatch)); #end;

    UX_CTF_S_k_q_nSw___ = torch.permute(torch.reshape(UX_CTF_S_k_q_wnS__,mtr((pm_n_w_max,pm_n_k_p_r,n_S))),mtr(mts((1,2,0))));

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;
    for nMbatch in range(n_Mbatch):
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;
        index_nM_in_Mbatch_ = int(nMbatch*n_M_per_Mbatch) + torch.arange(0,n_M_per_Mbatch).to(dtype=torch.int32);
        index_nM_in_Mbatch_ = index_nM_in_Mbatch_[efind(index_nM_in_Mbatch_<n_M)]; n_M_sub = numel(index_nM_in_Mbatch_);
        if (flag_verbose>1): disp(sprintf(' %% nMbatch %d/%d index_nM_in_Mbatch_ %d-->%d',nMbatch,n_Mbatch,index_nM_in_Mbatch_[0].item(),index_nM_in_Mbatch_[n_M_sub-1].item())); #end;
        if (flag_verbose>0 and np.mod(nMbatch,1)==0): disp(sprintf(' %% nMbatch %d/%d index_nM_in_Mbatch_ %d-->%d',nMbatch,n_Mbatch,index_nM_in_Mbatch_[0].item(),index_nM_in_Mbatch_[n_M_sub-1].item())); #end;
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;
        if (n_M_sub>0):
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;
            tmp_t = tic();
            tmp_i8_index_rhs_ = matlab_index_2d_0(n_w_sum,':',n_M,index_nM_in_Mbatch_);
            M_sub_k_p_wkM__ = torch.reshape(M_k_p_wkM__.ravel()[tmp_i8_index_rhs_],mtr((n_w_sum,n_M_sub)));
            M_sub_k_q_wkM__ = torch.reshape(M_k_q_wkM__.ravel()[tmp_i8_index_rhs_],mtr((n_w_sum,n_M_sub))).to(dtype=torch.complex64);
            tmp_t = toc(tmp_t); 
            if (flag_verbose>1): disp(sprintf(' %% M_sub_k_q_wkM__: %0.2fs',tmp_t)); #end;
            parameter = parameter_timing_update(parameter,sprintf('%s: M_sub_k_q_wkM__',str_thisfunction),tmp_t);
            UX_M_sub_l2_M_ = torch.zeros(n_M_sub).to(dtype=torch.float32);
            UX_T_M_sub_l2_dM__ = torch.zeros(mtr((n_delta_v,n_M_sub))).to(dtype=torch.float32);
            #%%%%;
            #% Prepare quasi-images. ;
            #%%%%;
            if flag_precompute_svd_V_UX_M_lwnM____==0:
                if FTK['flag_tf_vs_bf']==1:
                    tmp_t = tic();
                    svd_V_UX_M_sub_lwnM____ = tpmh_VUXM_lwnM____3(FTK,n_k_p_r,n_w_,n_M_sub,M_sub_k_q_wkM__,pm_n_UX_rank,pm_UX_kn__,pm_X_weight_r_).to(dtype=torch.complex64);
                    tmp_t = toc(tmp_t);
                    if (flag_verbose>1): disp(sprintf(' %% svd_V_UX_M_sub_lwnM____: %0.6fs',tmp_t)); #end;
                    parameter = parameter_timing_update(parameter,sprintf('%s: svd_V_UX_M_sub_lwnM____',str_thisfunction),tmp_t);
                #end;%if FTK['flag_tf_vs_bf']==1;
                if FTK['flag_tf_vs_bf']==0:
                    tmp_t = tic();
                    svd_V_UX_M_sub_lwnM____ = tpmh_UXTM_dwnM____0(FTK,n_k_p_r,k_p_r_,n_w_,n_M_sub,M_sub_k_p_wkM__,pm_n_UX_rank,pm_UX_kn__,pm_X_weight_r_).to(dtype=torch.complex64);
                    tmp_t = toc(tmp_t);
                    if (flag_verbose>1): disp(sprintf(' %% UX_T_M_k_q_sub_dwnM____: %0.6fs',tmp_t)); #end;
                    parameter = parameter_timing_update(parameter,sprintf('%s: UX_T_M_k_q_sub_dwnM____',str_thisfunction),tmp_t);
                #end;%if FTK['flag_tf_vs_bf']==0;
            #end;%if flag_precompute_svd_V_UX_M_lwnM____==0;
            if flag_precompute_svd_V_UX_M_lwnM____==1:
                if FTK['flag_tf_vs_bf']==1:
                    tmp_t = tic();
                    tmp_i8_index_rhs_ = matlab_index_4d_0(n_svd_l,':',n_w_max,':',size(svd_V_UX_M_lwnM____,2),torch.arange(pm_n_UX_rank),n_M,index_nM_in_Mbatch_);
                    svd_V_UX_M_sub_lwnM____ = torch.reshape(svd_V_UX_M_lwnM____.ravel()[tmp_i8_index_rhs_],mtr((n_svd_l,n_w_max,pm_n_UX_rank,n_M_sub))).to(dtype=torch.complex64);
                    tmp_t = toc(tmp_t);
                    if (flag_verbose>1): disp(sprintf(' %% loading svd_V_UX_M_sub_lwnM____: %0.6fs',tmp_t)); #end;
                    parameter = parameter_timing_update(parameter,sprintf('%s: loading svd_V_UX_M_sub_lwnM____',str_thisfunction),tmp_t);
                #end;%if FTK['flag_tf_vs_bf']==1;
                if FTK['flag_tf_vs_bf']==0:
                    tmp_t = tic();
                    tmp_i8_index_rhs_ = matlab_index_4d_0(n_delta_v,':',n_w_max,':',size(svd_V_UX_M_lwnM____,2),torch.arange(pm_n_UX_rank),n_M,index_nM_in_Mbatch_);
                    svd_V_UX_M_sub_lwnM____ = torch.reshape(svd_V_UX_M_lwnM____.ravel()[tmp_i8_index_rhs_],mtr((n_delta_v,n_w_max,pm_n_UX_rank,n_M_sub))).to(dtype=torch.complex64);
                    tmp_t = toc(tmp_t);
                    if (flag_verbose>1): disp(sprintf(' %% loading UX_T_M_k_q_sub_dwnM____: %0.6fs',tmp_t)); #end;
                    parameter = parameter_timing_update(parameter,sprintf('%s: loading UX_T_M_k_q_sub_dwnM____',str_thisfunction),tmp_t);
                #end;%if FTK['flag_tf_vs_bf']==0;
            #end;%if flag_precompute_svd_V_UX_M_lwnM____==1;
            tmp_t = tic();
            svd_V_UX_M_sub_nMwl____ = torch.permute(svd_V_UX_M_sub_lwnM____.to(dtype=torch.complex64),mtr(mts((2,3,1,0)))).to(dtype=torch.complex64);
            tmp_t = toc(tmp_t);
            if (flag_verbose>1): disp(sprintf(' %% svd_V_UX_M_sub_nMwl____: %0.6fs',tmp_t)); #end;
            parameter = parameter_timing_update(parameter,sprintf('%s: svd_V_UX_M_sub_nMwl____',str_thisfunction),tmp_t);
            #%%%%;
            #% Now calculate norms of the translated images. ;
            #%%%%;
            if flag_precompute_UX_T_M_l2_dM__==0:
                if FTK['flag_tf_vs_bf']==1:
                    tmp_t = tic();
                    UX_T_M_sub_l2_dM__ = tfpmh_UX_T_M_l2_dM__1(FTK,n_w_,n_M_sub,pm_n_UX_rank,svd_V_UX_M_sub_lwnM____).to(dtype=torch.float32);
                    tmp_t = toc(tmp_t);
                    if (flag_verbose>1): disp(sprintf(' %% tfpmh_UX_T_M_sub_l2_dM__1: %0.6fs',tmp_t)); #end;
                    parameter = parameter_timing_update(parameter,sprintf('%s: tfpmh_UX_T_M_sub_l2_dM__1',str_thisfunction),tmp_t);
                #end;%if FTK['flag_tf_vs_bf']==1;
                if FTK['flag_tf_vs_bf']==0:
                    tmp_t = tic();
                    UX_T_M_sub_l2_dM__ = tfpmh_UX_T_M_l2_dM__0(FTK,n_w_,n_M_sub,pm_n_UX_rank,svd_V_UX_M_sub_lwnM____).to(dtype=torch.float32);
                    tmp_t = toc(tmp_t);
                    if (flag_verbose>1): disp(sprintf(' %% tfpmh_UX_T_M_sub_l2_dM__0: %0.6fs',tmp_t)); #end;
                    parameter = parameter_timing_update(parameter,sprintf('%s: tfpmh_UX_T_M_sub_l2_dM__0',str_thisfunction),tmp_t);
                #end;%if FTK['flag_tf_vs_bf']==0;
                tmp_i8_index_lhs_ = matlab_index_2d_0(n_delta_v,':',n_M,index_nM_in_Mbatch_);
                UX_T_M_l2_dM__.ravel()[tmp_i8_index_lhs_] = UX_T_M_sub_l2_dM__.ravel(); #%<-- store results. ;
            #end;%if flag_precompute_UX_T_M_l2_dM__==0;
            if flag_precompute_UX_T_M_l2_dM__==1:
                tmp_i8_index_rhs_ = matlab_index_2d_0(n_delta_v,':',n_M,index_nM_in_Mbatch_);
                UX_T_M_sub_l2_dM__ = torch.reshape(UX_T_M_l2_dM__.ravel()[tmp_i8_index_rhs_].to(dtype=torch.float32),mtr((n_delta_v,n_M_sub)));
            #end;%if flag_precompute_UX_T_M_l2_dM__==1;
            #%%%%;
            if flag_precompute_UX_M_l2_M_==0:
                tmp_i8_index_rhs_ = matlab_index_2d_0(n_delta_v,tmp_index_d0,n_M_sub,':');
                UX_M_sub_l2_M_ = UX_T_M_sub_l2_dM__.ravel()[tmp_i8_index_rhs_].ravel().to(dtype=torch.float32); assert(numel(UX_M_sub_l2_M_)==n_M_sub);
                UX_M_l2_M_[index_nM_in_Mbatch_] = UX_M_sub_l2_M_; #%<-- store results. ;
            #end;%if flag_precompute_UX_M_l2_M_==0;
            if flag_precompute_UX_M_l2_M_==1:
                UX_M_sub_l2_M_ = UX_M_l2_M_.ravel()[index_nM_in_Mbatch_].to(dtype=torch.float32);
            #end;%if flag_precompute_UX_M_l2_M_==1;
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;
            for nSbatch in range(n_Sbatch):
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;
                index_nS_in_Sbatch_ = int(nSbatch*n_S_per_Sbatch) + torch.arange(n_S_per_Sbatch).to(dtype=torch.int32);
                index_nS_in_Sbatch_ = index_nS_in_Sbatch_[efind(index_nS_in_Sbatch_<n_S)]; n_S_sub = numel(index_nS_in_Sbatch_);
                if (flag_verbose>2): disp(sprintf(' %% nSbatch %d/%d index_nS_in_Sbatch_ %d-->%d',nSbatch,n_Sbatch,index_nS_in_Sbatch_[0].item(),index_nS_in_Sbatch_[n_S_sub-1].item())); #end;
                if (flag_verbose>1 and np.mod(nSbatch,32)==0): disp(sprintf(' %% nSbatch %d/%d index_nS_in_Sbatch_ %d-->%d',nSbatch,n_Sbatch,index_nS_in_Sbatch_[0].item(),index_nS_in_Sbatch_[n_S_sub-1].item())); #end;
                #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;
                if (n_S_sub>0):
                #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;
                    #%%%%;
                    #% extract compressed templates and template norms. ;
                    #%%%%;
                    UX_CTF_S_sub_l2_S_ = UX_CTF_S_l2_S_.ravel()[index_nS_in_Sbatch_].to(dtype=torch.float32);
                    tmp_i8_index_rhs_ = matlab_index_3d_0(pm_n_UX_rank,':',n_S,index_nS_in_Sbatch_,n_w_max,':');
                    UX_CTF_S_sub_k_q_nSw___ = torch.reshape(UX_CTF_S_k_q_nSw___.ravel()[tmp_i8_index_rhs_].to(dtype=torch.complex64),mtr((pm_n_UX_rank,n_S_sub,n_w_max)));
                    #%%%%;
                    #% Calculate innerproduct Z_sub_dwSM____. ;
                    #%%%%;
                    tmp_t = tic();
                    str_einsum = msr('nSw') + ',' + msr('nMwl') + '->' + msr('SMwl') ;
                    svd_CTF_S_sub_V_UX_M_sub_SMwl____ = torch.einsum(str_einsum,torch.conj(UX_CTF_S_sub_k_q_nSw___).to(dtype=torch.complex64),svd_V_UX_M_sub_nMwl____.to(dtype=torch.complex64)).to(dtype=torch.complex64);
                    tmp_t = toc(tmp_t); 
                    if (flag_verbose>1): disp(sprintf(' %% svd_CTF_S_sub_V_UX_M_sub_SMwl____: %0.6fs',tmp_t)); #end;
                    parameter = parameter_timing_update(parameter,sprintf('%s: svd_CTF_S_sub_V_UX_M_sub_SMwl____',str_thisfunction),tmp_t);
                    tmp_t = tic();
                    svd_CTF_S_sub_V_UX_M_sub_lwSM____ = torch.fft.ifft(torch.permute(svd_CTF_S_sub_V_UX_M_sub_SMwl____.to(dtype=torch.complex64),mtr(mts((3,2,0,1)))),dim=3-1).to(dtype=torch.complex64);
                    tmp_t = toc(tmp_t); 
                    if (flag_verbose>1): disp(sprintf(' %% svd_CTF_S_sub_V_UX_M_sub_lwSM____: %0.6fs',tmp_t)); #end;
                    parameter = parameter_timing_update(parameter,sprintf('%s: svd_CTF_S_sub_V_UX_M_sub_lwSM____',str_thisfunction),tmp_t);
                    if FTK['flag_tf_vs_bf']==1:
                        tmp_t = tic();
                        svd_UES_CTF_S_sub_V_UX_M_sub_dwSM____ = torch.reshape( mmmm( torch.reshape(FTK['c16_svd_U_d_expiw_s__'].to(dtype=torch.complex64),mtr((FTK['n_delta_v'],FTK['n_svd_l']))) , torch.reshape(svd_CTF_S_sub_V_UX_M_sub_lwSM____.to(dtype=torch.complex64)*n_w_max,mtr((FTK['n_svd_l'],n_w_max*n_S_sub*n_M_sub))) ),mtr((FTK['n_delta_v'],n_w_max,n_S_sub,n_M_sub))).to(dtype=torch.complex64);
                        tmp_t = toc(tmp_t); 
                        if (flag_verbose>1): disp(sprintf(' %% svd_UES_CTF_S_sub_V_UX_M_sub_dwSM____: %0.6fs',tmp_t)); #end;
                        parameter = parameter_timing_update(parameter,sprintf('%s: svd_UES_CTF_S_sub_V_UX_M_sub_dwSM____',str_thisfunction),tmp_t);
                    #end;%if FTK['flag_tf_vs_bf']==1;
                    if FTK['flag_tf_vs_bf']==0:
                        svd_UES_CTF_S_sub_V_UX_M_sub_dwSM____ = svd_CTF_S_sub_V_UX_M_sub_lwSM____; #%<-- actually CTF_S_sub_UX_T_M_sub_dwSM____. ;
                    #end;%if FTK['flag_tf_vs_bf']==0;
                    Z_sub_dwSM____  = torch.real(svd_UES_CTF_S_sub_V_UX_M_sub_dwSM____).to(dtype=torch.float32);
                    #%%%%;
                    #% Calculate correlation. ;
                    #%%%%;
                    tmp_t = tic();
                    X_sub_dwSM____ = Z_sub_dwSM____ / torch.maximum(torch.tensor(tolerance_machine).to(dtype=torch.float32),torch.reshape(torch.sqrt(UX_CTF_S_sub_l2_S_),mtr((1,1,n_S_sub,1)))) / torch.maximum(torch.tensor(tolerance_machine).to(dtype=torch.float32),torch.reshape(torch.sqrt(UX_T_M_sub_l2_dM__),mtr((n_delta_v,1,1,n_M_sub)))) ;
                    tmp_t = toc(tmp_t); 
                    if (flag_verbose>1): disp(sprintf(' %% X_sub_dwSM____: %0.6fs',tmp_t)); #end;
                    parameter = parameter_timing_update(parameter,sprintf('%s: X_sub_dwSM____',str_thisfunction),tmp_t);
                    #%%%%;
                    #% Store results. ;
                    #%%%%;
                    if flag_dwSM:
                        tmp_i8_index_lhs_ = matlab_index_4d_0(n_delta_v,':',n_w_max,':',n_S,index_nS_in_Sbatch_,n_M,index_nM_in_Mbatch_);
                        Z_dwSM____.ravel()[tmp_i8_index_lhs_] = Z_sub_dwSM____.ravel();
                        X_dwSM____.ravel()[tmp_i8_index_lhs_] = X_sub_dwSM____.ravel();
                    #%end;%if flag_dwSM;
                    #%%%%%%%%;
                    if flag_optimize_over_gamma_z==0:
                        tmp_t = tic();
                        n_wSM = n_w_max*n_S_sub*n_M_sub;
                        tmp_X_wSM_,tmp_index_delta_wSM_ = torch.max(torch.reshape(X_sub_dwSM____,mtr((n_delta_v,n_wSM))),dim=1-0); #%<-- maximize correlation. ;
                        tmp_X_wSM_ = tmp_X_wSM_; tmp_index_delta_wSM_ = tmp_index_delta_wSM_;
                        assert(torch.min(tmp_index_delta_wSM_.ravel()).item()>=0);
                        assert(torch.max(tmp_index_delta_wSM_.ravel()).item()<=n_delta_v-1);
                        tmp_X_wSM___ = torch.reshape(tmp_X_wSM_,mtr((n_w_max,n_S_sub,n_M_sub)));
                        tmp_i8_index_all_wSM_ = tmp_index_delta_wSM_.to(dtype=torch.int64).ravel() + torch.arange(n_wSM).ravel().to(dtype=torch.int64)*int(n_delta_v);
                        assert(torch.min(tmp_i8_index_all_wSM_.ravel()).item()>=0);
                        assert(torch.max(tmp_i8_index_all_wSM_.ravel()).item()<=n_delta_v*n_wSM-1);
                        tmp_Z_wSM___ = torch.reshape(Z_sub_dwSM____.ravel()[tmp_i8_index_all_wSM_],mtr((n_w_max,n_S_sub,n_M_sub)));
                        tmp_index_delta_wSM___ = torch.reshape(tmp_index_delta_wSM_,mtr((n_w_max,n_S_sub,n_M_sub)));
                        tmp_delta_x_wSM___ = FTK['r8_delta_x_'].to(dtype=torch.float32).ravel()[tmp_index_delta_wSM___];
                        tmp_delta_y_wSM___ = FTK['r8_delta_y_'].to(dtype=torch.float32).ravel()[tmp_index_delta_wSM___];
                        tmp_gamma_z_wSM___ = 2*pi*torch.arange(n_w_max).to(dtype=torch.float32)/np.maximum(1,n_w_max);
                        tmp_i8_index_lhs_ = matlab_index_3d_0(n_w_max,':',n_S,index_nS_in_Sbatch_,n_M,index_nM_in_Mbatch_);
                        Z_wSM___.ravel()[tmp_i8_index_lhs_] = tmp_Z_wSM___.ravel();
                        X_wSM___.ravel()[tmp_i8_index_lhs_] = tmp_X_wSM___.ravel();
                        delta_x_wSM___.ravel()[tmp_i8_index_lhs_] = torch.reshape(tmp_delta_x_wSM___,mtr((n_w_max,n_S_sub,n_M_sub))).ravel();
                        delta_y_wSM___.ravel()[tmp_i8_index_lhs_] = torch.reshape(tmp_delta_y_wSM___,mtr((n_w_max,n_S_sub,n_M_sub))).ravel();
                        gamma_z_wSM___.ravel()[tmp_i8_index_lhs_] = (torch.reshape(tmp_gamma_z_wSM___,mtr((n_w_max,1,1)))*torch.ones(mtr((1,n_S_sub,n_M_sub)))).to(dtype=torch.float32).ravel();
                        index_sub_wSM___.ravel()[tmp_i8_index_lhs_] = torch.reshape(tmp_index_delta_wSM_.to(dtype=torch.int32),mtr((n_w_max,n_S_sub,n_M_sub))).ravel();
                        UX_CTF_S_l2_S_[index_nS_in_Sbatch_] = UX_CTF_S_sub_l2_S_.ravel();
                        tmp_t = toc(tmp_t); 
                        if (flag_verbose>1): disp(sprintf(' %% X_wSM___: %0.6f',tmp_t)); #end;
                        parameter = parameter_timing_update(parameter,sprintf('%s: X_wSM___',str_thisfunction),tmp_t);
                    #end;%if flag_optimize_over_gamma_z==0;
                    #%%%%%%%%;
                    if flag_optimize_over_gamma_z==1:
                        tmp_t = tic();
                        n_dw = n_delta_v*n_w_max; n_SM = n_S_sub*n_M_sub;
                        tmp_X_SM_,tmp_index_dw_SM_ = torch.max(torch.reshape(X_sub_dwSM____,mtr((n_dw,n_SM))),dim=1-0); #%<-- maximize correlation. ;
                        tmp_X_SM_ = tmp_X_SM_; tmp_index_dw_SM_ = tmp_index_dw_SM_;
                        assert(torch.min(tmp_index_dw_SM_.ravel()).item()>=0);
                        assert(torch.max(tmp_index_dw_SM_.ravel()).item()<=n_dw-1);
                        tmp_X_SM__ = torch.reshape(tmp_X_SM_,mtr((n_S_sub,n_M_sub)));
                        tmp_i8_index_all_SM_ = tmp_index_dw_SM_.to(dtype=torch.int64).ravel() + torch.arange(n_SM).ravel().to(dtype=torch.int64)*int(n_dw);
                        assert(torch.min(tmp_i8_index_all_SM_.ravel()).item()>=0);
                        assert(torch.max(tmp_i8_index_all_SM_.ravel()).item()<=n_dw*n_SM-1);
                        tmp_Z_SM__ = torch.reshape(Z_sub_dwSM____.ravel()[tmp_i8_index_all_SM_],mtr((n_S_sub,n_M_sub)));
                        tmp_index_dw_SM__ = torch.reshape(tmp_index_dw_SM_,mtr((n_S_sub,n_M_sub)));
                        tmp_index_delta_SM__ = torch.fmod(tmp_index_dw_SM__,n_delta_v).to(dtype=torch.int32);
                        tmp_index_gamma_SM__ = torch.div(tmp_index_dw_SM__ - tmp_index_delta_SM__,torch.tensor(np.maximum(1,n_delta_v)).to(dtype=torch.int32),rounding_mode='floor').to(dtype=torch.int32);
                        assert(torch.min(tmp_index_delta_SM__.ravel()).item()>=0); 
                        assert(torch.max(tmp_index_delta_SM__.ravel()).item()<=n_delta_v-1);
                        assert(torch.min(tmp_index_gamma_SM__.ravel()).item()>=0);
                        assert(torch.max(tmp_index_gamma_SM__.ravel()).item()<=n_w_max-1);
                        tmp_index_delta_SM__ = torch.reshape(tmp_index_delta_SM__,mtr((n_S_sub,n_M_sub)));
                        tmp_index_gamma_SM__ = torch.reshape(tmp_index_gamma_SM__,mtr((n_S_sub,n_M_sub)));
                        tmp_delta_x_SM__ = FTK['r8_delta_x_'].to(dtype=torch.float32).ravel()[tmp_index_delta_SM__];
                        tmp_delta_y_SM__ = FTK['r8_delta_y_'].to(dtype=torch.float32).ravel()[tmp_index_delta_SM__];
                        tmp_gamma_z_SM__ = 2*pi*tmp_index_gamma_SM__.to(dtype=torch.float32)/np.maximum(1,n_w_max);
                        tmp_i8_index_lhs_ = matlab_index_2d_0(n_S,index_nS_in_Sbatch_,n_M,index_nM_in_Mbatch_);
                        Z_SM__.ravel()[tmp_i8_index_lhs_] = tmp_Z_SM__.ravel();
                        X_SM__.ravel()[tmp_i8_index_lhs_] = tmp_X_SM__.ravel();
                        delta_x_SM__.ravel()[tmp_i8_index_lhs_] = torch.reshape(tmp_delta_x_SM__,mtr((n_S_sub,n_M_sub))).ravel();
                        delta_y_SM__.ravel()[tmp_i8_index_lhs_] = torch.reshape(tmp_delta_y_SM__,mtr((n_S_sub,n_M_sub))).ravel();
                        gamma_z_SM__.ravel()[tmp_i8_index_lhs_] = torch.reshape(tmp_gamma_z_SM__,mtr((n_S_sub,n_M_sub))).ravel();
                        index_sub_SM__.ravel()[tmp_i8_index_lhs_] = torch.reshape(tmp_index_dw_SM_.to(dtype=torch.int32),mtr((n_S_sub,n_M_sub))).ravel();
                        UX_CTF_S_l2_S_[index_nS_in_Sbatch_] = UX_CTF_S_sub_l2_S_.ravel();
                        tmp_t = toc(tmp_t); 
                        if (flag_verbose>1): disp(sprintf(' %% X_SM__: %0.6f',tmp_t)); #end;
                        parameter = parameter_timing_update(parameter,sprintf('%s: X_SM__',str_thisfunction),tmp_t);
                    #end;%if flag_optimize_over_gamma_z==1;
                #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;
                #end;%if (n_S_sub>0);
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;
            #end;%for nSbatch=0:n_Sbatch-1;
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;
        #end;%if (n_M_sub>0);
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;
    #end;%for nMbatch=0:n_Mbatch-1;
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;
    
    if flag_optimize_over_gamma_z==1:
        Z_wSM___ = Z_SM__;
        X_wSM___ = X_SM__;
        delta_x_wSM___ = delta_x_SM__;
        delta_y_wSM___ = delta_y_SM__;
        gamma_z_wSM___ = gamma_z_SM__;
        index_sub_wSM___ = index_sub_SM__;
    #end;%if flag_optimize_over_gamma_z==1;

    #%%%%%%%%;
    if (flag_verbose): disp(sprintf(' %% [finished %s]',str_thisfunction)); #end;
    #%%%%%%%%;
    if flag_output_ravel:
        if not isempty(Z_wSM___): Z_wSM___ = Z_wSM___.ravel(); #end;
        if not isempty(UX_T_M_l2_dM__): UX_T_M_l2_dM__ = UX_T_M_l2_dM__.ravel(); #end;
        if not isempty(UX_M_l2_M_): UX_M_l2_M_ = UX_M_l2_M_.ravel(); #end;
        if not isempty(UX_CTF_S_l2_S_): UX_CTF_S_l2_S_ = UX_CTF_S_l2_S_.ravel(); #end;
        if not isempty(X_wSM___): X_wSM___ = X_wSM___.ravel(); #end;
        if not isempty(delta_x_wSM___): delta_x_wSM___ = delta_x_wSM___.ravel(); #end;
        if not isempty(delta_y_wSM___): delta_y_wSM___ = delta_y_wSM___.ravel(); #end;
        if not isempty(gamma_z_wSM___): gamma_z_wSM___ = gamma_z_wSM___.ravel(); #end;
        if not isempty(index_sub_wSM___): index_sub_wSM___ = index_sub_wSM___.ravel(); #end;
        if not isempty(Z_dwSM____): Z_dwSM____ = Z_dwSM____.ravel(); #end;
        if not isempty(X_dwSM____): X_dwSM____ = X_dwSM____.ravel(); #end;
    #end;%if flag_output_ravel;    
    return(
        parameter,
        Z_wSM___,
        UX_T_M_l2_dM__,
        UX_M_l2_M_,
        UX_CTF_S_l2_S_,
        X_wSM___,
        delta_x_wSM___,
        delta_y_wSM___,
        gamma_z_wSM___,
        index_sub_wSM___,
        Z_dwSM____,
        X_dwSM____,
    );

