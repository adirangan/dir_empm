from dir_empm.dir_matlab_macros import * ;
from dir_empm.tfpmhp_Z_wSM___14 import tfpmhp_Z_wSM___14 ;
from dir_empm.tfpmh_Z_wSM___14 import tfpmh_Z_wSM___14 ;
from dir_empm.tfpmh_Z_gpu_wSM___14 import tfpmh_Z_gpu_wSM___14 ;

def tfpmh_Z_cluster_wrap_SM__14(
        parameter=None,
        n_k_p_r=None,
        k_p_r_=None,
        k_p_r_max=None,
        n_w_=None,
        weight_2d_k_p_r_=None,
        weight_2d_k_p_wk_=None,
        n_S=None,
        S_k_p_wkS__=None,
        n_CTF=None,
        CTF_k_p_r_kC__=None,
        index_nCTF_from_nM_=None,
        n_M=None,
        M_k_p_wkM__=None,
        n_cluster=None,
        index_ncluster_from_nCTF_=None,
        pm_n_UX_rank_c_=None,
        pm_UX_knc___=None,
        pm_X_weight_rc__=None,
        FTK=None,
        index_nM_to_update_=torch.tensor([]).to(dtype=torch.int32),
        M_k_q_wkM__=None,
        UX_T_M_l2_dM__=None,
        UX_M_l2_M_=None,
        svd_V_UX_M_lwnM____=None, #%<-- or UX_T_M_k_q_dwnM____ ;
        index_nS_to_update_=torch.tensor([]).to(dtype=torch.int32),
        UX_CTF_S_k_q_wnS__=None,
        UX_CTF_S_l2_S_=None,
):

    #%%%%%%%%;
    #% wrapper for tfpmh_Z_wSM___14. ;
    #% Note that the radial compression is applied here, and is structured to align with the canonical innerproduct. ;
    #% (i.e., can use radial compression even when the loss is high). ;
    #% Assumes n_w_ = n_w_max*ones(n_k_p_r,1);
    #% We also allow for certain arrays to be passed in as precomputations. ;
    #%%%%%%%%;

    str_thisfunction = 'tfpmh_Z_cluster_wrap_SM__14' ;

    if not isinstance(parameter, dict): parameter = {'type': 'parameter'};
    if 'flag_verbose' not in parameter: parameter['flag_verbose'] = 0 ;
    flag_verbose = parameter['flag_verbose'];
    if 'flag_gpu' not in parameter: parameter['flag_gpu'] = 0 ;
    flag_gpu = parameter['flag_gpu'];
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

    #%%%%%%%%;
    if (flag_verbose): print(f' %% [entering {str_thisfunction}]');
    #%%%%%%%%;
    if isempty(index_nM_to_update_): index_nM_to_update_=torch.tensor([]).to(dtype=torch.int32); #end;
    if isempty(index_nS_to_update_): index_nS_to_update_=torch.tensor([]).to(dtype=torch.int32); #end;

    tolerance_machine = 1e-6;

    n_w_ = n_w_.ravel();
    n_w_max = int(torch.max(n_w_).item());
    n_w_sum = int(torch.sum(n_w_).item());
    n_w_csum_ = cumsum_0(n_w_);
    if (n_w_sum!=n_w_max*n_k_p_r): disp(sprintf(' %% Warning, n_w_sum %d ~= n_w_max*n_k_p_r %d*%d in %s',n_w_sum,n_w_max,n_k_p_r,str_thisfunction)); #end;
    if np.mod(n_w_max,2)!=0:  disp(sprintf(' %% Warning, n_w_max %d in %s',n_w_max,str_thisfunction)); #end;
    n_delta_v = FTK['n_delta_v'];
    n_svd_l = FTK['n_svd_l'];

    #%%%%%%%%;
    #% group images by cluster. ;
    #%%%%%%%%;
    n_cluster = 1+int(torch.max(index_ncluster_from_nCTF_).item());
    index_ncluster_from_nM_ = index_ncluster_from_nCTF_[index_nCTF_from_nM_];
    index_nM_from_ncluster__ = cell(n_cluster);
    n_index_nM_from_ncluster_ = torch.zeros(n_cluster).to(dtype=torch.int32);
    for ncluster in range(n_cluster):
        index_nM_from_ncluster__[ncluster] = efind(index_ncluster_from_nM_==ncluster);
        n_index_nM_from_ncluster_[ncluster] = numel(index_nM_from_ncluster__[ncluster]);
    #end;%for ncluster=0:n_cluster-1;
    #%%%%%%%%;
    pm_n_UX_rank_max = int(torch.max(pm_n_UX_rank_c_).item());
    pm_n_w_sum_max = n_w_max*pm_n_UX_rank_max;
    #%%%%%%%%;
    if isempty(M_k_q_wkM__): M_k_q_wkM__=torch.zeros(mtr((n_w_sum,n_M))).to(dtype=torch.complex64); #end;
    if isempty(UX_T_M_l2_dM__): UX_T_M_l2_dM__=torch.zeros(mtr((n_delta_v,n_M))).to(dtype=torch.float32); #end;
    if isempty(UX_M_l2_M_): UX_M_l2_M_=torch.zeros(n_M).to(dtype=torch.float32); #end;
    n_svd_l_vs_n_delta_v = n_svd_l;
    if FTK['flag_tf_vs_bf']==1: n_svd_l_vs_n_delta_v = n_svd_l; #end;
    if FTK['flag_tf_vs_bf']==0: n_svd_l_vs_n_delta_v = n_delta_v; #end;
    if isempty(svd_V_UX_M_lwnM____):
        svd_V_UX_M_lwnM____=torch.zeros(mtr((n_svd_l_vs_n_delta_v,n_w_max,pm_n_UX_rank_max,n_M))).to(dtype=torch.complex64); 
    #end;%if isempty(svd_V_UX_M_lwnM____);
    if isempty(UX_CTF_S_k_q_wnS__): UX_CTF_S_k_q_wnS__=torch.zeros(mtr((pm_n_w_sum_max,n_S))).to(dtype=torch.complex64); #end;
    if isempty(UX_CTF_S_l2_S_): UX_CTF_S_l2_S_=torch.zeros(n_S).to(dtype=torch.float32); #end;

    if 'Z_SM__' not in locals(): Z_SM__ = torch.zeros(mtr((n_S,n_M))).to(dtype=torch.float32); #end;
    if 'UX_CTF_S_l2_SM__' not in locals(): UX_CTF_S_l2_SM__ = torch.zeros(mtr((n_S,n_M))).to(dtype=torch.float32); #end;
    if 'UX_T_M_l2_SM__' not in locals(): UX_T_M_l2_SM__ = torch.zeros(mtr((n_S,n_M))).to(dtype=torch.float32); #end;
    if 'X_SM__' not in locals(): X_SM__ = torch.zeros(mtr((n_S,n_M))).to(dtype=torch.float32); #end;
    if 'delta_x_SM__' not in locals(): delta_x_SM__ = torch.zeros(mtr((n_S,n_M))).to(dtype=torch.float32); #end;
    if 'delta_y_SM__' not in locals(): delta_y_SM__ = torch.zeros(mtr((n_S,n_M))).to(dtype=torch.float32); #end;
    if 'gamma_z_SM__' not in locals(): gamma_z_SM__ = torch.zeros(mtr((n_S,n_M))).to(dtype=torch.float32); #end;
    if 'index_sub_SM__' not in locals(): index_sub_SM__ = torch.zeros(mtr((n_S,n_M))).to(dtype=torch.int32); #end;

    #%%%%%%%%;
    #% enforce CTF isotropy. ;
    #%%%%%%%%;
    if (size(CTF_k_p_r_kC__,0)==n_w_sum):
        CTF_k_p_wkC__ = CTF_k_p_r_kC__;
        CTF_k_p_r_kC__ = torch.reshape(torch.mean(torch.reshape(CTF_k_p_wkC__,mtr((n_w_max,n_k_p_r,n_CTF))),dim=2-0),mtr((n_k_p_r,n_CTF)));
    #end;%if (size(CTF_k_p_r_kC__,1)==n_w_sum);
    tmp_i8_index_rhs_ = matlab_index_2d_0(n_k_p_r,':',n_CTF,index_nCTF_from_nM_);
    CTF_k_p_r_kM__ = torch.reshape(CTF_k_p_r_kC__.ravel()[tmp_i8_index_rhs_],mtr((n_k_p_r,n_M)));

    for ncluster in range(n_cluster):
        index_nM_from_ncluster_ = index_nM_from_ncluster__[ncluster];
        n_M_from_ncluster = numel(index_nM_from_ncluster_); assert(n_M_from_ncluster==int(n_index_nM_from_ncluster_[ncluster].item()));
        if (n_M_from_ncluster>0):
            #%%%%%%%%;
            tmp_t=tic();
            tmp_i8_index_rhs_ = matlab_index_2d_0(n_k_p_r,':',n_M,index_nM_from_ncluster_);
            CTF_k_p_r_xavg_k_ = torch.mean(torch.reshape(CTF_k_p_r_kM__.ravel()[tmp_i8_index_rhs_],mtr((n_k_p_r,n_M_from_ncluster))),dim=1-1).ravel(); assert(numel(CTF_k_p_r_xavg_k_)==n_k_p_r);
            pm_n_UX_rank = int(pm_n_UX_rank_c_[ncluster].item()); pm_n_w_sum = n_w_max*pm_n_UX_rank;
            tmp_i8_index_rhs_ = matlab_index_3d_0(n_k_p_r,':',size(pm_UX_knc___,1),torch.arange(pm_n_UX_rank),n_cluster,ncluster);
            pm_UX_kn__ = torch.reshape(pm_UX_knc___.ravel()[tmp_i8_index_rhs_],mtr((n_k_p_r,pm_n_UX_rank)));
            tmp_i8_index_rhs_ = matlab_index_2d_0(n_k_p_r,':',n_cluster,ncluster);
            pm_X_weight_r_ = pm_X_weight_rc__.ravel()[tmp_i8_index_rhs_]; assert(numel(pm_X_weight_r_)==n_k_p_r);
            #%%%%%%%%;
            #% precomputation. ;
            #%%%%%%%%;
            nM_cap_,index_nM_to_update_from_nM_cap_,index_nM_from_ncluster_from_nM_cap_ = intersect_0(index_nM_to_update_,index_nM_from_ncluster_);
            if (flag_verbose>0): disp(sprintf(' %% ncluster %d/%d: n_M_from_ncluster %d %%<-- precomputation for %d images, %d templates',ncluster,n_cluster,n_M_from_ncluster,numel(nM_cap_),numel(index_nS_to_update_))); #end;
            tmp_t=tic();
            tmp_i8_index_lhs_wkM_ = matlab_index_2d_0(n_w_sum,':',n_M,index_nM_from_ncluster_);
            tmp_i8_index_lhs_dM_ = matlab_index_2d_0(n_delta_v,':',n_M,index_nM_from_ncluster_);
            tmp_i8_index_lhs_M_ = index_nM_from_ncluster_;
            tmp_i8_index_lhs_lwnM_ = matlab_index_4d_0(n_svd_l_vs_n_delta_v,':',n_w_max,':',pm_n_UX_rank_max,torch.arange(pm_n_UX_rank),n_M,index_nM_from_ncluster_);
            tmp_i8_index_lhs_wnS_ = matlab_index_2d_0(pm_n_w_sum_max,torch.arange(pm_n_w_sum),n_S,':');
            tmp_i8_index_lhs_S_ = torch.arange(n_S);
            tmp_i8_index_rhs_wkM_ = matlab_index_2d_0(n_w_sum,':',n_M,index_nM_from_ncluster_);
            tmp_i8_index_rhs_dM_ = matlab_index_2d_0(n_delta_v,':',n_M,index_nM_from_ncluster_);
            tmp_i8_index_rhs_M_ = index_nM_from_ncluster_;
            tmp_i8_index_rhs_lwnM_ = matlab_index_4d_0(n_svd_l_vs_n_delta_v,':',n_w_max,':',pm_n_UX_rank_max,torch.arange(pm_n_UX_rank),n_M,index_nM_from_ncluster_);
            tmp_i8_index_rhs_wnS_ = matlab_index_2d_0(pm_n_w_sum_max,torch.arange(pm_n_w_sum),n_S,':');
            tmp_i8_index_rhs_S_ = torch.arange(n_S);
            parameter['flag_output_ravel'] = 1;
            (
                parameter,
                M_k_q_wkM__.ravel()[tmp_i8_index_lhs_wkM_],
                UX_T_M_l2_dM__.ravel()[tmp_i8_index_lhs_dM_],
                UX_M_l2_M_.ravel()[tmp_i8_index_lhs_M_],
                svd_V_UX_M_lwnM____.ravel()[tmp_i8_index_lhs_lwnM_],
                UX_CTF_S_k_q_wnS__.ravel()[tmp_i8_index_lhs_wnS_],
                UX_CTF_S_l2_S_.ravel()[tmp_i8_index_lhs_S_],
            ) = tfpmhp_Z_wSM___14(
                parameter,
                n_k_p_r,
                k_p_r_,
                k_p_r_max,
                n_w_,
                weight_2d_k_p_r_,
                weight_2d_k_p_wk_,
                n_M_from_ncluster,
                torch.reshape(M_k_p_wkM__.ravel()[tmp_i8_index_rhs_wkM_],mtr((n_w_sum,n_M_from_ncluster))),
                CTF_k_p_r_xavg_k_,
                n_S,
                S_k_p_wkS__,
                pm_n_UX_rank,
                pm_UX_kn__,
                pm_X_weight_r_,
                FTK,
                index_nM_from_ncluster_from_nM_cap_,
                torch.reshape(M_k_q_wkM__.ravel()[tmp_i8_index_rhs_wkM_],mtr((n_w_sum,n_M_from_ncluster))),
                torch.reshape(UX_T_M_l2_dM__.ravel()[tmp_i8_index_rhs_dM_],mtr((n_delta_v,n_M_from_ncluster))),
                UX_M_l2_M_.ravel()[tmp_i8_index_rhs_M_],
                torch.reshape(svd_V_UX_M_lwnM____.ravel()[tmp_i8_index_rhs_lwnM_],mtr((n_svd_l_vs_n_delta_v,n_w_max,pm_n_UX_rank,n_M_from_ncluster))),
                index_nS_to_update_,
                torch.reshape(UX_CTF_S_k_q_wnS__.ravel()[tmp_i8_index_rhs_wnS_],mtr((pm_n_w_sum,n_S))),
                UX_CTF_S_l2_S_.ravel()[tmp_i8_index_rhs_S_],
            )[:7];
            tmp_t=toc(tmp_t);
            if (flag_verbose>0): disp(sprintf(' %% tfpmhp_Z_wSM___14: %0.3fs',tmp_t)); #end;
            parameter = parameter_timing_update(parameter,sprintf('%s: tfpmhp_Z_wSM___14',str_thisfunction),tmp_t);
            #%%%%%%%%;
            #% inner-product. ;
            #%%%%%%%%;
            parameter['flag_optimize_over_gamma_z']=1;
            parameter['flag_dwSM']=0;
            parameter['flag_output_ravel']=1;
            #%%%%;
            if flag_gpu==0:
                tmp_t=tic();
                parameter['n_M_per_Mbatch'] = 24;
                parameter['n_S_per_Sbatch'] = 24;
                parameter['device_use'] = 'cpu';
                tmp_i8_index_lhs_SM_ = matlab_index_2d_0(n_S,':',n_M,index_nM_from_ncluster_);
                tmp_i8_index_rhs_wkM_ = matlab_index_2d_0(n_w_sum,':',n_M,index_nM_from_ncluster_);
                tmp_i8_index_rhs_dM_ = matlab_index_2d_0(n_delta_v,':',n_M,index_nM_from_ncluster_);
                tmp_i8_index_rhs_M_ = index_nM_from_ncluster_;
                tmp_i8_index_rhs_lwnM_ = matlab_index_4d_0(n_svd_l_vs_n_delta_v,':',n_w_max,':',pm_n_UX_rank_max,torch.arange(pm_n_UX_rank),n_M,index_nM_from_ncluster_);
                tmp_i8_index_rhs_wnS_ = matlab_index_2d_0(pm_n_w_sum_max,torch.arange(pm_n_w_sum),n_S,':');
                tmp_i8_index_rhs_S_ = torch.arange(n_S);
                (
                    parameter,
                    Z_SM__.ravel()[tmp_i8_index_lhs_SM_],
                    UX_T_M_l2_dM__.ravel()[tmp_i8_index_lhs_dM_],
                    UX_M_l2_M_.ravel()[index_nM_from_ncluster_],
                    UX_CTF_S_l2_S_.ravel()[:],
                    X_SM__.ravel()[tmp_i8_index_lhs_SM_],
                    delta_x_SM__.ravel()[tmp_i8_index_lhs_SM_],
                    delta_y_SM__.ravel()[tmp_i8_index_lhs_SM_],
                    gamma_z_SM__.ravel()[tmp_i8_index_lhs_SM_],
                    index_sub_SM__.ravel()[tmp_i8_index_lhs_SM_],
                ) = tfpmh_Z_wSM___14(
                    parameter,
                    n_k_p_r,
                    k_p_r_,
                    k_p_r_max,
                    n_w_,
                    weight_2d_k_p_r_,
                    weight_2d_k_p_wk_,
                    n_M_from_ncluster,
                    torch.reshape(M_k_p_wkM__.ravel()[tmp_i8_index_rhs_wkM_],mtr((n_w_sum,n_M_from_ncluster))),
                    CTF_k_p_r_xavg_k_,
                    n_S,
                    S_k_p_wkS__,
                    pm_n_UX_rank,
                    pm_UX_kn__,
                    pm_X_weight_r_,
                    FTK,
                    torch.reshape(M_k_q_wkM__.ravel()[tmp_i8_index_rhs_wkM_],mtr((n_w_sum,n_M_from_ncluster))),
                    torch.reshape(UX_T_M_l2_dM__.ravel()[tmp_i8_index_rhs_dM_],mtr((n_delta_v,n_M_from_ncluster))),
                    UX_M_l2_M_.ravel()[tmp_i8_index_rhs_M_],
                    torch.reshape(svd_V_UX_M_lwnM____.ravel()[tmp_i8_index_rhs_lwnM_],mtr((n_svd_l_vs_n_delta_v,n_w_max,pm_n_UX_rank,n_M_from_ncluster))),
                    torch.reshape(UX_CTF_S_k_q_wnS__.ravel()[tmp_i8_index_rhs_wnS_],mtr((pm_n_w_sum,n_S))),
                    UX_CTF_S_l2_S_.ravel()[tmp_i8_index_rhs_S_],
                )[:10];
                tmp_t=toc(tmp_t);
                if (flag_verbose>0): disp(sprintf(' %% tfpmh_Z_wSM___14: %0.3fs',tmp_t)); #end;
                parameter = parameter_timing_update(parameter,sprintf('%s: tfpmh_Z_wSM___14',str_thisfunction),tmp_t);
            #end;%if flag_gpu==0;
            #%%%%;
            if flag_gpu==1: 
                tmp_t=tic();
                parameter['n_M_per_Mbatch'] = 24*4;
                parameter['n_S_per_Sbatch'] = 24*4;
                parameter['device_use'] = 'cuda';
                tmp_i8_index_lhs_SM_ = matlab_index_2d_0(n_S,':',n_M,index_nM_from_ncluster_);
                tmp_i8_index_rhs_wkM_ = matlab_index_2d_0(n_w_sum,':',n_M,index_nM_from_ncluster_);
                tmp_i8_index_rhs_dM_ = matlab_index_2d_0(n_delta_v,':',n_M,index_nM_from_ncluster_);
                tmp_i8_index_rhs_M_ = index_nM_from_ncluster_;
                tmp_i8_index_rhs_lwnM_ = matlab_index_4d_0(n_svd_l_vs_n_delta_v,':',n_w_max,':',pm_n_UX_rank_max,torch.arange(pm_n_UX_rank),n_M,index_nM_from_ncluster_);
                tmp_i8_index_rhs_wnS_ = matlab_index_2d_0(pm_n_w_sum_max,torch.arange(pm_n_w_sum),n_S,':');
                tmp_i8_index_rhs_S_ = torch.arange(n_S);
                (
                    parameter,
                    tmp_Z_gpu_SM__,
                    tmp_UX_T_M_l2_gpu_dM__,
                    tmp_UX_M_l2_gpu_M_,
                    tmp_UX_CTF_S_l2_gpu_S_,
                    tmp_X_gpu_SM__,
                    tmp_delta_x_gpu_SM__,
                    tmp_delta_y_gpu_SM__,
                    tmp_gamma_z_gpu_SM__,
                    tmp_index_sub_gpu_SM__,
                ) = tfpmh_Z_gpu_wSM___14(
                    parameter,
                    n_k_p_r,
                    k_p_r_,
                    k_p_r_max,
                    n_w_,
                    weight_2d_k_p_r_,
                    weight_2d_k_p_wk_,
                    n_M_from_ncluster,
                    torch.reshape(M_k_p_wkM__.ravel()[tmp_i8_index_rhs_wkM_],mtr((n_w_sum,n_M_from_ncluster))),
                    CTF_k_p_r_xavg_k_,
                    n_S,
                    S_k_p_wkS__,
                    pm_n_UX_rank,
                    pm_UX_kn__,
                    pm_X_weight_r_,
                    FTK,
                    torch.reshape(M_k_q_wkM__.ravel()[tmp_i8_index_rhs_wkM_],mtr((n_w_sum,n_M_from_ncluster))),
                    torch.reshape(UX_T_M_l2_dM__.ravel()[tmp_i8_index_rhs_dM_],mtr((n_delta_v,n_M_from_ncluster))),
                    UX_M_l2_M_.ravel()[tmp_i8_index_rhs_M_],
                    torch.reshape(svd_V_UX_M_lwnM____.ravel()[tmp_i8_index_rhs_lwnM_],mtr((n_svd_l_vs_n_delta_v,n_w_max,pm_n_UX_rank,n_M_from_ncluster))),
                    torch.reshape(UX_CTF_S_k_q_wnS__.ravel()[tmp_i8_index_rhs_wnS_],mtr((pm_n_w_sum,n_S))),
                    UX_CTF_S_l2_S_.ravel()[tmp_i8_index_rhs_S_],
                )[:10];
                Z_SM__.ravel()[tmp_i8_index_lhs_SM_] = tmp_Z_gpu_SM__.cpu().ravel();
                UX_T_M_l2_dM__.ravel()[tmp_i8_index_lhs_dM_] = tmp_UX_T_M_l2_gpu_dM__.cpu().ravel();
                UX_M_l2_M_.ravel()[index_nM_from_ncluster_] = tmp_UX_M_l2_gpu_M_.cpu().ravel();
                UX_CTF_S_l2_S_.ravel()[:] = tmp_UX_CTF_S_l2_gpu_S_.cpu().ravel();
                X_SM__.ravel()[tmp_i8_index_lhs_SM_] = tmp_X_gpu_SM__.cpu().ravel();
                delta_x_SM__.ravel()[tmp_i8_index_lhs_SM_] = tmp_delta_x_gpu_SM__.cpu().ravel();
                delta_y_SM__.ravel()[tmp_i8_index_lhs_SM_] = tmp_delta_y_gpu_SM__.cpu().ravel();
                gamma_z_SM__.ravel()[tmp_i8_index_lhs_SM_] = tmp_gamma_z_gpu_SM__.cpu().ravel();
                index_sub_SM__.ravel()[tmp_i8_index_lhs_SM_] = tmp_index_sub_gpu_SM__.cpu().ravel();
                del tmp_Z_gpu_SM__;
                del tmp_UX_T_M_l2_gpu_dM__;
                del tmp_UX_M_l2_gpu_M_;
                del tmp_UX_CTF_S_l2_gpu_S_;
                del tmp_X_gpu_SM__;
                del tmp_delta_x_gpu_SM__;
                del tmp_delta_y_gpu_SM__;
                del tmp_gamma_z_gpu_SM__;
                del tmp_index_sub_gpu_SM__;
                tmp_t=toc(tmp_t);
                parameter['device_use'] = 'cpu';
                if (flag_verbose>0): disp(sprintf(' %% tfpmh_Z_gpu_wSM___14: %0.3fs',tmp_t)); #end;
                parameter = parameter_timing_update(parameter,sprintf('%s: tfpmh_Z_gpu_wSM___14',str_thisfunction),tmp_t);
            #end;%if flag_gpu==1;
            #%%%%;
            #%%%%%%%%;
            flag_check=0;
            if flag_check:
                tmp_t=tic();
                for nM_from_ncluster in range(n_M_from_ncluster):
                    nM = int(index_nM_from_ncluster_[nM_from_ncluster].item());
                    for nS in range(n_S):
                        tmp_delta_x = delta_x_SM__[nM,nS].item(); tmp_delta_y = delta_y_SM__[nM,nS].item();
                        tmp_nd = int(torch.min(efind(torch.logical_and(torch.abs(FTK['r8_delta_x_']-tmp_delta_x)<tolerance_machine,torch.abs(FTK['r8_delta_y_']-tmp_delta_y)<tolerance_machine))).item());
                        assert(tmp_nd==np.mod(int(index_sub_SM__[nM,nS].item()),n_delta_v));
                        UX_T_M_l2_SM__[nM,nS] = UX_T_M_l2_dM__[nM,tmp_nd].item();
                    #end;%for nS=0:n_S-1;
                #end;%for nM_from_ncluster=0:n_M_from_ncluster-1;
                tmp_t=toc(tmp_t);
                if (flag_verbose>0): disp(sprintf(' %% UX_T_M_l2_SM__: %0.3fs',tmp_t)); #end;
                parameter = parameter_timing_update(parameter,sprintf('%s: UX_T_M_l2_SM__',str_thisfunction),tmp_t);
            #end;%if flag_check;
            #%%%%%%%%;
            tmp_index_lhs_ = matlab_index_2d_0(n_S,':',n_M,index_nM_from_ncluster_);
            UX_CTF_S_l2_SM__.ravel()[tmp_index_lhs_] = torch.tile(UX_CTF_S_l2_S_,mtr((1,n_M_from_ncluster))).ravel();
        #end;%if (n_M_from_ncluster>0);
    #end;%for ncluster=0:n_cluster-1;

    #%%%%%%%%;
    tmp_t=tic();
    for nM in range(n_M):
        tmp_index_rhs_SM_ = matlab_index_2d_0(n_S,':',n_M,nM);
        tmp_index_sub_ = torch.fmod(index_sub_SM__.ravel()[tmp_index_rhs_SM_],n_delta_v).to(dtype=torch.int32);
        tmp_index_rhs_dM_ = matlab_index_2d_0(n_delta_v,tmp_index_sub_,n_M,nM);
        tmp_index_lhs_SM_ = matlab_index_2d_0(n_S,':',n_M,nM);
        UX_T_M_l2_SM__.ravel()[tmp_index_lhs_SM_] = UX_T_M_l2_dM__.ravel()[tmp_index_rhs_dM_];
    #end;%for nM=0:n_M-1;
    tmp_t=toc(tmp_t);
    if (flag_verbose>0): disp(sprintf(' %% UX_T_M_l2_SM__: %0.3fs',tmp_t)); #end;
    parameter = parameter_timing_update(parameter,sprintf('%s: UX_T_M_l2_SM__',str_thisfunction),tmp_t);
    #%%%%%%%%;

    #%%%%%%%%;
    if (flag_verbose): print(f' %% [finished {str_thisfunction}]');
    #%%%%%%%%;
    return(
        parameter,
        Z_SM__,
        UX_CTF_S_l2_SM__,
        UX_T_M_l2_SM__,
        X_SM__,
        delta_x_SM__,
        delta_y_SM__,
        gamma_z_SM__,
        index_sub_SM__,
        index_nM_from_ncluster__,
        n_index_nM_from_ncluster_,
        M_k_q_wkM__,
        UX_T_M_l2_dM__,
        UX_M_l2_M_,
        svd_V_UX_M_lwnM____, #%<-- or UX_T_M_k_q_dwnM____ ;
        UX_CTF_S_k_q_wnS__,
    );

