from dir_empm.dir_matlab_macros import * ;
from dir_empm.knn_cluster_CTF_k_p_r_kC__1 import knn_cluster_CTF_k_p_r_kC__1 ;
from dir_empm.principled_marching_empirical_cost_matrix_2 import principled_marching_empirical_cost_matrix_2 ;
from dir_empm.principled_marching_cost_matrix_7 import principled_marching_cost_matrix_7 ;
from dir_empm.tfpmut_6 import tfpmut_6 ;

def tfpmut_wrap_6(
        parameter=None,
        n_k_p_r=None,
        k_p_r_=None,
        k_p_r_max=None,
        weight_3d_k_p_r_=None,
        weight_2d_k_p_r_=None,
        n_w_=None,
        weight_2d_k_p_wk_=None,
        l_max_=None,
        n_CTF=None,
        CTF_k_p_wkC__=None,
        index_nCTF_from_nM_=None,
        n_M=None,
        M_k_p_wkM__=None,
        euler_polar_a_ini_M_=None,
        euler_azimu_b_ini_M_=None,
        euler_gamma_z_ini_M_=None,
        image_delta_x_acc_ini_M_=None,
        image_delta_y_acc_ini_M_=None,
        a_k_Y_base_yk_=None,
        delta_sigma_base=None,
):

    #%%%%%%%%;

    str_thisfunction = 'tfpmut_wrap_6';
    tolerance_machine = 1e-6;
    tolerance_machine_1_ = torch.tensor(tolerance_machine).to(dtype=torch.float32);

    if isempty(parameter): parameter = {'type':'parameter'}; #end;%if isempty(parameter);
    #%%%%%%%%;
    if 'fname_pre' not in parameter: disp(sprintf(' %% Warning, fname_pre not set in %s',str_thisfunction)); #end; %<-- parameter
    #%%%%%%%%;
    if 'flag_verbose' not in parameter: parameter['flag_verbose'] = 0; #end; %<-- parameter_bookmark. ;
    flag_verbose = parameter['flag_verbose'];
    if 'tolerance_master' not in parameter: parameter['tolerance_master'] = 1e-2; #end; %<-- parameter_bookmark. ;
    tolerance_master = parameter['tolerance_master'];
    if 'flag_gpu' not in parameter: parameter['flag_gpu'] = 0; #end; %<-- parameter_bookmark. ;
    flag_gpu = parameter['flag_gpu'];
    if 'flag_rank_vs_tolerance' not in parameter: parameter['flag_rank_vs_tolerance'] = 0; #end; %<-- parameter_bookmark. ;
    flag_rank_vs_tolerance = parameter['flag_rank_vs_tolerance'];
    if 'flag_clump_vs_cluster' not in parameter: parameter['flag_clump_vs_cluster'] = parameter['flag_rank_vs_tolerance']; #end; %<-- parameter_bookmark. ;
    parameter['flag_clump_vs_cluster'] = parameter['flag_rank_vs_tolerance']; flag_clump_vs_cluster = parameter['flag_clump_vs_cluster']; #%<-- force;
    if 'tolerance_cluster' not in parameter: parameter['tolerance_cluster'] = parameter['tolerance_master']; #end; %<-- parameter_bookmark. ;
    tolerance_cluster = parameter['tolerance_cluster'];
    if 'tolerance_pm' not in parameter: parameter['tolerance_pm'] = parameter['tolerance_master']; #end; %<-- parameter_bookmark. ;
    tolerance_pm = parameter['tolerance_pm'];
    if 'rank_pm' not in parameter: parameter['rank_pm'] = 10; #end; %<-- parameter_bookmark. ;
    rank_pm = parameter['rank_pm'];
    if 'rank_CTF' not in parameter:
        tmp_i8_index_rhs_ = matlab_index_2d_0(n_w_sum,':',n_CTF,index_nCTF_from_nM_);
        _,SCTF_,_ = matlab_svds(torch.reshape(CTF_k_p_wkC__.ravel()[tmp_i8_index_rhs_],mtr((n_w_sum,n_M))),int(np.minimum(n_w_sum,n_M)));
        rank_CTF = 1+int(torch.max(efind(SCTF_/np.maximum(tolerance_machine,torch.max(SCTF_).item())>tolerance_master)).item());
        parameter['rank_CTF'] = rank_CTF; #%<-- parameter_bookmark. ;
    #end;%if (~isfield(parameter,'rank_CTF')); 
    rank_CTF = parameter['rank_CTF'];
    if 'rseed' not in parameter: parameter['rseed'] = 0; #end; %<-- parameter_bookmark. ;
    rseed = parameter['rseed'];
    if 'delta_r_max' not in parameter: parameter['delta_r_max'] = 0.1; #end; %<-- parameter_bookmark. ;
    delta_r_max = parameter['delta_r_max'];
    if 'n_iteration' not in parameter: parameter['n_iteration'] = 32; #end; %<-- parameter_bookmark. ;
    n_iteration = parameter['n_iteration'];
    if 'flag_alternate_MS_vs_SM' not in parameter: parameter['flag_alternate_MS_vs_SM'] = 1; #end; %<-- parameter_bookmark. ;
    flag_alternate_MS_vs_SM = parameter['flag_alternate_MS_vs_SM'];
    if 'flag_save_stage' not in parameter: parameter['flag_save_stage'] = 0; #end; %<-- parameter_bookmark. ;
    flag_save_stage = parameter['flag_save_stage'];
    #%%%%%%%%;

    if (flag_verbose>0): disp(sprintf(' %% [entering %s]',str_thisfunction)); #end;

    #%%%%%%%%;
    #% index bounds. ;
    #%%%%%%%%;
    n_w_ = n_w_[:n_k_p_r].ravel(); n_w_sum = int(torch.sum(n_w_).item()); n_w_max = int(torch.max(n_w_).item()); n_w_csum_ = cumsum_0(n_w_);
    if numel_unique(n_w_)> 1: disp(sprintf(' %% Warning, flag_uniform_over_n_k_p_r==1 expected in %s',str_thisfunction)); #end;
    n_w_max = int(n_w_csum_[1].item());
    if np.mod(n_w_max,2)!=0: disp(sprintf(' %% Warning, n_w_max %d in %s',n_w_max,str_thisfunction)); #end;
    assert(n_w_sum==n_w_max*n_k_p_r);
    l_max_max = int(torch.max(l_max_).item());
    n_y_ = (l_max_+1)**2;
    n_y_max = int(torch.max(n_y_).item());
    n_y_sum = int(torch.sum(n_y_).item());
    n_y_csum_ = cumsum_0(n_y_);

    #%%%%%%%%;
    if (flag_save_stage>1) and ('fname_pre' in parameter):
        fname_mat = sprintf('%s_stage_0.mat',parameter['fname_pre']);
        disp(sprintf(' %% writing %s',fname_mat));
        matlab_save(
            fname_mat=fname_mat,
            dictionary_original= {
                "flag_verbose":flag_verbose,
                "tolerance_master":tolerance_master,
                "flag_gpu":flag_gpu,
                "flag_rank_vs_tolerance":flag_rank_vs_tolerance,
                "flag_clump_vs_cluster":flag_clump_vs_cluster,
                "tolerance_cluster":tolerance_cluster,
                "tolerance_pm":tolerance_pm,
                "rank_pm":rank_pm,
                "rank_CTF":rank_CTF,
                "rseed":rseed,
                "delta_r_max":delta_r_max,
                "n_iteration":n_iteration,
                "flag_alternate_MS_vs_SM":flag_alternate_MS_vs_SM,
                "n_w_":n_w_,
                "n_w_sum":n_w_sum,
                "n_w_max":n_w_max,
                "n_w_csum_":n_w_csum_,
                "l_max_max":l_max_max,
                "n_y_":n_y_,
                "n_y_max":n_y_max,
                "n_y_sum":n_y_sum,
                "n_y_csum_":n_y_csum_,
            },
        );
    #end;%if ( isfield(parameter,'fname_pre'));
    #%%%%%%%%;

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;
    if (flag_clump_vs_cluster==0):
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;

        #%%%%%%%%;
        #% Find clusters. ;
        #%%%%%%%%;
        tmp_t=tic();
        CTF_k_p_r_kC__ = torch.reshape(torch.mean(torch.reshape(CTF_k_p_wkC__,mtr((n_w_max,n_k_p_r,n_CTF))),dim=2-0),mtr((n_k_p_r,n_CTF)));
        (
            parameter,
            index_ncluster_from_nCTF_,
        ) = knn_cluster_CTF_k_p_r_kC__1(
            parameter,
            n_k_p_r,
            k_p_r_,
            weight_2d_k_p_r_,
            n_CTF,
            CTF_k_p_r_kC__,
        )[:2];
        #%%%%%%%%;
        n_cluster = 1+int(torch.max(index_ncluster_from_nCTF_).item());
        index_ncluster_from_nM_ = index_ncluster_from_nCTF_[index_nCTF_from_nM_];
        index_nM_from_ncluster__ = cell(n_cluster);
        n_index_nM_from_ncluster_ = torch.zeros(n_cluster).to(dtype=torch.int32);
        for ncluster in range(n_cluster):
            index_nM_from_ncluster__[ncluster] = efind(index_ncluster_from_nM_==ncluster);
            n_index_nM_from_ncluster_[ncluster] = numel(index_nM_from_ncluster__[ncluster]);
        #end;%for ncluster=0:n_cluster-1;
        tmp_t = toc(tmp_t);
        if (flag_verbose>1): disp(sprintf(' %% knn_cluster_CTF_k_p_r_kC__1: time %0.2fs',tmp_t)); #end;
        parameter = parameter_timing_update(parameter,sprintf('%s: knn_cluster_CTF_k_p_r_kC__1',str_thisfunction),tmp_t);

        #%%%%%%%%;
        if (flag_save_stage>1) and ('fname_pre' in parameter):
            fname_mat = sprintf('%s_stage_1.mat',parameter['fname_pre']);
            disp(sprintf(' %% writing %s',fname_mat));
            matlab_save(
                fname_mat=fname_mat,
                dictionary_original= {
                    "index_ncluster_from_nCTF_":index_ncluster_from_nCTF_,
                    "n_k_p_r":n_k_p_r,
                    "k_p_r_":k_p_r_,
                    "weight_2d_k_p_r_":weight_2d_k_p_r_,
                    "n_CTF":n_CTF,
                    "CTF_k_p_r_kC__":CTF_k_p_r_kC__,
                    "index_nM_from_ncluster__":index_nM_from_ncluster__,
                    "n_index_nM_from_ncluster_":n_index_nM_from_ncluster_,
                },
            );
        #end;%if ( isfield(parameter,'fname_pre'));
        #%%%%%%%%;

        #%%%%%%%%;
        #% Now determine the principal-modes for each cluster. ;
        #%%%%%%%%;
        if  isempty(a_k_Y_base_yk_):
            tmp_t=tic();
            X_2d_Memp_d1_kkc___ = torch.zeros(mtr((n_k_p_r,n_k_p_r,n_cluster))).to(dtype=torch.float32);
            X_2d_Memp_d1_weight_rc__ = torch.zeros(mtr((n_k_p_r,n_cluster))).to(dtype=torch.float32);
            for ncluster in range(n_cluster):
                index_nM_from_ncluster_ = index_nM_from_ncluster__[ncluster];
                n_index_nM_from_ncluster = int(n_index_nM_from_ncluster_[ncluster].item());
                assert(n_index_nM_from_ncluster==numel(index_nM_from_ncluster_));
                tmp_i8_index_rhs_ = matlab_index_2d_0(n_w_sum,':',n_M,index_nM_from_ncluster_);
                (
                    X_2d_Memp_d1_kk__,
                    X_2d_Memp_d1_weight_r_,
                ) = principled_marching_empirical_cost_matrix_2(
                    n_k_p_r,
                    k_p_r_,
                    weight_2d_k_p_r_,
                    n_w_,
                    n_index_nM_from_ncluster,
                    torch.reshape(M_k_p_wkM__.ravel()[tmp_i8_index_rhs_],mtr((n_w_sum,n_index_nM_from_ncluster))),
                )[:2];
                X_2d_Memp_d1_kkc___[ncluster,:,:] = X_2d_Memp_d1_kk__;
                X_2d_Memp_d1_weight_rc__[ncluster,:] = X_2d_Memp_d1_weight_r_;
                del X_2d_Memp_d1_kk__; del X_2d_Memp_d1_weight_r_;
            #end;%for ncluster=0:n_cluster-1;
            pm_X_kkc___ = X_2d_Memp_d1_kkc___;
            pm_X_weight_rc__ = X_2d_Memp_d1_weight_rc__;
            del X_2d_Memp_d1_kkc___; del X_2d_Memp_d1_weight_rc__;
            tmp_t = toc(tmp_t);
            if (flag_verbose>1): disp(sprintf(' %% principled_marching_empirical_cost_matrix_2: time %0.2fs',tmp_t)); #end;
            parameter = parameter_timing_update(parameter,sprintf('%s: principled_marching_empirical_cost_matrix_2',str_thisfunction),tmp_t);
        #end;%if  isempty(a_k_Y_base_yk_);
        #%%%%%%%%;
        if not isempty(a_k_Y_base_yk_):
            if isempty(delta_sigma_base): delta_sigma_base = 0.0; #end; %<-- no translation as default. ;
            X_2d_xavg_dx_kkc___ = torch.zeros(mtr((n_k_p_r,n_k_p_r,n_cluster))).to(dtype=torch.float32);
            X_2d_xavg_dx_weight_rc__ = torch.zeros(mtr((n_k_p_r,n_cluster))).to(dtype=torch.float32);
            for ncluster in range(n_cluster):
                index_nM_from_ncluster_ = index_nM_from_ncluster__[ncluster];
                n_index_nM_from_ncluster = int(n_index_nM_from_ncluster_[ncluster].item());
                assert(n_index_nM_from_ncluster==numel(index_nM_from_ncluster_));
                tmp_i8_index_rhs_ = matlab_index_2d_0(n_k_p_r,':',n_CTF,index_nCTF_from_nM_[index_nM_from_ncluster_]);
                tmp_CTF_k_p_r_xavg_k_ = torch.mean(torch.reshape(CTF_k_p_r_kC__.ravel()[tmp_i8_index_rhs_],mtr((n_k_p_r,n_index_nM_from_ncluster))),dim=1-1).ravel(); assert(numel(tmp_CTF_k_p_r_xavg_k_)==n_k_p_r);
                tmp_CTF_k_p_r_xavg_kk__ = torch.reshape(tmp_CTF_k_p_r_xavg_k_,mtr((n_k_p_r,1)))*torch.reshape(tmp_CTF_k_p_r_xavg_k_,mtr((1,n_k_p_r)));
                tmp_t=tic();
                (
                    X_2d_xavg_dx_kk__,
                    X_2d_xavg_dx_weight_r_,
                ) = principled_marching_cost_matrix_7(
                    n_k_p_r,
                    k_p_r_,
                    weight_2d_k_p_r_,
                    l_max_,
                    None,
                    None,
                    a_k_Y_base_yk_,
                    tmp_CTF_k_p_r_xavg_kk__,
                    delta_sigma_base,
                )[:2];
                tmp_t = toc(tmp_t);
                if (flag_verbose>1): disp(sprintf(' %% principled_marching_cost_matrix_7: time %0.2fs',tmp_t)); #end;
                parameter = parameter_timing_update(parameter,sprintf('%s: principled_marching_cost_matrix_7',str_thisfunction),tmp_t);
                X_2d_xavg_dx_kkc___[ncluster,:,:] = X_2d_xavg_dx_kk__;
                X_2d_xavg_dx_weight_rc__[ncluster,:] = X_2d_xavg_dx_weight_r_;
                del X_2d_xavg_dx_kk__; del X_2d_xavg_dx_weight_r_;
            #end;%for ncluster=0:n_cluster-1;
            pm_X_kkc___ = X_2d_xavg_dx_kkc___;
            pm_X_weight_rc__ = X_2d_xavg_dx_weight_rc__;
            del X_2d_xavg_dx_kkc___; del X_2d_xavg_dx_weight_rc__;
        #end;%if ~isempty(a_k_Y_base_yk_);
        #%%%%%%%%;
        tmp_t = tic();
        n_UX_rank = n_k_p_r-1; #%<-- just to check dimensions. ;
        pm_UX_knc___ = torch.zeros(mtr((n_k_p_r,n_UX_rank,n_cluster))).to(dtype=torch.float32);
        pm_SX_kc__ = torch.zeros(mtr((n_UX_rank,n_cluster))).to(dtype=torch.float32);
        pm_n_UX_rank_c_ = torch.zeros(n_cluster).to(dtype=torch.int32);
        for ncluster in range(n_cluster):
            tmp_X_kk__ = torch.reshape(pm_X_kkc___[ncluster,:,:],mtr((n_k_p_r,n_k_p_r)));
            tmp_UX__,tmp_SX_,tmp_VX__ = matlab_svds(tmp_X_kk__,n_UX_rank);
            pm_n_UX_rank = 1+int(torch.max(efind(tmp_SX_/np.maximum(tolerance_machine,torch.max(tmp_SX_).item())> tolerance_pm)).item());
            tmp_i8_index_rhs_ = matlab_index_2d_0(n_k_p_r,":",n_UX_rank,torch.arange(n_UX_rank));
            pm_UX_knc___[ncluster,:,:] = torch.reshape(tmp_UX__.ravel()[tmp_i8_index_rhs_],mtr((n_k_p_r,n_UX_rank)));
            pm_SX_kc__[ncluster,:] = tmp_SX_[torch.arange(n_UX_rank)].ravel();
            pm_n_UX_rank_c_[ncluster] = pm_n_UX_rank;
            if (flag_verbose>1): disp(sprintf(' %% ncluster %.2d/%.2d: pm_n_UX_rank %d/%d',ncluster,n_cluster,pm_n_UX_rank,n_UX_rank)); #end;
        #end;%for ncluster=0:n_cluster-1;
        tmp_t = toc(tmp_t);
        if (flag_verbose>1): disp(sprintf(' %% pm_UX_knc___: %0.3fs',tmp_t)); #end;
        parameter = parameter_timing_update(parameter,sprintf('%s: tmp_X_kk__',str_thisfunction),tmp_t);
        #%%%%%%%%;

        #%%%%%%%%;
        if (flag_save_stage>1) and ('fname_pre' in parameter):
            fname_mat = sprintf('%s_stage_2.mat',parameter['fname_pre']);
            disp(sprintf(' %% writing %s',fname_mat));
            matlab_save(
                fname_mat=fname_mat,
                dictionary_original= {
                    "n_UX_rank":n_UX_rank,
                    "pm_UX_knc___":pm_UX_knc___,
                    "pm_SX_kc__":pm_SX_kc__,
                    "pm_n_UX_rank_c_":pm_n_UX_rank_c_,
                },
            );
        #end;%if ( isfield(parameter,'fname_pre'));
        #%%%%%%%%;

        #%%%%%%%%;
        #% Now run tfpmut_6. ;
        #%%%%%%%%;
        a_k_Y_reco_yki__ = None;
        FTK=None;
        rng(rseed);
        if not isempty(euler_polar_a_ini_M_): euler_polar_a_M_ = euler_polar_a_ini_M_; 
        else: euler_polar_a_M_= 1*pi*torch.rand(n_M).to(dtype=torch.float32); #end;
        if not isempty(euler_azimu_b_ini_M_): euler_azimu_b_M_ = euler_azimu_b_ini_M_; 
        else: euler_azimu_b_M_= 2*pi*torch.rand(n_M).to(dtype=torch.float32); #end;
        if not isempty(euler_gamma_z_ini_M_): euler_gamma_z_M_ = euler_gamma_z_ini_M_; 
        else: euler_gamma_z_M_= 2*pi*torch.rand(n_M).to(dtype=torch.float32); #end;
        if not isempty(image_delta_x_acc_ini_M_): image_delta_x_acc_M_ = image_delta_x_acc_ini_M_; 
        else: image_delta_x_acc_M_= torch.zeros(n_M).to(dtype=torch.float32); #end;
        if not isempty(image_delta_y_acc_ini_M_): image_delta_y_acc_M_ = image_delta_y_acc_ini_M_; 
        else: image_delta_y_acc_M_= torch.zeros(n_M).to(dtype=torch.float32); #end;
        image_delta_x_upd_M_ = None;
        image_delta_y_upd_M_ = None;
        flag_image_delta_upd_M_ = None;
        image_I_value_M_ = None;
        #%%%%%%%%;
        if (flag_save_stage>1) and ('fname_pre' in parameter):
            fname_mat = sprintf('%s_stage_3.mat',parameter['fname_pre']);
            disp(sprintf(' %% writing %s',fname_mat));
            matlab_save(
                fname_mat=fname_mat,
                dictionary_original= {
                    "n_k_p_r":n_k_p_r,
                    "k_p_r_":k_p_r_,
                    "k_p_r_max":k_p_r_max,
                    "weight_3d_k_p_r_":weight_3d_k_p_r_,
                    "weight_2d_k_p_r_":weight_2d_k_p_r_,
                    "n_w_":n_w_,
                    "weight_2d_k_p_wk_":weight_2d_k_p_wk_,
                    "l_max_":l_max_,
                    "n_CTF":n_CTF,
                    "CTF_k_p_wkC__":CTF_k_p_wkC__,
                    "index_nCTF_from_nM_":index_nCTF_from_nM_,
                    "n_M":n_M,
                    "M_k_p_wkM__":M_k_p_wkM__,
                    "n_cluster":n_cluster,
                    "index_ncluster_from_nCTF_":index_ncluster_from_nCTF_,
                    "pm_n_UX_rank_c_":pm_n_UX_rank_c_,
                    "pm_UX_knc___":pm_UX_knc___,
                    "pm_X_weight_rc__":pm_X_weight_rc__,
                    "euler_polar_a_M_":euler_polar_a_M_,
                    "euler_azimu_b_M_":euler_azimu_b_M_,
                    "euler_gamma_z_M_":euler_gamma_z_M_,
                    "image_delta_x_acc_M_":image_delta_x_acc_M_,
                    "image_delta_y_acc_M_":image_delta_y_acc_M_,
                },
            );
        #end;%if ( isfield(parameter,'fname_pre'));
        #%%%%%%%%;
        #%%%%;
        tmp_t = tic();
        (
            parameter,
            a_k_Y_reco_yki__,
            euler_polar_a_Mi__,
            euler_azimu_b_Mi__,
            euler_gamma_z_Mi__,
            image_delta_x_acc_Mi__,
            image_delta_y_acc_Mi__,
            image_delta_x_upd_Mi__,
            image_delta_y_upd_Mi__,
            flag_image_delta_upd_Mi__,
            image_I_value_Mi__,
            image_X_value_Mi__,
            image_S_index_Mi__,
        ) = tfpmut_6(
            parameter,
            n_k_p_r,
            k_p_r_,
            k_p_r_max,
            weight_3d_k_p_r_,
            weight_2d_k_p_r_,
            n_w_,
            weight_2d_k_p_wk_,
            l_max_,
            n_CTF,
            CTF_k_p_wkC__,
            index_nCTF_from_nM_,
            n_M,
            M_k_p_wkM__,
            n_cluster,
            index_ncluster_from_nCTF_,
            pm_n_UX_rank_c_,
            pm_UX_knc___,
            pm_X_weight_rc__,
            FTK,
            euler_polar_a_M_,
            euler_azimu_b_M_,
            euler_gamma_z_M_,
            image_delta_x_acc_M_,
            image_delta_y_acc_M_,
            image_delta_x_upd_M_,
            image_delta_y_upd_M_,
            flag_image_delta_upd_M_,
            image_I_value_M_,
        )[:13];
        tmp_t = toc(tmp_t);
        if (flag_verbose>1): disp(sprintf(' %% tfpmut_6: time %0.2fs',tmp_t)); #end;
        parameter = parameter_timing_update(parameter,sprintf('%s: tfpmut_6',str_thisfunction),tmp_t);
        #%%%%%%%%;

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;
    #end;%if (flag_clump_vs_cluster==0);
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;
    if (flag_clump_vs_cluster==1):
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;
        disp(sprintf(' %% Warning, flag_clump_vs_cluster==%d not implemented in %s',flag_clump_vs_cluster,str_thisfunction));
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;
    #end;%if (flag_clump_vs_cluster==1);
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;

    #%%%%%%%%;
    if 'fname_pre' in parameter:
        fname_mat = sprintf('%s.mat',parameter['fname_pre']);
        disp(sprintf(' %% writing %s',fname_mat));
        matlab_save(
            fname_mat=fname_mat,
            dictionary_original= {
                "a_k_Y_reco_yki__":a_k_Y_reco_yki__,
                "euler_polar_a_Mi__":euler_polar_a_Mi__,
                "euler_azimu_b_Mi__":euler_azimu_b_Mi__,
                "euler_gamma_z_Mi__":euler_gamma_z_Mi__,
                "image_delta_x_acc_Mi__":image_delta_x_acc_Mi__,
                "image_delta_y_acc_Mi__":image_delta_y_acc_Mi__,
                "image_delta_x_upd_Mi__":image_delta_x_upd_Mi__,
                "image_delta_y_upd_Mi__":image_delta_y_upd_Mi__,
                "flag_image_delta_upd_Mi__":flag_image_delta_upd_Mi__,
                "image_I_value_Mi__":image_I_value_Mi__,
                "image_X_value_Mi__":image_X_value_Mi__,
                "image_S_index_Mi__":image_S_index_Mi__,
            },
        );
    #end;%if ( isfield(parameter,'fname_pre'));
    #%%%%%%%%;

    if (flag_verbose>0): disp(sprintf(' %% [finished %s]',str_thisfunction)); #end;
    
    return(
        parameter,
        a_k_Y_reco_yki__,
        euler_polar_a_Mi__,
        euler_azimu_b_Mi__,
        euler_gamma_z_Mi__,
        image_delta_x_acc_Mi__,
        image_delta_y_acc_Mi__,
        image_delta_x_upd_Mi__,
        image_delta_y_upd_Mi__,
        flag_image_delta_upd_Mi__,
        image_I_value_Mi__,
        image_X_value_Mi__,
        image_S_index_Mi__,
    );

