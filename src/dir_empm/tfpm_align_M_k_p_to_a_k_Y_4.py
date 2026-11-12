from dir_empm.dir_matlab_macros import * ;
from dir_empm.tfh_FTK_4 import tfh_FTK_4 ;
from dir_empm.transf_p_to_p import transf_p_to_p ;
from dir_empm.qbp_uniform_over_n_k_p_r_10 import qbp_uniform_over_n_k_p_r_10 ;
from dir_empm.local_yk__from_yk_ import local_yk__from_yk_ ;
from dir_empm.pm_template_3 import pm_template_3 ;
from dir_empm.tfpmh_Z_cluster_wrap_SM__14 import tfpmh_Z_cluster_wrap_SM__14 ;
from dir_empm.tfpmh_MS_vs_SM_2 import tfpmh_MS_vs_SM_2 ;
from dir_empm.principled_marching_cost_matrix_7 import principled_marching_cost_matrix_7 ;
from dir_empm.principled_marching_cost_matrix_3 import principled_marching_cost_matrix_3 ;
from dir_empm.register_spharm_to_spharm_3 import register_spharm_to_spharm_3 ; #<-- TODO: not yet ported. ;

def tfpm_align_M_k_p_to_a_k_Y_4(
        parameter=None,
        n_k_p_r=None,
        k_p_r_=None,
        k_p_r_max=None,
        weight_3d_k_p_r_=None,
        weight_2d_k_p_r_=None,
        n_w_=None,
        weight_2d_k_p_wk_=None,
        n_M=None,
        M_orig_k_p_wkM__=None,
        n_cluster=None,
        index_ncluster_from_nCTF_=None,
        n_CTF=None,
        index_nCTF_from_nM_=None,
        CTF_k_p_wkC__=None,
        l_max_=None,
        a_k_Y_true_yk_=None,
        euler_polar_a_init_M_=None,
        euler_azimu_b_init_M_=None,
        euler_gamma_z_init_M_=None,
        image_delta_x_init_M_=None,
        image_delta_y_init_M_=None,
        euler_polar_a_true_M_=None,
        euler_azimu_b_true_M_=None,
        euler_gamma_z_true_M_=None,
        image_delta_x_true_M_=None,
        image_delta_y_true_M_=None,
):

    #%%%%%%%%;

    str_thisfunction = 'tfpm_align_M_k_p_to_a_k_Y_4';
    tolerance_machine = 1e-6;
    tolerance_machine_1_ = torch.tensor(tolerance_machine).to(dtype=torch.float32);

    if isempty(parameter): parameter = {'type':'parameter'}; #end;%if isempty(parameter);
    #%%%%%%%%;
    if 'flag_verbose' not in parameter: parameter['flag_verbose'] = 0; #end; %<-- parameter_bookmark. ;
    flag_verbose = parameter['flag_verbose'];
    if 'tolerance_master' not in parameter: parameter['tolerance_master'] = 1e-2; #end; %<-- parameter_bookmark. ;
    tolerance_master = parameter['tolerance_master'];
    if 'flag_gpu' not in parameter: parameter['flag_gpu'] = 0; #end; %<-- parameter_bookmark. ;
    flag_gpu = parameter['flag_gpu'];
    if 'rseed' not in parameter: parameter['rseed'] = 0; #end; %<-- parameter_bookmark. ;
    rseed = parameter['rseed'];
    if 'fname_align_a_k_Y_pre' not in parameter: parameter['fname_align_a_k_Y_pre'] = None; #end; %<-- parameter_bookmark. ;
    if 'tolerance_pm' not in parameter: parameter['tolerance_pm'] = parameter['tolerance_master']; #end; %<-- parameter_bookmark. ;
    tolerance_pm = parameter['tolerance_pm'];
    if 'rseed' not in parameter: parameter['rseed'] = 0; #end; %<-- parameter_bookmark. ;
    rseed = parameter['rseed'];
    if 'n_iteration' not in parameter: parameter['n_iteration'] = 1; #end; %<-- parameter_bookmark. ;
    n_iteration = parameter['n_iteration'];
    if 'delta_r_max' not in parameter: parameter['delta_r_max'] = 0.1; #end; %<-- parameter_bookmark. ;
    delta_r_max = parameter['delta_r_max'];
    if 'delta_r_upb' not in parameter: parameter['delta_r_upb'] = 2*parameter['delta_r_max']; #end; %<-- parameter_bookmark. ;
    delta_r_upb = parameter['delta_r_upb'];
    if 'n_delta_v_requested' not in parameter: parameter['n_delta_v_requested'] = 0; #end; %<-- parameter_bookmark. ;
    n_delta_v_requested = parameter['n_delta_v_requested'];
    if 'flag_MS_vs_SM' not in parameter: parameter['flag_MS_vs_SM'] = 0; #end; %<-- parameter_bookmark. ;
    flag_MS_vs_SM = parameter['flag_MS_vs_SM'];
    if 't_eq_d_double' not in parameter: parameter['t_eq_d_double'] = 1.0; #end; %<-- parameter_bookmark. ;
    t_eq_d_double = parameter['t_eq_d_double'];
    if 'svd_eps' not in parameter: parameter['svd_eps'] = tolerance_master; #end; %<-- parameter_bookmark. ;
    svd_eps = parameter['svd_eps'];
    if 'qbp_eps' not in parameter: parameter['qbp_eps'] = tolerance_master; #end; %<-- parameter_bookmark. ;
    qbp_eps = parameter['qbp_eps'];
    #%%%%%%%%;

    if (flag_verbose>0): disp(sprintf(' %% [entering %s]',str_thisfunction)); #end;

    #%%%%%%%%;
    n_w_ = n_w_[:n_k_p_r].ravel(); n_w_sum = int(sum_0(n_w_)); n_w_max = int(max_0(n_w_)); n_w_csum_ = cumsum_0(n_w_);
    if numel_unique(n_w_)> 1: disp(sprintf(' %% Warning, flag_uniform_over_n_k_p_r==1 expected in %s',str_thisfunction)); #end;
    n_w_max = int(n_w_csum_[1].item());
    if np.mod(n_w_max,2)!=0: disp(sprintf(' %% Warning, n_w_max %d in %s',n_w_max,str_thisfunction)); #end;
    assert(n_w_sum==n_w_max*n_k_p_r);
    l_max_max = int(max_0(l_max_));
    n_y_ = (l_max_+1)**2;
    n_y_max = int(max_0(n_y_));
    n_y_sum = int(sum_0(n_y_));
    n_y_csum_ = cumsum_0(n_y_);
    #%%%%%%%%;
    n_cluster = 1+int(max_0(index_ncluster_from_nCTF_));
    index_ncluster_from_nM_ = index_ncluster_from_nCTF_[index_nCTF_from_nM_];
    index_nM_from_ncluster__ = cell(n_cluster);
    n_index_nM_from_ncluster_ = torch.zeros(n_cluster).to(dtype=torch.int32);
    for ncluster in range(n_cluster):
        index_nM_from_ncluster__[ncluster] = efind(index_ncluster_from_nM_==ncluster);
        n_index_nM_from_ncluster_[ncluster] = numel(index_nM_from_ncluster__[ncluster]);
    #end;%for ncluster=0:n_cluster-1;
    #%%%%%%%%;

    #%%%%%%%%;
    #% construct CTF of same size as images. ;
    #%%%%%%%%
    if (matlab_size(CTF_k_p_wkC__,0)==n_k_p_r):
        n_CTF = matlab_size(CTF_k_p_wkC__,1);
        CTF_k_p_r_kC__ = CTF_k_p_wkC__;
        CTF_k_p_wkC__ = torch.reshape(torch.tile(torch.reshape(CTF_k_p_r_kC__,mtr((1,n_k_p_r,n_CTF))),mtr((n_w_max,1,1))),mtr((n_w_sum,n_CTF)));
    #end;%if (matlab_size(CTF_k_p_wkC__,1+0)==n_k_p_r);
    #%%%%%%%%;
    #% Note: in the matlab, CTF_k_p_r_kC__ is only defined when the CTF is given as n_k_p_r-by-n_CTF. ;
    #% Here (as in tfpmut_6) it is always defined, as the angular average of CTF_k_p_wkC__. ;
    #%%%%%%%%;
    CTF_k_p_r_kC__ = torch.reshape(torch.mean(torch.reshape(CTF_k_p_wkC__,mtr((n_w_max,n_k_p_r,n_CTF))),2-0),mtr((n_k_p_r,n_CTF)));
    tmp_i8_index_rhs_ = matlab_index_2d_0(n_w_sum,':',n_CTF,index_nCTF_from_nM_);
    CTF_k_p_wkM__ = torch.reshape(CTF_k_p_wkC__.ravel()[tmp_i8_index_rhs_],mtr((n_w_sum,n_M)));
    #%%%%%%%%;

    flag_found=0;
    if not isempty(parameter['fname_align_a_k_Y_pre']):
        tmp_fname_mat = sprintf('%s.mat',parameter['fname_align_a_k_Y_pre']);
        if matlab_exist_file(tmp_fname_mat):
            if (flag_verbose>0): disp(sprintf(' %% %s found, loading',tmp_fname_mat)); #end;
            tmp_ = matlab_load(fname_mat=tmp_fname_mat);
            a_k_Y_reco_yki__ = tmp_['a_k_Y_reco_yki__'];
            corr_a_k_Y_i_ = tmp_['corr_a_k_Y_i_'];
            euler_polar_a_Mi__ = tmp_['euler_polar_a_Mi__'];
            euler_azimu_b_Mi__ = tmp_['euler_azimu_b_Mi__'];
            euler_gamma_z_Mi__ = tmp_['euler_gamma_z_Mi__'];
            image_delta_x_acc_Mi__ = tmp_['image_delta_x_acc_Mi__'];
            image_delta_y_acc_Mi__ = tmp_['image_delta_y_acc_Mi__'];
            image_delta_x_upd_Mi__ = tmp_['image_delta_x_upd_Mi__'];
            image_delta_y_upd_Mi__ = tmp_['image_delta_y_upd_Mi__'];
            flag_image_delta_upd_Mi__ = tmp_['flag_image_delta_upd_Mi__'];
            image_I_value_Mi__ = tmp_['image_I_value_Mi__'];
            image_X_value_Mi__ = tmp_['image_X_value_Mi__'];
            image_S_index_Mi__ = tmp_['image_S_index_Mi__'];
            n_S = tmp_['n_S'];
            viewing_azimu_b_S_ = tmp_['viewing_azimu_b_S_'];
            viewing_polar_a_S_ = tmp_['viewing_polar_a_S_'];
            X_SMi___ = tmp_['X_SMi___'];
            delta_x_SMi___ = tmp_['delta_x_SMi___'];
            delta_y_SMi___ = tmp_['delta_y_SMi___'];
            gamma_z_SMi___ = tmp_['gamma_z_SMi___'];
            I_value_SMi___ = tmp_['I_value_SMi___'];
            del tmp_;
            flag_found=1;
        #end;%if  exist(tmp_fname_mat,'file');
    #end;%if (~isempty(parameter.fname_align_a_k_Y_pre));

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;
    if not flag_found:
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;
        if (flag_verbose>0): disp(sprintf(' %% mat-file not found, creating')); #end;

        #%%%%%%%%;
        #% First calculate average CTFs for each cluster. ;
        #%%%%%%%%;
        CTF_k_p_r_xavg_kc__ = torch.zeros(mtr((n_k_p_r,n_cluster))).to(dtype=torch.float32);
        for ncluster in range(n_cluster):
            index_nM_from_ncluster_ = index_nM_from_ncluster__[ncluster];
            n_index_nM_from_ncluster = int(n_index_nM_from_ncluster_[ncluster].item());
            assert(n_index_nM_from_ncluster==numel(index_nM_from_ncluster_));
            tmp_i8_index_rhs_ = matlab_index_2d_0(n_w_sum,':',n_CTF,index_nCTF_from_nM_[index_nM_from_ncluster_]);
            CTF_k_p_r_xavg_k_ = torch.mean(torch.reshape(CTF_k_p_wkC__.ravel()[tmp_i8_index_rhs_],mtr((n_w_max,n_k_p_r,n_index_nM_from_ncluster))),[2-0,2-2]).ravel(); assert(numel(CTF_k_p_r_xavg_k_)==n_k_p_r);
            CTF_k_p_r_xavg_kc__[ncluster,:] = CTF_k_p_r_xavg_k_;
        #end;%for ncluster=0:n_cluster-1;

        #%%%%%%%%;
        #% Now calculate idealized principal-modes for each nCTF. ;
        #%%%%%%%%;
        n_UX_rank = n_k_p_r-1; #%<-- just to check dimensions. ;
        a_k_Y_true_yk__ = local_yk__from_yk_(n_k_p_r,l_max_,a_k_Y_true_yk_);
        #%%%%%%%%;
        tmp_t = tic();
        X_2d_xavg_dx_kkc___ = torch.zeros(mtr((n_k_p_r,n_k_p_r,n_cluster))).to(dtype=torch.float32);
        X_2d_xavg_dx_weight_rc__ = torch.zeros(mtr((n_k_p_r,n_cluster))).to(dtype=torch.float32);
        for ncluster in range(n_cluster):
            index_nM_from_ncluster_ = index_nM_from_ncluster__[ncluster];
            n_index_nM_from_ncluster = int(n_index_nM_from_ncluster_[ncluster].item());
            assert(n_index_nM_from_ncluster==numel(index_nM_from_ncluster_));
            CTF_k_p_r_xavg_k_ = CTF_k_p_r_xavg_kc__[ncluster,:].ravel();
            CTF_k_p_r_xavg_kk__ = torch.reshape(CTF_k_p_r_xavg_k_,mtr((n_k_p_r,1)))*torch.reshape(CTF_k_p_r_xavg_k_,mtr((1,n_k_p_r)));
            tmp_delta_sigma = delta_r_max;
            if (tmp_delta_sigma> 0):
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
                    a_k_Y_true_yk_,
                    CTF_k_p_r_xavg_kk__,
                    tmp_delta_sigma,
                )[:2];
            #end;%if (tmp_delta_sigma> 0);
            if (tmp_delta_sigma==0):
                (
                    X_2d_xavg_dx_kk__,
                    X_2d_xavg_dx_weight_r_,
                ) = principled_marching_cost_matrix_3(
                    n_k_p_r,
                    weight_2d_k_p_r_,
                    l_max_max,
                    a_k_Y_true_yk__,
                    CTF_k_p_r_xavg_kk__,
                )[:2];
            #end;%if (tmp_delta_sigma==0);
            X_2d_xavg_dx_kkc___[ncluster,:,:] = X_2d_xavg_dx_kk__;
            X_2d_xavg_dx_weight_rc__[ncluster,:] = X_2d_xavg_dx_weight_r_;
            del X_2d_xavg_dx_kk__; del X_2d_xavg_dx_weight_r_;
        #end;%for ncluster=0:n_cluster-1;
        pm_X_weight_rc__ = X_2d_xavg_dx_weight_rc__;
        tmp_t = toc(tmp_t);
        if (flag_verbose>0): disp(sprintf(' %% X_2d_xavg_dx_kkc___: %0.3fs',tmp_t)); #end;
        parameter = parameter_timing_update(parameter,'X_2d_xavg_dx_kkc___',tmp_t);
        #%%%%%%%%;
        tmp_t = tic();
        pm_UX_knc___ = torch.zeros(mtr((n_k_p_r,n_UX_rank,n_cluster))).to(dtype=torch.float32);
        pm_SX_kc__ = torch.zeros(mtr((n_UX_rank,n_cluster))).to(dtype=torch.float32);
        pm_n_UX_rank_c_ = torch.zeros(n_cluster).to(dtype=torch.int32);
        for ncluster in range(n_cluster):
            tmp_X__ = torch.reshape(X_2d_xavg_dx_kkc___[ncluster,:,:],mtr((n_k_p_r,n_k_p_r)));
            tmp_UX__,tmp_SX_,tmp_VX__ = matlab_svds(tmp_X__,n_UX_rank);
            pm_n_UX_rank = int(max_0(1+efind(tmp_SX_/np.maximum(tolerance_machine,max_0(tmp_SX_))> tolerance_pm)));
            tmp_i8_index_rhs_ = matlab_index_2d_0(n_k_p_r,':',n_UX_rank,torch.arange(n_UX_rank));
            pm_UX_knc___[ncluster,:,:] = torch.reshape(tmp_UX__.ravel()[tmp_i8_index_rhs_],mtr((n_k_p_r,n_UX_rank)));
            pm_SX_kc__[ncluster,:] = tmp_SX_[torch.arange(n_UX_rank)].ravel();
            pm_n_UX_rank_c_[ncluster] = pm_n_UX_rank;
            if (flag_verbose>1): disp(sprintf(' %% ncluster %.2d/%.2d: pm_n_UX_rank %d/%d',ncluster,n_cluster,pm_n_UX_rank,n_UX_rank)); #end;
        #end;%for ncluster=0:n_cluster-1;
        tmp_t = toc(tmp_t);
        if (flag_verbose>0): disp(sprintf(' %% pm_UX_knc___: %0.3fs',tmp_t)); #end;
        parameter = parameter_timing_update(parameter,'pm_UX_knc___',tmp_t);
        #%%%%%%%%;

        pm_n_UX_rank_max = int(max_0(pm_n_UX_rank_c_));
        pm_n_y_max_max = (1+l_max_max)**2;
        if (flag_verbose>0):
            for ncluster in range(n_cluster):
                pm_n_UX_rank = int(pm_n_UX_rank_c_[ncluster].item());
                if (flag_verbose>0): disp(sprintf(' %% ncluster %.2d/%.2d --> pm_n_UX_rank %d/%d',ncluster,n_cluster,pm_n_UX_rank,pm_n_UX_rank_max)); #end;
            #end;%for ncluster=0:n_cluster-1;
        #end;%if (flag_verbose>0);

        FTK = None;
        if isempty(FTK):
            tmp_t = tic();
            r8_delta_r_max = float(delta_r_max);
            r8_svd_eps = float(svd_eps);
            parameter_FTK = parameter.copy(); #<-- note copy. ;
            parameter_FTK['r8_delta_r_max'] = r8_delta_r_max;
            parameter_FTK['r8_svd_eps'] = r8_svd_eps;
            parameter_FTK['n_delta_v_requested'] = n_delta_v_requested;
            parameter_FTK,FTK = tfh_FTK_4(parameter_FTK,n_k_p_r,k_p_r_,k_p_r_max);
            tmp_t = toc(tmp_t);
            if (flag_verbose>0): disp(sprintf(' %% FTK: %0.3fs',tmp_t)); #end;
            parameter = parameter_timing_update(parameter,'tfh_FTK_4',tmp_t);
        #end;%if isempty(FTK);
        assert(FTK['r8_svd_d_max']>=delta_r_max);
        assert(FTK['n_delta_v']>=n_delta_v_requested);

        #%%%%%%%%;
        tmp_t = tic();
        (
            S_k_p_wkS__,
            _,
            n_S,
            viewing_azimu_b_S_,
            viewing_polar_a_S_,
        ) = pm_template_3(
            0*flag_verbose,
            l_max_max,
            n_k_p_r,
            torch.reshape(a_k_Y_true_yk__,mtr((n_y_max,n_k_p_r))),
            t_eq_d_double/np.maximum(1e-12,k_p_r_max),
            -1,
            n_w_max,
        )[:5];
        S_k_p_wkS__ = torch.reshape(S_k_p_wkS__,mtr((n_w_max*n_k_p_r,n_S)));
        tmp_t = toc(tmp_t);
        if (flag_verbose>0): disp(sprintf(' %% pm_template_3 (n_S %d): %0.3fs',n_S,tmp_t)); #end;
        #%%%%%%%%;

        #%%%%%%%%;
        euler_polar_a_Mi__ = torch.zeros(mtr((n_M,n_iteration+1))).to(dtype=torch.float32);
        euler_azimu_b_Mi__ = torch.zeros(mtr((n_M,n_iteration+1))).to(dtype=torch.float32);
        euler_gamma_z_Mi__ = torch.zeros(mtr((n_M,n_iteration+1))).to(dtype=torch.float32);
        image_delta_x_acc_Mi__ = torch.zeros(mtr((n_M,n_iteration+1))).to(dtype=torch.float32);
        image_delta_y_acc_Mi__ = torch.zeros(mtr((n_M,n_iteration+1))).to(dtype=torch.float32);
        image_delta_x_upd_Mi__ = torch.zeros(mtr((n_M,n_iteration+1))).to(dtype=torch.float32);
        image_delta_y_upd_Mi__ = torch.zeros(mtr((n_M,n_iteration+1))).to(dtype=torch.float32);
        flag_image_delta_upd_Mi__ = torch.zeros(mtr((n_M,n_iteration))).to(dtype=torch.int32);
        image_I_value_Mi__ = torch.ones(mtr((n_M,n_iteration))).to(dtype=torch.float32);
        image_X_value_Mi__ = torch.zeros(mtr((n_M,n_iteration))).to(dtype=torch.float32);
        image_S_index_Mi__ = torch.zeros(mtr((n_M,n_iteration))).to(dtype=torch.int32);
        a_k_Y_reco_yki__ = torch.zeros(mtr((n_y_sum,n_iteration))).to(dtype=torch.complex64);
        corr_a_k_Y_i_ = torch.zeros(n_iteration).to(dtype=torch.complex64);
        X_SMi___ = torch.zeros(mtr((n_S,n_M,n_iteration))).to(dtype=torch.float32);
        delta_x_SMi___ = torch.zeros(mtr((n_S,n_M,n_iteration))).to(dtype=torch.float32);
        delta_y_SMi___ = torch.zeros(mtr((n_S,n_M,n_iteration))).to(dtype=torch.float32);
        gamma_z_SMi___ = torch.zeros(mtr((n_S,n_M,n_iteration))).to(dtype=torch.float32);
        I_value_SMi___ = torch.ones(mtr((n_S,n_M,n_iteration))).to(dtype=torch.float32);
        #%%%%%%%%;
        #% initialize current euler-angles randomly. ;
        #%%%%%%%%;
        rng(rseed);
        if not isempty(euler_polar_a_init_M_): euler_polar_a_M_ = euler_polar_a_init_M_.detach().clone(); #end;
        if isempty(euler_polar_a_init_M_): euler_polar_a_M_ = 1*pi*torch.rand(n_M).to(dtype=torch.float32); #end;
        if not isempty(euler_azimu_b_init_M_): euler_azimu_b_M_ = euler_azimu_b_init_M_.detach().clone(); #end;
        if isempty(euler_azimu_b_init_M_): euler_azimu_b_M_ = 2*pi*torch.rand(n_M).to(dtype=torch.float32); #end;
        if not isempty(euler_gamma_z_init_M_): euler_gamma_z_M_ = euler_gamma_z_init_M_.detach().clone(); #end;
        if isempty(euler_gamma_z_init_M_): euler_gamma_z_M_ = 2*pi*torch.rand(n_M).to(dtype=torch.float32); #end;
        image_delta_x_acc_M_ = torch.zeros(n_M).to(dtype=torch.float32); #%<-- accumulated displacement (i.e., current image center). ;
        if not isempty(image_delta_x_init_M_): image_delta_x_acc_M_ = image_delta_x_init_M_.detach().clone().to(dtype=torch.float32); #end;
        image_delta_y_acc_M_ = torch.zeros(n_M).to(dtype=torch.float32); #%<-- accumulated displacement (i.e., current image center). ;
        if not isempty(image_delta_y_init_M_): image_delta_y_acc_M_ = image_delta_y_init_M_.detach().clone().to(dtype=torch.float32); #end;
        image_delta_x_upd_M_ = torch.zeros(n_M).to(dtype=torch.float32); #%<-- update to displacement (i.e., current image shift). ;
        image_delta_y_upd_M_ = torch.zeros(n_M).to(dtype=torch.float32); #%<-- update to displacement (i.e., current image shift). ;
        image_delta_x_bit_M_ = torch.zeros(n_M).to(dtype=torch.float32); #%<-- increment to displacement update (calculated each iteration). ;
        image_delta_y_bit_M_ = torch.zeros(n_M).to(dtype=torch.float32); #%<-- increment to displacement update (calculated each iteration). ;
        flag_image_delta_upd_M_ = torch.ones(n_M).to(dtype=torch.int32); #%<-- flag identifying principal-images that need to be recalculated. ;
        image_I_value_M_ = torch.ones(n_M).to(dtype=torch.float32);
        #%%%%%%%%;
        M_pert_k_p_wkM__ = M_orig_k_p_wkM__.detach().clone() ; #<-- note copy. ;
        #%%%%%%%%;
        flag_precompute_UX_CTF_S_k_q_wnS__ = 1;
        flag_precompute_UX_CTF_S_l2_S_ = 1;

        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;
        for niteration in range(n_iteration):
        #%%%%%%%%;
            #%%%%%%%%;
            #% Construct M_pert_k_p_wkM__ while taking into account the translations. ;
            #%%%%%%%%;
            index_nM_to_update_ = efind(flag_image_delta_upd_M_); tmp_n_M = numel(index_nM_to_update_);
            if (flag_verbose>0): disp(sprintf(' %% updating M_pert_k_q_wkM__ for tmp_n_M %d/%d images',tmp_n_M,n_M)); #end;
            tmp_t = tic();
            for tmp_nM in range(tmp_n_M):
                tmp_i8_index_lhs_wkM_ = matlab_index_2d_0(n_w_sum,':',n_M,index_nM_to_update_[tmp_nM].item());
                tmp_i8_index_rhs_wkM_ = matlab_index_2d_0(n_w_sum,':',n_M,index_nM_to_update_[tmp_nM].item());
                M_pert_k_p_wkM__ = matlab_assign_lhs_from_rhs_(M_pert_k_p_wkM__,tmp_i8_index_lhs_wkM_, transf_p_to_p(n_k_p_r,k_p_r_,n_w_,n_w_sum,M_orig_k_p_wkM__.ravel()[tmp_i8_index_rhs_wkM_],+image_delta_x_acc_M_[index_nM_to_update_[tmp_nM].item()].item(),+image_delta_y_acc_M_[index_nM_to_update_[tmp_nM].item()].item()));
            #end;%for tmp_nM=0:tmp_n_M-1;
            tmp_t = toc(tmp_t);
            if (flag_verbose>0): disp(sprintf(' %% M_pert_k_q_wkM__: %0.3fs',tmp_t)); #end;
            parameter = parameter_timing_update(parameter,'M_pert_k_p_wkM__',tmp_t);
            #%%%%%%%%;
            tmp_t = tic();
            a_k_Y_reco_yk_ = qbp_uniform_over_n_k_p_r_10(
                qbp_eps,
                n_k_p_r,
                k_p_r_,
                l_max_,
                n_w_,
                n_M,
                M_orig_k_p_wkM__,
                index_nCTF_from_nM_,
                CTF_k_p_r_kC__,
                euler_polar_a_M_,
                euler_azimu_b_M_,
                euler_gamma_z_M_,
                +image_delta_x_acc_M_+image_delta_x_upd_M_,
                +image_delta_y_acc_M_+image_delta_y_upd_M_,
            )[0];
            tmp_t = toc(tmp_t);
            if (flag_verbose>0): disp(sprintf(' %% a_k_Y_reco_yk_: %0.3fs',tmp_t)); #end;
            parameter = parameter_timing_update(parameter,'qbp_uniform_over_n_k_p_r_10',tmp_t);
            #%%%%%%%%;
            #% Now correlate the two models. ;
            #%%%%%%%%;
            tmp_verbose = 0;
            (
                _,
                corr_a_k_Y,
            ) = register_spharm_to_spharm_3(
                tmp_verbose,
                n_k_p_r,
                k_p_r_,
                weight_3d_k_p_r_,
                l_max_,
                a_k_Y_true_yk_,
                a_k_Y_reco_yk_,
            )[:2];
            if (flag_verbose>0): disp(sprintf(' %% niteration %d/%d, correlation %0.4f',niteration,n_iteration,np.real(corr_a_k_Y))); #end;
            #%%%%%%%%;
            #% Now store image-parameters. ;
            #%%%%%%%%;
            a_k_Y_reco_yki__[niteration,:] = a_k_Y_reco_yk_;
            corr_a_k_Y_i_[niteration] = corr_a_k_Y;
            euler_polar_a_Mi__[niteration,:] = euler_polar_a_M_;
            euler_azimu_b_Mi__[niteration,:] = euler_azimu_b_M_;
            euler_gamma_z_Mi__[niteration,:] = euler_gamma_z_M_;
            image_delta_x_acc_Mi__[niteration,:] = image_delta_x_acc_M_;
            image_delta_y_acc_Mi__[niteration,:] = image_delta_y_acc_M_;
            image_delta_x_upd_Mi__[niteration,:] = image_delta_x_upd_M_;
            image_delta_y_upd_Mi__[niteration,:] = image_delta_y_upd_M_;
            image_I_value_Mi__[niteration,:] = image_I_value_M_;
            #%%%%%%%%;
            #% Now, even if we are at the final iteration, align the principal images to the principal volume. ;
            #%%%%%%%%;
            if (niteration< n_iteration-1+1):
                #%%%%%%%%;
                #% Use given volume to align principal-images. ;
                #% Groups principal-images by cluster. ;
                #% Calculates principal-templates associated with each cluster. ;
                #% Uses precomputation for M and S. ;
                #% Batches images into batches of size n_M_per_Mbatch (default 24). ;
                #% Batches templates into batches of size n_S_per_Sbatch (default 24). ;
                #% Only stores the optimal translation for each principal-image. ;
                #%%%%%%%%;
                if 'M_k_q_wkM__' not in locals(): M_k_q_wkM__=None; #end;
                if 'UX_T_M_l2_dM__' not in locals(): UX_T_M_l2_dM__=None; #end;
                if 'UX_M_l2_M_' not in locals(): UX_M_l2_M_=None; #end;
                if 'svd_V_UX_M_lwnM____' not in locals(): svd_V_UX_M_lwnM____=None; #end;
                if 'UX_CTF_S_k_q_wnS__' not in locals(): UX_CTF_S_k_q_wnS__=None; #end;
                if 'UX_CTF_S_l2_S_' not in locals(): UX_CTF_S_l2_S_=None; #end;
                parameter['flag_verbose'] = int(np.maximum(0,flag_verbose-1));
                parameter['flag_precompute_M_k_q_wkM__'] = 1;
                parameter['flag_precompute_UX_T_M_l2_dM__'] = 1;
                parameter['flag_precompute_UX_M_l2_M_'] = 1;
                parameter['flag_precompute_svd_V_UX_M_lwnM____'] = 1;
                parameter['flag_precompute_UX_CTF_S_k_q_wnS__'] = flag_precompute_UX_CTF_S_k_q_wnS__;
                parameter['flag_precompute_UX_CTF_S_l2_S_'] = flag_precompute_UX_CTF_S_l2_S_;
                if parameter['flag_precompute_UX_CTF_S_k_q_wnS__']==0 and parameter['flag_precompute_UX_CTF_S_l2_S_']==0:
                    index_nS_to_update_ = torch.arange(0).to(dtype=torch.int32);
                else:
                    index_nS_to_update_ = torch.arange(n_S).to(dtype=torch.int32);
                    flag_precompute_UX_CTF_S_k_q_wnS__ = 0; #%<-- for next iteration. ;
                    flag_precompute_UX_CTF_S_l2_S_ = 0; #%<-- for next iteration. ;
                #end;%if;
                tmp_t = tic();
                (
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
                    svd_V_UX_M_lwnM____,
                    UX_CTF_S_k_q_wnS__,
                ) = tfpmh_Z_cluster_wrap_SM__14(
                    parameter,
                    n_k_p_r,
                    k_p_r_,
                    k_p_r_max,
                    n_w_,
                    weight_2d_k_p_r_,
                    weight_2d_k_p_wk_,
                    n_S,
                    S_k_p_wkS__,
                    n_CTF,
                    CTF_k_p_r_kC__,
                    index_nCTF_from_nM_,
                    n_M,
                    M_pert_k_p_wkM__,
                    n_cluster,
                    index_ncluster_from_nCTF_,
                    pm_n_UX_rank_c_,
                    pm_UX_knc___,
                    pm_X_weight_rc__,
                    FTK,
                    index_nM_to_update_,
                    M_k_q_wkM__,
                    UX_T_M_l2_dM__,
                    UX_M_l2_M_,
                    svd_V_UX_M_lwnM____,
                    index_nS_to_update_,
                    UX_CTF_S_k_q_wnS__,
                    UX_CTF_S_l2_S_,
                )[:16];
                parameter['flag_verbose'] = flag_verbose;
                tmp_t = toc(tmp_t);
                if (flag_verbose>0): disp(sprintf(' %% tfpmh_Z_cluster_wrap_SM__14: %0.3fs',tmp_t)); #end;
                parameter = parameter_timing_update(parameter,'tfpmh_Z_cluster_wrap_SM__14',tmp_t);
                #%%%%%%%%;
                #% Use current correlations to update current euler-angles. ;
                #%%%%%%%%;
                tmp_t = tic();
                parameter['flag_verbose'] = int(np.maximum(0,flag_verbose-1));
                (
                    parameter,
                    euler_polar_a_M_,
                    euler_azimu_b_M_,
                    euler_gamma_z_M_,
                    image_delta_x_bit_M_,
                    image_delta_y_bit_M_,
                    image_I_value_M_,
                    image_X_value_M_,
                    image_S_index_M_,
                ) = tfpmh_MS_vs_SM_2(
                    parameter,
                    n_w_max,
                    n_S,
                    viewing_azimu_b_S_,
                    viewing_polar_a_S_,
                    n_M,
                    X_SM__,
                    delta_x_SM__,
                    delta_y_SM__,
                    gamma_z_SM__,
                )[:9];
                parameter['flag_verbose'] = flag_verbose;
                tmp_t = toc(tmp_t);
                if (flag_verbose>0): disp(sprintf(' %% MS_vs_SM: update euler_polar_a_M_ euler_azimu_b_M_ euler_gamma_z_M_ : %0.3fs',tmp_t)); #end;
                parameter = parameter_timing_update(parameter,'tfpmh_MS_vs_SM_2',tmp_t);
                #%%%%%%%%;
                #% Update translations. ;
                #% Here we update image_delta_x_upd_M_ and image_delta_y_upd_M_ by image_delta_x_bit_M_ and image_delta_y_bit_M_. ;
                #% This update by itself will not trigger a re-precomputation. ;
                #% However, when image_delta_r_upd_norm_M_ is sufficiently large (i.e., greater than delta_r_upd_threshold), ;
                #% then we bake in the displacement (stored in image_delta_x_acc_M_ and image_delta_y_acc_M_), ;
                #% and then flag those images for re-precomputation. ;
                #%%%%%%%%;
                if 'delta_r_upd_threshold' not in parameter: parameter['delta_r_upd_threshold'] = 0.0*delta_r_max; #end; %<-- parameter_bookmark. ;
                delta_r_upd_threshold = parameter['delta_r_upd_threshold'];
                image_delta_x_upd_M_ = 0*image_delta_x_upd_M_ + 1*image_delta_x_bit_M_;
                image_delta_y_upd_M_ = 0*image_delta_y_upd_M_ + 1*image_delta_y_bit_M_;
                image_delta_r_upd_prenorm_M_ = torch.sqrt(image_delta_x_upd_M_**2 + image_delta_y_upd_M_**2);
                image_delta_x_tot_M_ = image_delta_x_acc_M_ + image_delta_x_upd_M_;
                image_delta_y_tot_M_ = image_delta_y_acc_M_ + image_delta_y_upd_M_;
                image_delta_r_tot_M_ = torch.sqrt(image_delta_x_tot_M_**2 + image_delta_y_tot_M_**2);
                image_delta_x_nrm_M_ = image_delta_x_tot_M_.detach().clone(); #<-- note copy. ;
                image_delta_y_nrm_M_ = image_delta_y_tot_M_.detach().clone(); #<-- note copy. ;
                tmp_index_ = efind(image_delta_r_tot_M_> delta_r_upb);
                if (numel(tmp_index_)> 0):
                    if (flag_verbose>0): disp(sprintf(' %% normalizing %d/%d image displacements',numel(tmp_index_),n_M)); #end;
                    image_delta_x_nrm_M_[tmp_index_] = image_delta_x_tot_M_[tmp_index_]*delta_r_upb/torch.maximum(tolerance_machine_1_,image_delta_r_tot_M_[tmp_index_]);
                    image_delta_y_nrm_M_[tmp_index_] = image_delta_y_tot_M_[tmp_index_]*delta_r_upb/torch.maximum(tolerance_machine_1_,image_delta_r_tot_M_[tmp_index_]);
                    image_delta_x_upd_M_[tmp_index_] = image_delta_x_nrm_M_[tmp_index_] - image_delta_x_acc_M_[tmp_index_];
                    image_delta_y_upd_M_[tmp_index_] = image_delta_y_nrm_M_[tmp_index_] - image_delta_y_acc_M_[tmp_index_];
                #end;%if (numel(tmp_index_)> 0);
                flag_image_delta_upd_M_ = torch.zeros(n_M).to(dtype=torch.int32);
                image_delta_r_upd_posnorm_M_ = torch.sqrt(image_delta_x_upd_M_**2 + image_delta_y_upd_M_**2);
                tmp_index_ = efind( torch.logical_or( image_delta_r_upd_prenorm_M_>=delta_r_upd_threshold , image_delta_r_upd_posnorm_M_>=delta_r_upd_threshold ) );
                if (numel(tmp_index_)> 0):
                    if (flag_verbose>0): disp(sprintf(' %% accumulating %d/%d image displacements',numel(tmp_index_),n_M)); #end;
                    flag_image_delta_upd_M_[tmp_index_] = 1;
                    image_delta_x_acc_M_[tmp_index_] = image_delta_x_acc_M_[tmp_index_] + image_delta_x_upd_M_[tmp_index_];
                    image_delta_y_acc_M_[tmp_index_] = image_delta_y_acc_M_[tmp_index_] + image_delta_y_upd_M_[tmp_index_];
                    image_delta_x_upd_M_[tmp_index_] = 0;
                    image_delta_y_upd_M_[tmp_index_] = 0;
                #end;%if (numel(tmp_index_)> 0);
                #%%%%%%%%;
                flag_image_delta_upd_Mi__[niteration,:] = flag_image_delta_upd_M_; #%<-- these actually refer to end of iteration. ;
                image_X_value_Mi__[niteration,:] = image_X_value_M_; #%<-- these actually refer to end of iteration. ;
                image_S_index_Mi__[niteration,:] = image_S_index_M_; #%<-- these actually refer to end of iteration. ;
                X_SMi___[niteration,:,:] = X_SM__; #%<-- these actually refer to end of iteration. ;
                delta_x_SMi___[niteration,:,:] = delta_x_SM__; #%<-- these actually refer to end of iteration. ;
                delta_y_SMi___[niteration,:,:] = delta_y_SM__; #%<-- these actually refer to end of iteration. ;
                gamma_z_SMi___[niteration,:,:] = gamma_z_SM__; #%<-- these actually refer to end of iteration. ;
                I_value_SMi___[niteration,:,:] = torch.ones(mtr((n_S,n_M))).to(dtype=torch.float32); #%I_value_SM__; %<-- these actually refer to end of iteration. ;
                #%%%%%%%%;
                euler_polar_a_Mi__[niteration+1,:] = euler_polar_a_M_;
                euler_azimu_b_Mi__[niteration+1,:] = euler_azimu_b_M_;
                euler_gamma_z_Mi__[niteration+1,:] = euler_gamma_z_M_;
                image_delta_x_acc_Mi__[niteration+1,:] = image_delta_x_acc_M_;
                image_delta_y_acc_Mi__[niteration+1,:] = image_delta_y_acc_M_;
                image_delta_x_upd_Mi__[niteration+1,:] = image_delta_x_upd_M_;
                image_delta_y_upd_Mi__[niteration+1,:] = image_delta_y_upd_M_;
                #%%%%%%%%;
            #end;%if (niteration< n_iteration-1+1);
            #%%%%%%%%;
        #end;%for niteration=0:n_iteration-1;
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;
    #end;%if ~flag_found;
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;

    flag_found=0;
    if not isempty(parameter['fname_align_a_k_Y_pre']):
        tmp_fname_mat = sprintf('%s.mat',parameter['fname_align_a_k_Y_pre']);
        if not matlab_exist_file(tmp_fname_mat):
            if (flag_verbose>0): disp(sprintf(' %% %s not found, creating',tmp_fname_mat)); #end;
            matlab_save(
                fname_mat=tmp_fname_mat,
                dictionary_original= {
                    "a_k_Y_reco_yki__":a_k_Y_reco_yki__,
                    "corr_a_k_Y_i_":corr_a_k_Y_i_,
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
                    "n_S":n_S,
                    "viewing_azimu_b_S_":viewing_azimu_b_S_,
                    "viewing_polar_a_S_":viewing_polar_a_S_,
                    "X_SMi___":X_SMi___,
                    "delta_x_SMi___":delta_x_SMi___,
                    "delta_y_SMi___":delta_y_SMi___,
                    "gamma_z_SMi___":gamma_z_SMi___,
                    "I_value_SMi___":I_value_SMi___,
                },
            );
        #end;%if (~exist(tmp_fname_mat,'file'));
        flag_found=1;
    #end;%if (~isempty(parameter.fname_align_a_k_Y_pre));

    flag_true = (
        (not isempty(euler_polar_a_true_M_))
        and (not isempty(euler_azimu_b_true_M_))
        and (not isempty(euler_gamma_z_true_M_))
        and (not isempty(image_delta_x_true_M_))
        and (not isempty(image_delta_y_true_M_))
    );

    if (flag_verbose>0): disp(sprintf(' %% [finished %s]',str_thisfunction)); #end;

    return(
        parameter,
        a_k_Y_reco_yki__,
        corr_a_k_Y_i_,
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
        n_S,
        viewing_azimu_b_S_,
        viewing_polar_a_S_,
        X_SMi___,
        delta_x_SMi___,
        delta_y_SMi___,
        gamma_z_SMi___,
        I_value_SMi___,
    );
