from dir_empm.dir_matlab_macros import * ;
from dir_empm.tfh_FTK_4 import tfh_FTK_4 ;
from dir_empm.transf_p_to_p import transf_p_to_p ;
from dir_empm.qbp_uniform_over_n_k_p_r_10 import qbp_uniform_over_n_k_p_r_10 ;
from dir_empm.spharm_normalize_1 import spharm_normalize_1 ;
from dir_empm.local_yk__from_yk_ import local_yk__from_yk_ ;
from dir_empm.pm_template_3 import pm_template_3 ;
from dir_empm.tfpmh_Z_cluster_wrap_SM__14 import tfpmh_Z_cluster_wrap_SM__14 ;
from dir_empm.tfpmh_MS_vs_SM_2 import tfpmh_MS_vs_SM_2 ;

def tfpmut_6(
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
        M_orig_k_p_wkM__=None,
        n_cluster=None,
        index_ncluster_from_nCTF_=None,
        pm_n_UX_rank_c_=None,
        pm_UX_knc___=None,
        pm_X_weight_rc__=None,
        FTK=None,
        euler_polar_a_M_=None,
        euler_azimu_b_M_=None,
        euler_gamma_z_M_=None,
        image_delta_x_acc_M_=None,
        image_delta_y_acc_M_=None,
        image_delta_x_upd_M_=None,
        image_delta_y_upd_M_=None,
        flag_image_delta_upd_M_=None,
        image_I_value_M_=None,
):

    #%%%%%%%%;

    str_thisfunction = 'tfpmut_6';
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
    if 'n_iteration' not in parameter: parameter['n_iteration'] = 1; #end; %<-- parameter_bookmark. ;
    n_iteration = parameter['n_iteration'];
    if 'flag_alternate_MS_vs_SM' not in parameter: parameter['flag_alternate_MS_vs_SM'] = 1; #end; %<-- parameter_bookmark. ;
    flag_alternate_MS_vs_SM = parameter['flag_alternate_MS_vs_SM'];
    if 'flag_MS_vs_SM' not in parameter: parameter['flag_MS_vs_SM'] = 1; #end; %<-- parameter_bookmark. ;
    flag_MS_vs_SM = parameter['flag_MS_vs_SM'];
    if 'order_limit_MS' not in parameter: parameter['order_limit_MS'] = -1; #end; %<-- parameter_bookmark. ;
    order_limit_MS = parameter['order_limit_MS'];
    if 'delta_r_max' not in parameter: parameter['delta_r_max'] = 0.1; #end; %<-- parameter_bookmark. ;
    delta_r_max = parameter['delta_r_max'];
    if 'delta_r_upb' not in parameter: parameter['delta_r_upb'] = 2*delta_r_max; #end; %<-- parameter_bookmark. ;
    delta_r_upb = parameter['delta_r_upb'];
    if 'n_delta_v_requested' not in parameter: parameter['n_delta_v_requested'] = 0; #end; %<-- parameter_bookmark. ;
    n_delta_v_requested = parameter['n_delta_v_requested'];
    if 'template_viewing_k_eq_d' not in parameter: parameter['template_viewing_k_eq_d'] = 1.0/np.maximum(1e-12,k_p_r_max); #end; %<-- parameter_bookmark. ;
    template_viewing_k_eq_d = parameter['template_viewing_k_eq_d'];
    if 'flag_save_stage' not in parameter: parameter['flag_save_stage'] = 0; #end; %<-- parameter_bookmark. ;
    flag_save_stage = parameter['flag_save_stage'];
    #%%%%%%%%;
    svd_eps = tolerance_master;

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
    n_cluster = 1+int(torch.max(index_ncluster_from_nCTF_).item());
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
    tmp_t=tic();
    if (size(CTF_k_p_wkC__,0)==n_k_p_r):
        n_CTF = size(CTF_k_p_wkC__,1);
        CTF_k_p_r_kC__ = CTF_k_p_wkC__;
        CTF_k_p_wkC__ = torch.reshape(torch.tile(torch.reshape(CTF_k_p_r_kC__,mtr((1,n_k_p_r,n_CTF))),mtr((n_w_max,n_1,n_1))),mtr((n_w_sum,n_CTF)));
    #end;%if (size(CTF_k_p_wkC__,1+0)==n_k_p_r);
    #%%%%%%%%;
    CTF_k_p_r_kC__ = torch.reshape(torch.mean(torch.reshape(CTF_k_p_wkC__,mtr((n_w_max,n_k_p_r,n_CTF))),2-0),mtr((n_k_p_r,n_CTF)));
    tmp_i8_index_rhs_ = matlab_index_2d_0(n_w_sum,':',n_CTF,index_nCTF_from_nM_);
    CTF_k_p_wkM__ = torch.reshape(CTF_k_p_wkC__.ravel()[tmp_i8_index_rhs_],mtr((n_w_sum,n_M)));
    #%%%%%%%%;
    #% then calculate average CTFs for each cluster. ;
    #%%%%%%%%;
    CTF_k_p_r_xavg_kc__ = torch.zeros(mtr((n_k_p_r,n_cluster)));
    for ncluster in range(n_cluster):
        index_nM_from_ncluster_ = index_nM_from_ncluster__[ncluster];
        n_index_nM_from_ncluster = int(n_index_nM_from_ncluster_[ncluster].item());
        assert(n_index_nM_from_ncluster==numel(index_nM_from_ncluster_));
        tmp_i8_index_rhs_ = matlab_index_2d_0(n_w_sum,':',n_CTF,index_nCTF_from_nM_[index_nM_from_ncluster_]);
        CTF_k_p_r_xavg_k_ = torch.mean(torch.reshape(CTF_k_p_wkC__.ravel()[tmp_i8_index_rhs_],mtr((n_w_max,n_k_p_r,n_index_nM_from_ncluster))),[2-0,2-2]).ravel(); assert(numel(CTF_k_p_r_xavg_k_)==n_k_p_r);
        CTF_k_p_r_xavg_kc__[ncluster,:] = CTF_k_p_r_xavg_k_;
    #end;%for ncluster=0:n_cluster-1;
    tmp_t = toc(tmp_t);
    if (flag_verbose>1): disp(sprintf(' %% CTF_k_p_r_xavg_kc__: %0.3fs',tmp_t)); #end;
    parameter = parameter_timing_update(parameter,sprintf('%s: CTF_k_p_r_xavg_kc__',str_thisfunction),tmp_t);

    if isempty(FTK):
        tmp_t = tic();
        r8_delta_r_max = float(delta_r_max);
        r8_svd_eps = float(svd_eps);
        parameter_FTK = parameter;
        parameter_FTK['r8_delta_r_max'] = r8_delta_r_max;
        parameter_FTK['r8_svd_eps'] = r8_svd_eps;
        parameter_FTK['n_delta_v_requested'] = n_delta_v_requested;
        parameter_FTK,FTK = tfh_FTK_4(parameter_FTK,n_k_p_r,k_p_r_,k_p_r_max);
        tmp_t = toc(tmp_t);
        if (flag_verbose>1): disp(sprintf(' %% FTK: %0.3fs',tmp_t)); #end;
        parameter = parameter_timing_update(parameter,sprintf('%s: tfh_FTK_4',str_thisfunction),tmp_t);
    #end;%if isempty(FTK);
    assert(FTK['r8_svd_d_max']>=delta_r_max);
    assert(FTK['n_delta_v']>=n_delta_v_requested);
    n_delta_v = FTK['n_delta_v'];
    n_svd_l = FTK['n_svd_l'];

    euler_polar_a_Mi__ = torch.zeros(mtr((n_M,n_iteration))).to(dtype=torch.float32);
    euler_azimu_b_Mi__ = torch.zeros(mtr((n_M,n_iteration))).to(dtype=torch.float32);
    euler_gamma_z_Mi__ = torch.zeros(mtr((n_M,n_iteration))).to(dtype=torch.float32);
    image_delta_x_acc_Mi__ = torch.zeros(mtr((n_M,n_iteration))).to(dtype=torch.float32);
    image_delta_y_acc_Mi__ = torch.zeros(mtr((n_M,n_iteration))).to(dtype=torch.float32);
    image_delta_x_upd_Mi__ = torch.zeros(mtr((n_M,n_iteration))).to(dtype=torch.float32);
    image_delta_y_upd_Mi__ = torch.zeros(mtr((n_M,n_iteration))).to(dtype=torch.float32);
    flag_image_delta_upd_Mi__ = torch.zeros(mtr((n_M,n_iteration))).to(dtype=torch.int32);
    image_I_value_Mi__ = torch.zeros(mtr((n_M,n_iteration))).to(dtype=torch.float32);
    image_X_value_Mi__ = torch.zeros(mtr((n_M,n_iteration))).to(dtype=torch.float32);
    image_S_index_Mi__ = torch.zeros(mtr((n_M,n_iteration))).to(dtype=torch.int32);
    a_k_Y_reco_yki__ = torch.zeros(mtr((n_y_sum,n_iteration))).to(dtype=torch.complex64);

    #%%%%%%%%;
    #% initialize current euler-angles randomly. ;
    #%%%%%%%%;
    rng(rseed);
    if isempty(euler_polar_a_M_): euler_polar_a_M_ = 1*pi*torch.rand(n_M).to(dtype=torch.float32); #end;
    if isempty(euler_azimu_b_M_): euler_azimu_b_M_ = 2*pi*torch.rand(n_M).to(dtype=torch.float32); #end;
    if isempty(euler_gamma_z_M_): euler_gamma_z_M_ = 2*pi*torch.rand(n_M).to(dtype=torch.float32); #end;
    if isempty(image_delta_x_acc_M_): image_delta_x_acc_M_ = torch.zeros(n_M).to(dtype=torch.float32); #end; %<-- accumulated displacement (i.e., current image center). ;
    if isempty(image_delta_y_acc_M_): image_delta_y_acc_M_ = torch.zeros(n_M).to(dtype=torch.float32); #end; %<-- accumulated displacement (i.e., current image center). ;
    if isempty(image_delta_x_upd_M_): image_delta_x_upd_M_ = torch.zeros(n_M).to(dtype=torch.float32); #end; %<-- update to displacement (i.e., current image shift). ;
    if isempty(image_delta_y_upd_M_): image_delta_y_upd_M_ = torch.zeros(n_M).to(dtype=torch.float32); #end; %<-- update to displacement (i.e., current image shift). ;
    image_delta_x_bit_M_ = torch.zeros(n_M).to(dtype=torch.float32); #%<-- increment to displacement update (calculated each iteration). ;
    image_delta_y_bit_M_ = torch.zeros(n_M).to(dtype=torch.float32); #%<-- increment to displacement update (calculated each iteration). ;
    if isempty(flag_image_delta_upd_M_): flag_image_delta_upd_M_ = torch.ones(n_M).to(dtype=torch.int32); #end; %<-- flag identifying principal-images that need to be recalculated. ;
    if isempty(image_I_value_M_): image_I_value_M_ = torch.ones(n_M).to(dtype=torch.float32); #end;
    #%%%%%%%%;
    M_pert_k_p_wkM__ = M_orig_k_p_wkM__.clone().detach() ; #<-- necessary to avoid clobbering M_orig_k_p_wkM__. ;
    #%%%%%%%%;
    flag_precompute_UX_CTF_S_k_q_wnS__ = 1;
    flag_precompute_UX_CTF_S_l2_S_ = 1;

    #%%%%%%%%;
    if (flag_save_stage>2) and ('fname_pre' in parameter):
        fname_mat = sprintf('%s_stage_4.mat',parameter['fname_pre']);
        disp(sprintf(' %% writing %s',fname_mat));
        matlab_save(
            fname_mat=fname_mat,
            dictionary_original= {
                "flag_verbose":flag_verbose,
                "tolerance_master":tolerance_master,
                "flag_gpu":flag_gpu,
                "rseed":rseed,
                "n_iteration":n_iteration,
                "n_delta_v_requested":n_delta_v_requested,
                "flag_MS_vs_SM":flag_MS_vs_SM,
                "order_limit_MS":order_limit_MS,
                "delta_r_max":delta_r_max,
                "delta_r_upb":delta_r_upb,
                "template_viewing_k_eq_d":template_viewing_k_eq_d,
                "flag_alternate_MS_vs_SM":flag_alternate_MS_vs_SM,
                "svd_eps":svd_eps,
                "n_w_":n_w_,
                "n_w_sum":n_w_sum,
                "n_w_max":n_w_max,
                "n_w_csum_":n_w_csum_,
                "l_max_max":l_max_max,
                "n_y_":n_y_,
                "n_y_max":n_y_max,
                "n_y_sum":n_y_sum,
                "n_y_csum_":n_y_csum_,
                "n_cluster":n_cluster,
                "index_ncluster_from_nM_":index_ncluster_from_nM_,
                "index_nM_from_ncluster__":index_nM_from_ncluster__,
                "n_index_nM_from_ncluster_":n_index_nM_from_ncluster_,
                "CTF_k_p_wkC__":CTF_k_p_wkC__,
                "n_CTF":n_CTF,
                "CTF_k_p_r_kC__":CTF_k_p_r_kC__,
                "CTF_k_p_wkM__":CTF_k_p_wkM__,
                "CTF_k_p_r_xavg_kc__":CTF_k_p_r_xavg_kc__,
                "n_delta_v":n_delta_v,
                "n_svd_l":n_svd_l,
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
                "a_k_Y_reco_yki__":a_k_Y_reco_yki__,
                "rseed":rseed,
                "euler_polar_a_M_":euler_polar_a_M_,
                "euler_azimu_b_M_":euler_azimu_b_M_,
                "euler_gamma_z_M_":euler_gamma_z_M_,
                "image_delta_x_acc_M_":image_delta_x_acc_M_,
                "image_delta_y_acc_M_":image_delta_y_acc_M_,
                "image_delta_x_upd_M_":image_delta_x_upd_M_,
                "image_delta_y_upd_M_":image_delta_y_upd_M_,
                "image_delta_x_bit_M_":image_delta_x_bit_M_,
                "image_delta_y_bit_M_":image_delta_y_bit_M_,
                "flag_image_delta_upd_M_":flag_image_delta_upd_M_,
                "image_I_value_M_":image_I_value_M_,
                "M_pert_k_p_wkM__":M_pert_k_p_wkM__,
                "flag_precompute_UX_CTF_S_k_q_wnS__":flag_precompute_UX_CTF_S_k_q_wnS__,
                "flag_precompute_UX_CTF_S_l2_S_":flag_precompute_UX_CTF_S_l2_S_,
            },
        );
    #end;%if ( isfield(parameter,'fname_pre'));
    #%%%%%%%%;

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;
    for niteration in range(n_iteration):
    #%%%%%%%%;
        if (flag_verbose>0): disp(sprintf(' %% niteration %.2d/%.2d',niteration,n_iteration)); #end;
        if flag_alternate_MS_vs_SM==0:
            parameter['flag_MS_vs_SM'] = (niteration< np.floor(n_iteration/2)); #%<-- 1,1,1,...,0,0,0,... ;
        #end;%if flag_alternate_MS_vs_SM==0;
        if flag_alternate_MS_vs_SM==1:
            parameter['flag_MS_vs_SM'] = int(np.mod(1+niteration,2)); #%<-- 1,0,1,0,1,0,... ;
        #end;%if flag_alternate_MS_vs_SM==1;
        flag_MS_vs_SM = parameter['flag_MS_vs_SM'];
        #%%%%%%%%;
        #% Construct M_pert_k_p_wkM__ while taking into account the translations. ;
        #%%%%%%%%;
        index_nM_to_update_ = efind(flag_image_delta_upd_M_); tmp_n_M = numel(index_nM_to_update_);
        if (flag_verbose>0): disp(sprintf(' %% %% updating M_pert_k_q_wkM__ for tmp_n_M %d/%d images',tmp_n_M,n_M)); #end;
        tmp_t = tic();
        r'''
        tmp_i8_index_lhs_wkM_ = matlab_index_2d_0(n_w_sum,':',n_M,index_nM_to_update_);
        tmp_i8_index_rhs_wkM_ = matlab_index_2d_0(n_w_sum,':',n_M,index_nM_to_update_);
        M_pert_k_p_wkM__.ravel()[tmp_i8_index_lhs_wkM_] = transf_p_to_p(
            n_k_p_r,
            k_p_r_,
            n_w_,
            n_w_sum,
            torch.reshape(M_orig_k_p_wkM__.ravel()[tmp_i8_index_rhs_wkM_],mtr((n_w_sum,tmp_n_M))),
            +image_delta_x_acc_M_.ravel()[index_nM_to_update_],
            +image_delta_y_acc_M_.ravel()[index_nM_to_update_],
        )[0];
        '''
        for tmp_nM in range(tmp_n_M):
            tmp_i8_index_lhs_wkM_ = matlab_index_2d_0(n_w_sum,':',n_M,index_nM_to_update_[tmp_nM].item());
            tmp_i8_index_rhs_wkM_ = matlab_index_2d_0(n_w_sum,':',n_M,index_nM_to_update_[tmp_nM].item());
            M_pert_k_p_wkM__.ravel()[tmp_i8_index_lhs_wkM_] = transf_p_to_p(
                n_k_p_r,
                k_p_r_,
                n_w_,
                n_w_sum,
                M_orig_k_p_wkM__.ravel()[tmp_i8_index_rhs_wkM_],
                +image_delta_x_acc_M_[index_nM_to_update_[tmp_nM].item()].item(),
                +image_delta_y_acc_M_[index_nM_to_update_[tmp_nM].item()].item(),
            );
        #end;%
        tmp_t = toc(tmp_t); 
        if (flag_verbose>0): disp(sprintf(' %% %% M_pert_k_q_wkM__: %0.3fs',tmp_t)); #end;
        parameter = parameter_timing_update(parameter,sprintf('%s: M_pert_k_p_wkM__',str_thisfunction),tmp_t);

        #%%%%%%%%;
        if (flag_save_stage>2) and ('fname_pre' in parameter):
            fname_mat = sprintf('%s_stage_5_%d.mat',parameter['fname_pre'],niteration);
            disp(sprintf(' %% writing %s',fname_mat));
            matlab_save(
                fname_mat=fname_mat,
                dictionary_original= {
                    "index_nM_to_update_":index_nM_to_update_,
                    "tmp_n_M":tmp_n_M,
                    "flag_MS_vs_SM":flag_MS_vs_SM,
                    "M_pert_k_p_wkM__":M_pert_k_p_wkM__,
                    "M_orig_k_p_wkM__":M_orig_k_p_wkM__,
                    "image_delta_x_acc_M_":image_delta_x_acc_M_,
                    "image_delta_y_acc_M_":image_delta_y_acc_M_,
                },
            );
        #end;%if ( isfield(parameter,'fname_pre'));
        #%%%%%%%%;

        #%%%%%%%%;
        #% use current euler-angles and displacements to solve for current model. ;
        #%%%%%%%%;
        tmp_t = tic();
        qbp_eps = tolerance_master;
        a_k_Y_reco_yk_ = qbp_uniform_over_n_k_p_r_10(
            qbp_eps,
            n_k_p_r,
            k_p_r_,
            l_max_,
            n_w_,
            n_M,
            M_orig_k_p_wkM__,
            index_nCTF_from_nM_,
            CTF_k_p_wkC__,
            euler_polar_a_M_,
            euler_azimu_b_M_,
            euler_gamma_z_M_,
            +image_delta_x_acc_M_+image_delta_x_upd_M_,
            +image_delta_y_acc_M_+image_delta_y_upd_M_,
        )[0];
        tmp_t = toc(tmp_t); 
        if (flag_verbose>0): disp(sprintf(' %% %% a_k_Y_reco_yk_: %0.3fs',tmp_t)); #end;
        parameter = parameter_timing_update(parameter,sprintf('%s: qbp_uniform_over_n_k_p_r_10',str_thisfunction),tmp_t);

        #%%%%%%%%;
        if (flag_save_stage>2) and ('fname_pre' in parameter):
            fname_mat = sprintf('%s_stage_6_%d.mat',parameter['fname_pre'],niteration);
            disp(sprintf(' %% writing %s',fname_mat));
            matlab_save(
                fname_mat=fname_mat,
                dictionary_original= {
                    "a_k_Y_reco_yk_":a_k_Y_reco_yk_,
                },
            );
        #end;%if ( isfield(parameter,'fname_pre'));
        #%%%%%%%%;

        #%%%%%%%%;
        #% Now normalize a_k_Y_reco_yk_. ;
        #% This step is necessary to prevent the intensity from diverging over successive iterations. ;
        #%%%%%%%%;
        a_k_Y_reco_yk_ = spharm_normalize_1(n_k_p_r,k_p_r_,weight_3d_k_p_r_,l_max_,a_k_Y_reco_yk_)[0];

        #%%%%%%%%;
        if (flag_save_stage>2) and ('fname_pre' in parameter):
            fname_mat = sprintf('%s_stage_7_%d.mat',parameter['fname_pre'],niteration);
            disp(sprintf(' %% writing %s',fname_mat));
            matlab_save(
                fname_mat=fname_mat,
                dictionary_original= {
                    "a_k_Y_reco_yk_":a_k_Y_reco_yk_,
                },
            );
        #end;%if ( isfield(parameter,'fname_pre'));
        #%%%%%%%%;

        #%%%%%%%%;
        #% Now store image-parameters. ;
        #%%%%%%%%;
        a_k_Y_reco_yki__[niteration,:] = a_k_Y_reco_yk_;
        euler_polar_a_Mi__[niteration,:] = euler_polar_a_M_;
        euler_azimu_b_Mi__[niteration,:] = euler_azimu_b_M_;
        euler_gamma_z_Mi__[niteration,:] = euler_gamma_z_M_;
        image_delta_x_acc_Mi__[niteration,:] = image_delta_x_acc_M_;
        image_delta_y_acc_Mi__[niteration,:] = image_delta_y_acc_M_;
        image_delta_x_upd_Mi__[niteration,:] = image_delta_x_upd_M_;
        image_delta_y_upd_Mi__[niteration,:] = image_delta_y_upd_M_;
        image_I_value_Mi__[niteration,:] = image_I_value_M_;
        #%%%%%%%%;
        #% Construct templates using the volume. ;
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
            torch.reshape(local_yk__from_yk_(n_k_p_r,l_max_,a_k_Y_reco_yk_),mtr((n_y_max,n_k_p_r))),
            template_viewing_k_eq_d,
            -1,
            n_w_max,
        )[:5];
        S_k_p_wkS__ = torch.reshape(S_k_p_wkS__,mtr((n_w_max*n_k_p_r,n_S)));
        tmp_t = toc(tmp_t); 
        if (flag_verbose>0): disp(sprintf(' %% %% pm_template_3 (n_S %d): %0.3fs',n_S,tmp_t)); #end;

        #%%%%%%%%;
        if (flag_save_stage>2) and ('fname_pre' in parameter):
            fname_mat = sprintf('%s_stage_8_%d.mat',parameter['fname_pre'],niteration);
            disp(sprintf(' %% writing %s',fname_mat));
            matlab_save(
                fname_mat=fname_mat,
                dictionary_original= {
                    "S_k_p_wkS__":S_k_p_wkS__,
                    "n_S":n_S,
                    "viewing_azimu_b_S_":viewing_azimu_b_S_,
                    "viewing_polar_a_S_":viewing_polar_a_S_,
                },
            );
        #end;%if ( isfield(parameter,'fname_pre'));
        #%%%%%%%%;

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
        parameter['flag_precompute_M_k_q_wkM__'] = 1*1;
        parameter['flag_precompute_UX_T_M_l2_dM__'] = 1*1;
        parameter['flag_precompute_UX_M_l2_M_'] = 1*1;
        parameter['flag_precompute_svd_V_UX_M_lwnM____'] = 1*1;
        parameter['flag_precompute_UX_CTF_S_k_q_wnS__'] = 1*flag_precompute_UX_CTF_S_k_q_wnS__;
        parameter['flag_precompute_UX_CTF_S_l2_S_'] = 1*flag_precompute_UX_CTF_S_l2_S_;
        index_nS_to_update_ = torch.arange(n_S).to(dtype=torch.int32);
        if parameter['flag_precompute_UX_CTF_S_k_q_wnS__']==0 and parameter['flag_precompute_UX_CTF_S_l2_S_']==0: index_nS_to_update_ = torch.arange(0).to(dtype=torch.int32); #end;%if;
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
        if (flag_verbose>0): disp(sprintf(' %% %% X_SM__: %0.3fs',tmp_t)); #end;
        parameter = parameter_timing_update(parameter,sprintf('%s: tfpmh_Z_cluster_wrap_SM__14',str_thisfunction),tmp_t);

        #%%%%%%%%;
        if (flag_save_stage>2) and ('fname_pre' in parameter):
            fname_mat = sprintf('%s_stage_9_%d.mat',parameter['fname_pre'],niteration);
            disp(sprintf(' %% writing %s',fname_mat));
            matlab_save(
                fname_mat=fname_mat,
                dictionary_original= {
                    "n_k_p_r":n_k_p_r,
                    "k_p_r_":k_p_r_,
                    "k_p_r_max":k_p_r_max,
                    "n_w_":n_w_,
                    "weight_2d_k_p_r_":weight_2d_k_p_r_,
                    "weight_2d_k_p_wk_":weight_2d_k_p_wk_,
                    "n_S":n_S,
                    "S_k_p_wkS__":S_k_p_wkS__,
                    "n_CTF":n_CTF,
                    "CTF_k_p_r_kC__":CTF_k_p_r_kC__,
                    "index_nCTF_from_nM_":index_nCTF_from_nM_,
                    "n_M":n_M,
                    "M_pert_k_p_wkM__":M_pert_k_p_wkM__,
                    "n_cluster":n_cluster,
                    "index_ncluster_from_nCTF_":index_ncluster_from_nCTF_,
                    "pm_n_UX_rank_c_":pm_n_UX_rank_c_,
                    "pm_UX_knc___":pm_UX_knc___,
                    "pm_X_weight_rc__":pm_X_weight_rc__,
                    "Z_SM__":Z_SM__,
                    "UX_CTF_S_l2_SM__":UX_CTF_S_l2_SM__,
                    "UX_T_M_l2_SM__":UX_T_M_l2_SM__,
                    "X_SM__":X_SM__,
                    "delta_x_SM__":delta_x_SM__,
                    "delta_y_SM__":delta_y_SM__,
                    "gamma_z_SM__":gamma_z_SM__,
                    "index_sub_SM__":index_sub_SM__,
                    "index_nM_from_ncluster__":index_nM_from_ncluster__,
                    "n_index_nM_from_ncluster_":n_index_nM_from_ncluster_,
                    "M_pert_k_p_wkM__":M_pert_k_p_wkM__,
                    "M_k_q_wkM__":M_k_q_wkM__,
                    "UX_T_M_l2_dM__":UX_T_M_l2_dM__,
                    "UX_M_l2_M_":UX_M_l2_M_,
                    "svd_V_UX_M_lwnM____":svd_V_UX_M_lwnM____,
                    "UX_CTF_S_k_q_wnS__":UX_CTF_S_k_q_wnS__,
                },
            );
        #end;%if ( isfield(parameter,'fname_pre'));
        #%%%%%%%%;

        #%%%%%%%%;
        #% Use current correlations to udate current euler-angles. ;
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
        tmp_str = 'SM';
        if (parameter['flag_MS_vs_SM']): tmp_str = 'MS'; #end;
        tmp_t = toc(tmp_t); 
        if (flag_verbose>0): disp(sprintf(' %% %% %s: update euler_polar_a_M_ euler_azimu_b_M_ euler_gamma_z_M_ : %0.3fs',tmp_str,tmp_t)); #end;
        parameter = parameter_timing_update(parameter,sprintf('%s: tfpmh_MS_vs_SM_2',str_thisfunction),tmp_t);

        #%%%%%%%%;
        if (flag_save_stage>2) and ('fname_pre' in parameter):
            fname_mat = sprintf('%s_stage_10_%d.mat',parameter['fname_pre'],niteration);
            disp(sprintf(' %% writing %s',fname_mat));
            matlab_save(
                fname_mat=fname_mat,
                dictionary_original= {
                    "euler_polar_a_M_":euler_polar_a_M_,
                    "euler_azimu_b_M_":euler_azimu_b_M_,
                    "euler_gamma_z_M_":euler_gamma_z_M_,
                    "image_delta_x_bit_M_":image_delta_x_bit_M_,
                    "image_delta_y_bit_M_":image_delta_y_bit_M_,
                    "image_I_value_M_":image_I_value_M_,
                    "image_X_value_M_":image_X_value_M_,
                    "image_S_index_M_":image_S_index_M_,
                },
            );
        #end;%if ( isfield(parameter,'fname_pre'));
        #%%%%%%%%;

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
        image_delta_x_nrm_M_ = image_delta_x_tot_M_;
        image_delta_y_nrm_M_ = image_delta_y_tot_M_;
        tmp_index_ = efind(image_delta_r_tot_M_> delta_r_upb);
        if (numel(tmp_index_)> 0):
            if (flag_verbose>0): disp(sprintf(' %% %% normalizing %d/%d image displacements',numel(tmp_index_),n_M)); #end;
            image_delta_x_nrm_M_[tmp_index_] = image_delta_x_tot_M_[tmp_index_]*delta_r_upb/torch.maximum(tolerance_machine_1_,image_delta_r_tot_M_[tmp_index_]);
            image_delta_y_nrm_M_[tmp_index_] = image_delta_y_tot_M_[tmp_index_]*delta_r_upb/torch.maximum(tolerance_machine_1_,image_delta_r_tot_M_[tmp_index_]);
            image_delta_x_upd_M_[tmp_index_] = image_delta_x_nrm_M_[tmp_index_] - image_delta_x_acc_M_[tmp_index_];
            image_delta_y_upd_M_[tmp_index_] = image_delta_y_nrm_M_[tmp_index_] - image_delta_y_acc_M_[tmp_index_];
        #end;%if (numel(tmp_index_)> 0);
        flag_image_delta_upd_M_ = torch.zeros(n_M).to(dtype=torch.int32);
        image_delta_r_upd_posnorm_M_ = torch.sqrt(image_delta_x_upd_M_**2 + image_delta_y_upd_M_**2);
        tmp_index_ = efind( torch.logical_or( image_delta_r_upd_prenorm_M_>=delta_r_upd_threshold , image_delta_r_upd_posnorm_M_>=delta_r_upd_threshold ) );
        if (numel(tmp_index_)> 0):
            if (flag_verbose>0): disp(sprintf(' %% %% accumulating %d/%d image displacements',numel(tmp_index_),n_M)); #end;
            flag_image_delta_upd_M_[tmp_index_] = 1;
            image_delta_x_acc_M_[tmp_index_] = image_delta_x_acc_M_[tmp_index_] + image_delta_x_upd_M_[tmp_index_];
            image_delta_y_acc_M_[tmp_index_] = image_delta_y_acc_M_[tmp_index_] + image_delta_y_upd_M_[tmp_index_];
            image_delta_x_upd_M_[tmp_index_] = 0;
            image_delta_y_upd_M_[tmp_index_] = 0;
        #end;%if (numel(tmp_index_)> 0);

        #%%%%%%%%;
        if (flag_save_stage>2) and ('fname_pre' in parameter):
            fname_mat = sprintf('%s_stage_11_%d.mat',parameter['fname_pre'],niteration);
            disp(sprintf(' %% writing %s',fname_mat));
            matlab_save(
                fname_mat=fname_mat,
                dictionary_original= {
                    "delta_r_upd_threshold":delta_r_upd_threshold,
                    "image_delta_x_tot_M_":image_delta_x_tot_M_,
                    "image_delta_y_tot_M_":image_delta_y_tot_M_,
                    "image_delta_r_tot_M_":image_delta_r_tot_M_,
                    "image_delta_x_nrm_M_":image_delta_x_nrm_M_,
                    "image_delta_y_nrm_M_":image_delta_y_nrm_M_,
                    "tmp_index_":tmp_index_,
                    "flag_image_delta_upd_M_":flag_image_delta_upd_M_,
                    "image_delta_r_upd_prenorm_M_":image_delta_r_upd_prenorm_M_,
                    "image_delta_r_upd_posnorm_M_":image_delta_r_upd_posnorm_M_,
                    "image_delta_x_acc_M_":image_delta_x_acc_M_,
                    "image_delta_y_acc_M_":image_delta_y_acc_M_,
                    "image_delta_x_upd_M_":image_delta_x_upd_M_,
                    "image_delta_y_upd_M_":image_delta_y_upd_M_,
                },
            );
        #end;%if ( isfield(parameter,'fname_pre'));
        #%%%%%%%%;

        #%%%%%%%%;
        if niteration<n_iteration-1:
            flag_image_delta_upd_Mi__[niteration,:] = flag_image_delta_upd_M_; #%<-- these actually refer to end of iteration. ;
            image_X_value_Mi__[niteration,:] = image_X_value_M_; #%<-- these actually refer to end of iteration. ;
            image_S_index_Mi__[niteration,:] = image_S_index_M_; #%<-- these actually refer to end of iteration. ;
        #end;%if niteration<n_iteration-1;
        #%%%%%%%%;
        #% Now return to beginning of loop. ;
        #%%%%%%%%;
    #end;%for niteration=0:n_iteration-1;
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;

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

