from dir_empm.dir_matlab_macros import * ;
from dir_empm.tfh_FTK_4 import tfh_FTK_4 ;
from dir_empm.tfpmh_Z_cluster_wrap_SM__14 import tfpmh_Z_cluster_wrap_SM__14 ;

str_thisfunction = 'test_tfpmut_wrap_6_stage_9';
#%%%%;
flag_verbose=1;
tmp_dir_base = '/data/rangan/dir_cryoem/';
tmp_dir_pm_mat = sprintf('%s/dir_FINYU_nx64s003nM1024nCTF50p40V300D25000r050A010/dir_pm_mat',tmp_dir_base);
tmp_fname_mat = sprintf('%s/test_tfpmut_wrap_6_a1t0017p10r1.mat',tmp_dir_pm_mat);
#%%%%;
tmp_ = matlab_load(fname_mat = tmp_fname_mat);
n_k_p_r = int(tmp_['n_k_p_r'].item());
k_p_r_ = tmp_['k_p_r_'].to(dtype=torch.float32);
k_p_r_max = float(tmp_['k_p_r_max'].item());
weight_3d_k_p_r_ = tmp_['weight_3d_k_p_r_'].to(dtype=torch.float32);
weight_2d_k_p_r_ = tmp_['weight_2d_k_p_r_'].to(dtype=torch.float32);
n_w_ = tmp_['n_w_'].to(dtype=torch.int32);
weight_2d_k_p_wk_ = tmp_['weight_2d_k_p_wk_'].to(dtype=torch.float32);
l_max_ = tmp_['l_max_'].to(dtype=torch.int32);
n_CTF = int(tmp_['n_CTF'].item());
CTF_k_p_wkC__ = tmp_['CTF_k_p_wkC__'].to(dtype=torch.float32);
index_nCTF_from_nM_ = tmp_['index_nCTF_from_nM_'].to(dtype=torch.int32);
n_M = int(tmp_['n_M'].item());
M_k_p_wkM__ = tmp_['M_k_p_wkM__'].to(dtype=torch.complex64);
tmp_XA_euler_polar_a_M_ = tmp_['tmp_XA_euler_polar_a_M_'].to(dtype=torch.float32);
tmp_XA_euler_azimu_b_M_ = tmp_['tmp_XA_euler_azimu_b_M_'].to(dtype=torch.float32);
tmp_XA_euler_gamma_z_M_ = tmp_['tmp_XA_euler_gamma_z_M_'].to(dtype=torch.float32);
tmp_XA_image_delta_x_M_ = tmp_['tmp_XA_image_delta_x_M_'].to(dtype=torch.float32);
tmp_XA_image_delta_y_M_ = tmp_['tmp_XA_image_delta_y_M_'].to(dtype=torch.float32);
tmp_XA_a_k_Y_reco_yk_ = tmp_['tmp_XA_a_k_Y_reco_yk_'].to(dtype=torch.complex64);
#%%%%;
parameter = {'type':'parameter'};
parameter['flag_verbose']=flag_verbose;
parameter['rseed']=0;
parameter['tolerance_pm']=0.100000000000000;
parameter['delta_r_max']=0.017631663493955;
parameter['delta_r_upb']=0.141053307951639;
parameter['dir_pm ']= '/data/rangan/dir_cryoem/dir_FINYU_nx64s003nM1024nCTF50p40V300D25000r050A010/dir_pm';
parameter['flag_alternate_MS_vs_SM']=1;
parameter['flag_force_create_mat']=0;
parameter['flag_force_create_tmp']=1;
parameter['flag_gpu']=1;
parameter['tolerance_master']=0.010000000000000;
parameter['flag_rank_vs_tolerance']=0;
parameter['flag_clump_vs_cluster']=0;
parameter['rank_pm']=10;
parameter['rank_CTF']=4;
parameter['k_p_r_max']=7.639437268410976;
parameter['sample_sphere_k_eq_d']=0.159154943091895;
parameter['half_diameter_x_c']=1;
parameter['n_x_u_pack']=64;
parameter['cg_lsq_n_order']=5;
parameter['date_diff_threshold']=0.250000000000000;
parameter['flag_qbp_vs_lsq']=1;
parameter['qbp_eps']=0.010000000000000;
parameter['str_strategy_prefix']='';
parameter['n_complete_calculation']=0;
parameter['n_iteration']=32;
#%%%%%%%%;

#%%%%%%%%;
fname_mat = '/data/rangan/dir_cryoem/dir_FINYU_nx64s003nM1024nCTF50p40V300D25000r050A010/dir_pm_mat/test_tfpmut_wrap_6_XB_from_python_stage_4.mat';
tmp_ = matlab_load(fname_mat = fname_mat);
delta_r_max = float(tmp_['delta_r_max'].item());
svd_eps = float(tmp_['svd_eps'].item());
n_delta_v_requested = int(tmp_['n_delta_v_requested'].item());
flag_precompute_UX_CTF_S_k_q_wnS__ = int(tmp_['flag_precompute_UX_CTF_S_k_q_wnS__'].item());
flag_precompute_UX_CTF_S_l2_S_ = int(tmp_['flag_precompute_UX_CTF_S_l2_S_'].item());
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
assert(FTK['r8_svd_d_max']>=delta_r_max);
assert(FTK['n_delta_v']>=n_delta_v_requested);
n_delta_v = FTK['n_delta_v'];
n_svd_l = FTK['n_svd_l'];
#%%%%%%%%;

#%%%%%%%%;
fname_mat = '/data/rangan/dir_cryoem/dir_FINYU_nx64s003nM1024nCTF50p40V300D25000r050A010/dir_pm_mat/test_tfpmut_wrap_6_XB_from_python_stage_9_0.mat';
tmp_ = matlab_load(fname_mat = fname_mat);
n_k_p_r = int(tmp_['n_k_p_r'].item());
k_p_r_ = tmp_['k_p_r_'].to(dtype=torch.float32);
k_p_r_max = float(tmp_['k_p_r_max'].item());
weight_2d_k_p_r_ = tmp_['weight_2d_k_p_r_'].to(dtype=torch.float32);
n_w_ = tmp_['n_w_'].to(dtype=torch.int32);
weight_2d_k_p_wk_ = tmp_['weight_2d_k_p_wk_'].to(dtype=torch.float32);
n_CTF = int(tmp_['n_CTF'].item());
index_nCTF_from_nM_ = tmp_['index_nCTF_from_nM_'].to(dtype=torch.int32);
n_M = int(tmp_['n_M'].item());
n_cluster = int(tmp_['n_cluster'].item());
index_ncluster_from_nCTF_ = tmp_['index_ncluster_from_nCTF_'].to(dtype=torch.int32);
pm_n_UX_rank_c_ = tmp_['pm_n_UX_rank_c_'].to(dtype=torch.int32);
pm_UX_knc___ = tmp_['pm_UX_knc___'].to(dtype=torch.float32);
pm_X_weight_rc__ = tmp_['pm_X_weight_rc__'].to(dtype=torch.float32);
index_nM_from_ncluster__ = tmp_['index_nM_from_ncluster__'];
n_index_nM_from_ncluster_ = tmp_['n_index_nM_from_ncluster_'].to(dtype=torch.int32);
CTF_k_p_r_kC__ = tmp_['CTF_k_p_r_kC__'].to(dtype=torch.float32);
M_pert_k_p_wkM__ = tmp_['M_pert_k_p_wkM__'].to(dtype=torch.complex64);
S_k_p_wkS__ = tmp_['S_k_p_wkS__'].to(dtype=torch.complex64);
n_S = int(tmp_['n_S'].item());
M_k_q_wkM__ = tmp_['M_k_q_wkM__'].to(dtype=torch.complex64);
UX_T_M_l2_dM__ = tmp_['UX_T_M_l2_dM__'].to(dtype=torch.float32);
UX_M_l2_M_ = tmp_['UX_M_l2_M_'].to(dtype=torch.float32);
svd_V_UX_M_lwnM____ = tmp_['svd_V_UX_M_lwnM____'].to(dtype=torch.complex64);
UX_CTF_S_k_q_wnS__ = tmp_['UX_CTF_S_k_q_wnS__'].to(dtype=torch.complex64);
UX_CTF_S_l2_S_ = tmp_['UX_CTF_S_l2_S_'].to(dtype=torch.float32);
Z_SM__ = tmp_['Z_SM__'].to(dtype=torch.float32);
UX_T_M_l2_SM__ = tmp_['UX_T_M_l2_SM__'].to(dtype=torch.float32);
X_SM__ = tmp_['X_SM__'].to(dtype=torch.float32);
delta_x_SM__ = tmp_['delta_x_SM__'].to(dtype=torch.float32);
delta_y_SM__ = tmp_['delta_y_SM__'].to(dtype=torch.float32);
gamma_z_SM__ = tmp_['gamma_z_SM__'].to(dtype=torch.float32);
index_sub_SM__ = tmp_['index_sub_SM__'].to(dtype=torch.int32);
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
#%%%%;
index_nM_to_update_ = torch.randperm(n_M)[0:8];
parameter['flag_verbose'] = 0;
#%%%%;
tmp_t = tic();
(
    parameter,
    Z_SM__,
    UX_CTF_S_l2_S_,
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
parameter_timing_printf(parameter);



