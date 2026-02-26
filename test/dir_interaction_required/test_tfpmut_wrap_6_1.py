from dir_empm.dir_matlab_macros import * ;
from dir_empm.tfpmut_wrap_6 import tfpmut_wrap_6 ;

#%%%%;
flag_verbose=1;
tmp_dir_base = '/data/rangan/dir_cryoem';
tmp_dir_tfpm_mat = sprintf('%s/dir_trpv1_x0/dir_tfpm_mat',tmp_dir_base);
tmp_fname_mat = sprintf('%s/test_tfpmut_wrap_6_a1t0014p15r1.mat',tmp_dir_tfpm_mat);
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
parameter['dir_tfpm ']= '/data/rangan/dir_cryoem/dir_trpv1_x0/dir_tfpm';
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
XA_parameter = parameter;
XA_parameter['flag_gpu']=1;
XA_parameter['fname_pre']='/data/rangan/dir_cryoem/dir_trpv1_x0/dir_tfpm_mat/test_tfpmut_wrap_6_XA_from_python';
tmp_t=tic();
XA_parameter = tfpmut_wrap_6(
    XA_parameter,
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
)[0];
tmp_t = toc(tmp_t);
parameter_timing_printf(XA_parameter);
if (flag_verbose>1): disp(sprintf(' %% tfpmut_wrap_6: time %0.2fs',tmp_t)); end;
#%%%%%%%%;
XB_parameter = parameter;
XB_parameter['flag_gpu']=1;
XB_parameter['fname_pre']='/data/rangan/dir_cryoem/dir_trpv1_x0/dir_tfpm_mat/test_tfpmut_wrap_6_XB_from_python';
tmp_t=tic();
XB_parameter = tfpmut_wrap_6(
    XB_parameter,
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
    tmp_XA_euler_polar_a_M_,
    tmp_XA_euler_azimu_b_M_,
    tmp_XA_euler_gamma_z_M_,
    tmp_XA_image_delta_x_M_,
    tmp_XA_image_delta_y_M_,
    tmp_XA_a_k_Y_reco_yk_,
)[0];
tmp_t = toc(tmp_t);
parameter_timing_printf(XB_parameter);
if (flag_verbose>1): disp(sprintf(' %% tfpmut_wrap_6: time %0.2fs',tmp_t)); end;
#%%%%%%%%;



