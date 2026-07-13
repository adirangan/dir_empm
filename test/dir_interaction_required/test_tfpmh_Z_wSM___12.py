from dir_empm.dir_matlab_macros import * ;
from dir_empm.get_weight_3d_1 import get_weight_3d_1 ;
from dir_empm.get_weight_2d_2 import get_weight_2d_2 ;
from dir_empm.tfh_FTK_2 import tfh_FTK_2 ;
from dir_empm.tfpmh_Z_wSM___12 import tfpmh_Z_wSM___12 ;

flag_verbose=1; flag_disp=0;nf=0;
print(f' %% testing tfpmh_Z_wSM___12');
k_p_r_max = 48.0/(2*pi); k_eq_d = 1.0/(2*pi); str_L = 'L';
(
    n_k_p_r,
    k_p_r_,
    weight_3d_k_p_r_,
) = get_weight_3d_1(
    0*flag_verbose,
    k_p_r_max,
    k_eq_d,
    str_L,
)[:3];
#%%%%;
n_w_int = 1;
l_max_upb = matlab_scalar_round(2*pi*k_p_r_max);
l_max_max = int(np.minimum(l_max_upb,1+np.ceil(2*pi*k_p_r_[-1].item())));
n_w_max = int(n_w_int*2*(l_max_max+1)); n_w_0in_ = n_w_max*torch.ones(n_k_p_r).to(dtype=torch.int32);
(
    n_w_,
    weight_2d_k_p_r_,
    weight_2d_k_p_wk_,
    k_p_r_wk_,
    k_p_w_wk_,
    k_c_0_wk_,
    k_c_1_wk_,
) = get_weight_2d_2(
    0*flag_verbose,
    n_k_p_r,
    k_p_r_,
    k_p_r_max,
    -1,
    n_w_0in_,
    weight_3d_k_p_r_,
);
n_w_sum = int(torch.sum(n_w_).item()); n_w_csum_ = cumsum_0(n_w_);
#%%%%;
n_M = 17;
delta_2M__ = torch.reshape(torch.linspace(-0.05,+0.05,n_2*n_M),mtr((n_2,n_M))).to(dtype=torch.float32);
M_k_p_wkM__ = torch.zeros(mtr((n_w_sum,n_M))).to(dtype=torch.complex64);
for nM in range(n_M):
    delta_0 = delta_2M__[nM,0].item(); delta_1 = delta_2M__[nM,1].item();
    M_k_p_wkM__[nM,:] = 1*torch.exp(+i*2*pi*(k_c_0_wk_*delta_0 + k_c_1_wk_*delta_1));
#end;%for nM=0:n_M-1;
#%%%%;
n_S = 19;
delta_2S__ = torch.reshape(torch.linspace(-0.07,+0.07,n_2*n_S),mtr((n_2,n_S))).to(dtype=torch.float32);
S_k_p_wkS__ = torch.zeros(mtr((n_w_sum,n_S))).to(dtype=torch.complex64);
for nS in range(n_S):
    delta_0 = delta_2S__[nS,0].item(); delta_1 = delta_2S__[nS,1].item();
    S_k_p_wkS__[nS,:] = i*torch.exp(+i*2*pi*(k_c_0_wk_*delta_0 + k_c_1_wk_*delta_1));
#end;%for nM=0:n_M-1;
#%%%%;
CTF_phi = 2*pi/3;
CTF_k_p_wk_ = torch.zeros(n_w_sum).to(dtype=torch.float32);
CTF_k_p_wk_ = 2*k_p_r_wk_*torch.cos(k_p_w_wk_ - CTF_phi);
#%%%%;
pm_n_UX_rank = n_k_p_r-1;
tmp_n = int(np.maximum(n_k_p_r,pm_n_UX_rank)); pm_UX_kn__ = torch.eye(tmp_n).to(dtype=torch.float32); tmp_i8_index_rhs_ = matlab_index_2d_0(tmp_n,torch.arange(n_k_p_r),tmp_n,torch.arange(pm_n_UX_rank)); pm_UX_kn__ = torch.reshape(pm_UX_kn__.ravel()[tmp_i8_index_rhs_],mtr((n_k_p_r,pm_n_UX_rank))).to(dtype=torch.float32);
pm_X_weight_r_ = torch.sqrt(weight_2d_k_p_r_);
#%%%%;
delta_r_max = 0.5/np.maximum(1e-12,k_p_r_max); svd_eps = 1e-3; n_delta_v_requested = 7;
FTK = tfh_FTK_2(n_k_p_r,k_p_r_.to(dtype=torch.float64),float(k_p_r_max),float(delta_r_max),float(svd_eps),n_delta_v_requested);
n_delta_v = FTK['n_delta_v'];
n_svd_l = FTK['n_svd_l'];

#%%%%%%%%;
parameter = {'type': 'parameter'} ;
parameter['flag_verbose'] = 1;
parameter['flag_optimize_over_gamma_z'] = 0;
parameter['n_M_per_Mbatch'] = 3;
parameter['n_S_per_Sbatch'] = 5;
parameter['flag_dwSM'] = 1;
(
    parameter,
    tfpmh_Z_wSM___,
    tfpmh_UX_CTF_R_S_l2_wS__,
    tfpmh_CTF_R_S_l2_wS__,
    tfpmh_UX_T_M_l2_dM__,
    tfpmh_UX_M_l2_M_,
    tfpmh_X_wSM___,
    tfpmh_delta_x_wSM___,
    tfpmh_delta_y_wSM___,
    tfpmh_gamma_z_wSM___,
    tfpmh_index_sub_wSM___,
    tfpmh_Z_dwSM____,
    tfpmh_X_dwSM____,
) = tfpmh_Z_wSM___12(
    parameter,
    n_k_p_r,
    k_p_r_,
    k_p_r_max,
    n_w_,
    weight_2d_k_p_r_,
    weight_2d_k_p_wk_,
    n_M,
    M_k_p_wkM__,
    CTF_k_p_wk_,
    n_S,
    S_k_p_wkS__,
    pm_n_UX_rank,
    pm_UX_kn__,
    pm_X_weight_r_,
    FTK,
);
#%%%%%%%%;

dir_base = '/data/rangan' ;
dir_ascii = dir_base + '/dir_cryoem/dir_rangan_python/dir_ascii' ;

fname_ascii = dir_ascii + '/tfpmh_Z_wSM___.ascii' ;
print(f' %% writing fname_ascii: {fname_ascii}');
np.savetxt(fname_ascii,tfpmh_Z_wSM___.numpy().ravel());
fname_ascii = dir_ascii + '/tfpmh_UX_CTF_R_S_l2_wS__.ascii' ;
print(f' %% writing fname_ascii: {fname_ascii}');
np.savetxt(fname_ascii,tfpmh_UX_CTF_R_S_l2_wS__.numpy().ravel());
fname_ascii = dir_ascii + '/tfpmh_CTF_R_S_l2_wS__.ascii' ;
print(f' %% writing fname_ascii: {fname_ascii}');
np.savetxt(fname_ascii,tfpmh_CTF_R_S_l2_wS__.numpy().ravel());
fname_ascii = dir_ascii + '/tfpmh_UX_T_M_l2_dM__.ascii' ;
print(f' %% writing fname_ascii: {fname_ascii}');
np.savetxt(fname_ascii,tfpmh_UX_T_M_l2_dM__.numpy().ravel());
fname_ascii = dir_ascii + '/tfpmh_UX_M_l2_M_.ascii' ;
print(f' %% writing fname_ascii: {fname_ascii}');
np.savetxt(fname_ascii,tfpmh_UX_M_l2_M_.numpy().ravel());
fname_ascii = dir_ascii + '/tfpmh_X_wSM___.ascii' ;
print(f' %% writing fname_ascii: {fname_ascii}');
np.savetxt(fname_ascii,tfpmh_X_wSM___.numpy().ravel());
fname_ascii = dir_ascii + '/tfpmh_delta_x_wSM___.ascii' ;
print(f' %% writing fname_ascii: {fname_ascii}');
np.savetxt(fname_ascii,tfpmh_delta_x_wSM___.numpy().ravel());
fname_ascii = dir_ascii + '/tfpmh_delta_y_wSM___.ascii' ;
print(f' %% writing fname_ascii: {fname_ascii}');
np.savetxt(fname_ascii,tfpmh_delta_y_wSM___.numpy().ravel());
fname_ascii = dir_ascii + '/tfpmh_gamma_z_wSM___.ascii' ;
print(f' %% writing fname_ascii: {fname_ascii}');
np.savetxt(fname_ascii,tfpmh_gamma_z_wSM___.numpy().ravel());
fname_ascii = dir_ascii + '/tfpmh_index_sub_wSM___.ascii' ;
print(f' %% writing fname_ascii: {fname_ascii}');
np.savetxt(fname_ascii,tfpmh_index_sub_wSM___.numpy().ravel());
if tfpmh_Z_dwSM____ is not None:
    fname_ascii = dir_ascii + '/tfpmh_Z_dwSM____.ascii' ;
    print(f' %% writing fname_ascii: {fname_ascii}');
    np.savetxt(fname_ascii,tfpmh_Z_dwSM____.numpy().ravel());
#end;%if tfpmh_Z_dwSM____ is not None;
if tfpmh_X_dwSM____ is not None:
    fname_ascii = dir_ascii + '/tfpmh_X_dwSM____.ascii' ;
    print(f' %% writing fname_ascii: {fname_ascii}');
    np.savetxt(fname_ascii,tfpmh_X_dwSM____.numpy().ravel());
#end;%if tfpmh_X_dwSM____ is not None;

#%%%%%%%%;
parameter = {'type': 'parameter'} ;
parameter['flag_verbose'] = 1;
parameter['flag_optimize_over_gamma_z'] = 1;
parameter['n_M_per_Mbatch'] = 3;
parameter['n_S_per_Sbatch'] = 5;
parameter['flag_dwSM'] = 0;
(
    parameter,
    tfpmh_Z_SM__,
    tfpmh_UX_CTF_R_S_l2_wS__,
    tfpmh_CTF_R_S_l2_wS__,
    tfpmh_UX_T_M_l2_dM__,
    tfpmh_UX_M_l2_M_,
    tfpmh_X_SM__,
    tfpmh_delta_x_SM__,
    tfpmh_delta_y_SM__,
    tfpmh_gamma_z_SM__,
    tfpmh_index_sub_SM__,
) = tfpmh_Z_wSM___12(
    parameter,
    n_k_p_r,
    k_p_r_,
    k_p_r_max,
    n_w_,
    weight_2d_k_p_r_,
    weight_2d_k_p_wk_,
    n_M,
    M_k_p_wkM__,
    CTF_k_p_wk_,
    n_S,
    S_k_p_wkS__,
    pm_n_UX_rank,
    pm_UX_kn__,
    pm_X_weight_r_,
    FTK,
)[:11];
#%%%%%%%%;

fname_ascii = dir_ascii + '/tfpmh_Z_SM__.ascii' ;
print(f' %% writing fname_ascii: {fname_ascii}');
np.savetxt(fname_ascii,tfpmh_Z_SM__.numpy().ravel());
fname_ascii = dir_ascii + '/tfpmh_UX_CTF_R_S_l2_wS__.ascii' ;
print(f' %% writing fname_ascii: {fname_ascii}');
np.savetxt(fname_ascii,tfpmh_UX_CTF_R_S_l2_wS__.numpy().ravel());
fname_ascii = dir_ascii + '/tfpmh_CTF_R_S_l2_wS__.ascii' ;
print(f' %% writing fname_ascii: {fname_ascii}');
np.savetxt(fname_ascii,tfpmh_CTF_R_S_l2_wS__.numpy().ravel());
fname_ascii = dir_ascii + '/tfpmh_UX_T_M_l2_dM__.ascii' ;
print(f' %% writing fname_ascii: {fname_ascii}');
np.savetxt(fname_ascii,tfpmh_UX_T_M_l2_dM__.numpy().ravel());
fname_ascii = dir_ascii + '/tfpmh_UX_M_l2_M_.ascii' ;
print(f' %% writing fname_ascii: {fname_ascii}');
np.savetxt(fname_ascii,tfpmh_UX_M_l2_M_.numpy().ravel());
fname_ascii = dir_ascii + '/tfpmh_X_SM__.ascii' ;
print(f' %% writing fname_ascii: {fname_ascii}');
np.savetxt(fname_ascii,tfpmh_X_SM__.numpy().ravel());
fname_ascii = dir_ascii + '/tfpmh_delta_x_SM__.ascii' ;
print(f' %% writing fname_ascii: {fname_ascii}');
np.savetxt(fname_ascii,tfpmh_delta_x_SM__.numpy().ravel());
fname_ascii = dir_ascii + '/tfpmh_delta_y_SM__.ascii' ;
print(f' %% writing fname_ascii: {fname_ascii}');
np.savetxt(fname_ascii,tfpmh_delta_y_SM__.numpy().ravel());
fname_ascii = dir_ascii + '/tfpmh_gamma_z_SM__.ascii' ;
print(f' %% writing fname_ascii: {fname_ascii}');
np.savetxt(fname_ascii,tfpmh_gamma_z_SM__.numpy().ravel());
fname_ascii = dir_ascii + '/tfpmh_index_sub_SM__.ascii' ;
print(f' %% writing fname_ascii: {fname_ascii}');
np.savetxt(fname_ascii,tfpmh_index_sub_SM__.numpy().ravel());

