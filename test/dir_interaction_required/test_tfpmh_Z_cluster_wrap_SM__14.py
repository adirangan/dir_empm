from dir_empm.dir_matlab_macros import * ;
from dir_empm.h2d import h2d ;
from dir_empm.sample_sphere_7 import sample_sphere_7 ;
from dir_empm.get_weight_2d_2 import get_weight_2d_2 ;
from dir_empm.convert_k_p_to_spharm_uniform_over_n_k_p_r_5 import convert_k_p_to_spharm_uniform_over_n_k_p_r_5 ;
from dir_empm.plane_wave_expansion_1 import plane_wave_expansion_1 ;
from dir_empm.local_yk__from_yk_ import local_yk__from_yk_ ;
from dir_empm.pm_template_3 import pm_template_3 ;
from dir_empm.cg_rhs_2 import cg_rhs_2 ;
from dir_empm.interp_p_to_q import interp_p_to_q ;
from dir_empm.transf_p_to_p import transf_p_to_p ;
from dir_empm.rotate_p_to_p_fftw import rotate_p_to_p_fftw ;
from dir_empm.tfh_FTK_4 import tfh_FTK_4 ;
from dir_empm.principled_marching_empirical_cost_matrix_1 import principled_marching_empirical_cost_matrix_1 ;
from dir_empm.tfpmh_Z_cluster_wrap_SM__14 import tfpmh_Z_cluster_wrap_SM__14 ;

flag_tf_vs_bf = 1;
flag_gpu = 1;

str_thisfunction = 'tfpmh_Z_cluster_wrap_SM__14';
#%%%%%%%%;
flag_verbose=1;
k_p_r_max = 48/(2*pi); k_eq_d = 1.0/(2*pi); str_L = 'L'; flag_uniform_over_n_k_p_r = 1; flag_uniform_over_polar_a = 0;
if (flag_verbose>0): disp(sprintf(' %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;')); #end;
if (flag_verbose>0): disp(sprintf(' %% [testing %s]',str_thisfunction)); #end;
if (flag_verbose>0): disp(sprintf(' %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;')); #end;
if (flag_verbose>0): disp(sprintf(' %% Test grid and template generation: ;')); #end;
if (flag_verbose>0): disp(sprintf(' %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;')); #end;
#%%%%%%%%;
tmp_t=tic();
(
    n_qk,
    n_qk_csum_,
    k_p_r_qk_,
    k_p_azimu_b_qk_,
    k_p_polar_a_qk_,
    weight_3d_k_p_qk_,
    weight_shell_qk_,
    n_k_p_r,
    k_p_r_,
    weight_3d_k_p_r_,
    k_c_0_qk_,
    k_c_1_qk_,
    k_c_2_qk_,
) = sample_sphere_7(
    0*flag_verbose,
    k_p_r_max,
    k_eq_d,
    str_L,
    flag_uniform_over_n_k_p_r,
    flag_uniform_over_polar_a,
)[:13];
tmp_t=toc(tmp_t);
if (flag_verbose>0): disp(sprintf(' %% sample_sphere_7: %0.2fs',tmp_t)); #end;
if (flag_verbose>0): disp(sprintf(' %% k_p_r_max %0.2f k_eq_d %0.2f n_qk %d n_k_p_r %d',k_p_r_max,k_eq_d,n_qk,n_k_p_r)); #end;
#%%%%%%%%;
l_max_upb = round(2*pi*k_p_r_max); #%<-- typically sufficient for 2-3 digits of precision. ;
l_max_ = torch.zeros(n_k_p_r).to(dtype=torch.int32);
for nk_p_r in range(n_k_p_r):
    l_max_[nk_p_r] = int(np.maximum(0,np.minimum(l_max_upb,1+np.ceil(2*pi*k_p_r_[nk_p_r].item()))));
#end;%for nk_p_r=0:n_k_p_r-1;
n_y_ = ((l_max_+1)**2).to(dtype=torch.int32);
n_y_max = int(torch.max(n_y_).item());
n_y_sum = int(torch.sum(n_y_).item());
n_y_csum_ = cumsum_0(n_y_);
l_max_max = int(torch.max(l_max_).item());
m_max_ = torch.arange(-l_max_max,+l_max_max+1).to(dtype=torch.int32);
n_m_max = numel(m_max_);
#%%%%%%%%;
n_w_max = 2*(l_max_max+1);
template_k_eq_d = -1;
n_w_0in_ = n_w_max*torch.ones(n_k_p_r).to(dtype=torch.int32);
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
    template_k_eq_d,
    n_w_0in_,
    weight_3d_k_p_r_,
)[:7];
n_w_sum = int(torch.sum(n_w_).item()); n_w_max = int(torch.max(n_w_).item()); n_w_csum_ = cumsum_0(n_w_);
if (flag_verbose>0): disp(sprintf(' %% n_w_max %d n_w_sum %d',n_w_max,n_w_sum)); #end;
#%%%%%%%%;
delta_orig_ = torch.tensor([+0.12,-0.3,+0.23]).to(dtype=torch.float32);
a_k_p_orig_ = torch.exp(+2*pi*i*(k_c_0_qk_*delta_orig_[0].item() + k_c_1_qk_*delta_orig_[1].item() + k_c_2_qk_*delta_orig_[2].item())).to(dtype=torch.complex64);
tmp_t=tic();
(
    a_k_Y_quad_,
) = convert_k_p_to_spharm_uniform_over_n_k_p_r_5(
    0*flag_verbose,
    n_qk,
    n_qk_csum_,
    k_p_r_qk_,
    k_p_azimu_b_qk_,
    k_p_polar_a_qk_,
    weight_3d_k_p_qk_,
    weight_shell_qk_,
    n_k_p_r,
    k_p_r_,
    weight_3d_k_p_r_,
    l_max_,
    a_k_p_orig_,
)[:1];
tmp_t=toc(tmp_t); disp(sprintf(' %% convert_k_p_to_spharm_5 time %0.2fs',tmp_t));
#%%%%%%%%;
tmp_t=tic();
a_k_Y_form_ = plane_wave_expansion_1(n_k_p_r,k_p_r_,delta_orig_,l_max_);
tmp_t=toc(tmp_t); disp(sprintf(' %% plane_wave_expansion_1: time %0.2fs',tmp_t));
fnorm_disp(flag_verbose,'a_k_Y_form_',a_k_Y_form_,'a_k_Y_quad_',a_k_Y_quad_);
#%%%%%%%%;
viewing_k_eq_d = 1.0*128; #%<-- subsample just a few templates for visualization. ;
tmp_t=tic();
(
    S_k_p_wkS__,
    _,
    n_viewing_S,
    viewing_azimu_b_S_,
    viewing_polar_a_S_,
    viewing_weight_S_,
    n_viewing_polar_a,
    viewing_polar_a_,
    n_viewing_azimu_b_,
) = pm_template_3(
    0*flag_verbose,
    l_max_max,
    n_k_p_r,
    torch.reshape(local_yk__from_yk_(n_k_p_r,l_max_,a_k_Y_quad_),mtr((n_y_max,n_k_p_r))),
    viewing_k_eq_d/np.maximum(1e-12,k_p_r_max),
    template_k_eq_d,
    n_w_max,
)[:9];
tmp_t=toc(tmp_t);
if (flag_verbose>0): disp(sprintf(' %% pm_template_3: %0.2fs',tmp_t)); #end;
n_S = n_viewing_S;
S_k_p_wkS__ = torch.reshape(S_k_p_wkS__,mtr((n_w_max*n_k_p_r,n_viewing_S)));
#%%%%%%%%;
(
    template_k_p_polar_a_wS__,
    template_k_p_azimu_b_wS__,
    template_k_c_0_wS__,
    template_k_c_1_wS__,
    template_k_c_2_wS__,
    template_k_p_r01_wS__,
) = cg_rhs_2(
    n_S,
    n_w_max,
    viewing_polar_a_S_,
    viewing_azimu_b_S_,
    torch.zeros(n_S).to(dtype=torch.float32),
)[:6];
template_k_p_polar_a_wkS___ = torch.tile(torch.reshape(template_k_p_polar_a_wS__,mtr((n_w_max,1,n_S))),mtr((1,n_k_p_r,1)));
template_k_p_azimu_b_wkS___ = torch.tile(torch.reshape(template_k_p_azimu_b_wS__,mtr((n_w_max,1,n_S))),mtr((1,n_k_p_r,1)));
template_k_p_r_wkS___ = torch.tile(torch.reshape(k_p_r_,mtr((1,n_k_p_r,1))),mtr((n_w_max,1,n_S)));
template_k_c_0_wkS__ = torch.reshape(template_k_p_r_wkS___*torch.sin(template_k_p_polar_a_wkS___)*torch.cos(template_k_p_azimu_b_wkS___),mtr((n_w_sum,n_S)));
template_k_c_1_wkS__ = torch.reshape(template_k_p_r_wkS___*torch.sin(template_k_p_polar_a_wkS___)*torch.sin(template_k_p_azimu_b_wkS___),mtr((n_w_sum,n_S)));
template_k_c_2_wkS__ = torch.reshape(template_k_p_r_wkS___*torch.cos(template_k_p_polar_a_wkS___),mtr((n_w_sum,n_S)));
#%%%%%%%%;
pole_k_c_0_ = k_p_r_wk_*torch.cos(k_p_w_wk_);
pole_k_c_1_ = k_p_r_wk_*torch.sin(k_p_w_wk_);
pole_k_c_2_ = torch.zeros(n_w_sum).to(dtype=torch.float32);
#%%%%%%%%;
#% after calling get_template_1: ;
#% A template with viewing angle viewing_polar_a and viewing_azimu_b corresponds to the evaluations: ;
#% let sa and ca be sin(polar_a) and cos(polar_a), respectively. ;
#% let sb and cb be sin(azimu_b) and cos(azimu_b), respectively. ;
#% let sc and cc be sin(gamma_z) and cos(gamma_z), respectively. ;
#% [ template_k_c_0 ; template_k_c_1 ; template_k_c_2 ] = [ +cb*ca*cc - sb*sc ; +sb*ca*cc + cb*sc ; -sa*cc ] ;
#% for gamma_z = 2*pi*[0:n_gamma_z-1]/n_gamma_z. ;
#% Given that the original function is a plane-wave defined as: ;
#% a_k_p_ = exp(+2*pi*i*( delta_orig_(1+0)*template_k_c_0 + delta_orig_(1+1)*template_k_c_1 + delta_orig_(1+2)*template_k_c_2 )) ;
#% we have that the template evaluates to: ;
#% S_k_p_ = exp(+2*pi*i*( delta_orig_ * Rz(azimu_b) * Ry(polar_a) * Rz(gamma_z) * [1;0;0]*k_p_r )) ;
#% S_k_p_ = exp(+2*pi*i*( (Ry(-polar_a) * Rz(-azimu_b) * delta_orig_) * Rz(gamma_z) * [1;0;0]*k_p_r )) ;
#%%%%%%%%;
n_test = 8;
for ntest in range(n_test):
    nS=int(np.maximum(0,np.minimum(n_S-1,np.floor(n_S*np.random.rand()))));
    disp(sprintf(' %% ntest %d/%d --> nS %d',ntest,n_test,nS));
    S_k_p_quad_ = S_k_p_wkS__[nS,:];
    S_k_p_orig_ = torch.exp(+2*pi*i*(template_k_c_0_wkS__[nS,:]*delta_orig_[0].item() + template_k_c_1_wkS__[nS,:]*delta_orig_[1].item() + template_k_c_2_wkS__[nS,:]*delta_orig_[2].item())).to(dtype=torch.complex64);
    viewing_azimu_b = viewing_azimu_b_S_[nS].item();
    cb = np.cos(+viewing_azimu_b); sb = np.sin(+viewing_azimu_b);
    Rz = torch.reshape(torch.tensor([ +cb , +sb , 0 , -sb , +cb , 0 , 0 , 0 , 1 ]).to(dtype=torch.float32),mtr((3,3))); #%<-- rotation about the positive z-axis. ;
    viewing_polar_a = viewing_polar_a_S_[nS].item();
    ca = np.cos(+viewing_polar_a); sa = np.sin(+viewing_polar_a);
    Ry = torch.reshape(torch.tensor([ +ca , 0 , -sa , 0 , 1 , 0 , +sa , 0 , +ca ]).to(dtype=torch.float32),mtr((3,3))); #%<-- rotation about the positive y-axis. ;
    delta_temp_ = mmvm( Ry.T , mmvm( Rz.T , delta_orig_ ) );
    S_k_p_form_ = torch.exp(+2*pi*i*(pole_k_c_0_*delta_temp_[0].item() + pole_k_c_1_*delta_temp_[1].item() + pole_k_c_2_*delta_temp_[2].item())).to(dtype=torch.complex64);
    fnorm_disp(flag_verbose,'S_k_p_orig_',S_k_p_orig_,'S_k_p_quad_',S_k_p_quad_);
    fnorm_disp(flag_verbose,'S_k_p_orig_',S_k_p_orig_,'S_k_p_form_',S_k_p_form_);
#end;%for ntest=0:n_test-1;
#%%%%%%%%;
S_k_q_wkS__ = torch.reshape(interp_p_to_q(n_k_p_r,n_w_,n_w_sum,S_k_p_wkS__),mtr((n_w_sum,n_S)));
tmp_t=tic();
S_k_p_form__ = torch.zeros(mtr((n_w_sum,n_S))).to(dtype=torch.complex64);
S_k_q_form__ = torch.zeros(mtr((n_w_sum,n_S))).to(dtype=torch.complex64);
for nS in range(n_S):
    viewing_azimu_b = viewing_azimu_b_S_[nS].item();
    cb = np.cos(+viewing_azimu_b); sb = np.sin(+viewing_azimu_b);
    Rz = torch.reshape(torch.tensor([ +cb , +sb , 0 , -sb , +cb , 0 , 0 , 0 , 1 ]).to(dtype=torch.float32),mtr((3,3))); #%<-- rotation about the positive z-axis. ;
    viewing_polar_a = viewing_polar_a_S_[nS].item();
    ca = np.cos(+viewing_polar_a); sa = np.sin(+viewing_polar_a);
    Ry = torch.reshape(torch.tensor([ +ca , 0 , -sa , 0 , 1 , 0 , +sa , 0 , +ca ]).to(dtype=torch.float32),mtr((3,3))); #%<-- rotation about the positive y-axis. ;
    delta_temp_ = mmvm( Ry.T , mmvm( Rz.T , delta_orig_ ) );
    S_k_p_form_ = torch.exp(+2*pi*i*(pole_k_c_0_*delta_temp_[0].item() + pole_k_c_1_*delta_temp_[1].item() + pole_k_c_2_*delta_temp_[2].item())).to(dtype=torch.complex64);
    S_k_p_form__[nS,:] = S_k_p_form_;
    S_k_q_form_ = interp_p_to_q(n_k_p_r,n_w_,n_w_sum,S_k_p_form_);
    S_k_q_form__[nS,:] = S_k_q_form_;
#end;%for nS=0:n_S-1;
tmp_t=toc(tmp_t);
if (flag_verbose>0): disp(sprintf(' %% S_k_q_form__: %0.3fs',tmp_t)); #end;
fnorm_disp(flag_verbose,'S_k_p_form__',S_k_p_form__,'S_k_p_wkS__',S_k_p_wkS__);
fnorm_disp(flag_verbose,'S_k_q_form__',S_k_q_form__,'S_k_q_wkS__',S_k_q_wkS__);
#%%%%%%%%;
if (flag_verbose>0): disp(sprintf(' %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;')); #end;
if (flag_verbose>0): disp(sprintf(' %% A simple test assuming isotropic CTF: ;')); #end;
if (flag_verbose>0): disp(sprintf(' %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;')); #end;
#%%%%%%%%;
delta_r_max = 0.05; n_delta_v_requested = 64;
delta_r_p = 0.05;
delta_r_s = delta_r_max/np.maximum(1e-12,np.sqrt(2*np.log(1/np.maximum(1e-12,delta_r_p))));
delta_r_N = delta_r_max * (2*pi*k_p_r_max) / (pi*np.sqrt(2));
svd_eps = 1e-12;
tmp_t=tic();
r8_delta_r_max = float(delta_r_max); r8_svd_eps = float(svd_eps);
parameter_FTK = {'type':'FTK'};
parameter_FTK['r8_delta_r_max'] = r8_delta_r_max;
parameter_FTK['r8_svd_eps'] = r8_svd_eps;
parameter_FTK['n_delta_v_requested'] = n_delta_v_requested;
parameter_FTK,FTK = tfh_FTK_4(parameter_FTK,n_k_p_r,k_p_r_,k_p_r_max);
tmp_t=toc(tmp_t);
if (flag_verbose>0): disp(sprintf(' %% FTK: %0.3fs',tmp_t)); #end;
FTK['flag_tf_vs_bf'] = flag_tf_vs_bf;
n_delta_v = FTK['n_delta_v'];
n_svd_l = FTK['n_svd_l'];
disp(sprintf(' %% p-val %0.4f delta_r_max %0.6f sigma %0.4f N_pixel %0.4f --> FTK.n_svd_l %d, n_delta_v_requested %d',delta_r_p,delta_r_max,delta_r_s,delta_r_N,FTK['n_svd_l'],n_delta_v_requested));
#%%%%%%%%;
#% design index_nCTF_from_nM_. ;
#%%%%%%%%;
n_M = n_S - 2; #%<-- just to check dimensions. ;
n_CTF = 2;
index_nCTF_from_nM_ = torch.fmod(torch.arange(n_M).to(dtype=torch.int32),n_CTF).to(dtype=torch.int32);
CTF_k_p_r_kC__ = torch.zeros(mtr((n_k_p_r,n_CTF))).to(dtype=torch.float32);
for nCTF in range(n_CTF):
    CTF_k_p_r_kC__[nCTF,:] = 1.5 + torch.sin(2*pi*(1+nCTF/np.maximum(1,n_CTF))*k_p_r_/np.maximum(1e-12,k_p_r_max)); #%<-- invertible. ;
#end;%for nCTF=0:n_CTF-1;
#%%%%%%%%;
#%  Construct CTF of same size as images. ;
#%%%%%%%%;
CTF_k_p_wkC__ = torch.zeros(mtr((n_w_sum,n_CTF))).to(dtype=torch.float32);
for nCTF in range(n_CTF):
    for nk_p_r in range(n_k_p_r):
        n_w = int(n_w_[nk_p_r].item());
        tmp_index_ = (int(n_w_csum_[nk_p_r].item()) + torch.arange(n_w)).to(dtype=torch.int32);
        tmp_i8_index_lhs_ = matlab_index_2d_0(n_w_sum,tmp_index_,n_CTF,nCTF);
        CTF_k_p_wkC__.ravel()[tmp_i8_index_lhs_] = CTF_k_p_r_kC__[nCTF,nk_p_r].item();
    #end;%for nk_p_r=0:n_k_p_r-1;
#end;%for nCTF=0:n_CTF-1;
CTF_inv_k_p_wkC__ = torch.tensor([1.0])/torch.maximum(torch.tensor([1e-12]),CTF_k_p_wkC__);
#%%%%%%%%;
#% Generate images to use. ;
#%%%%%%%%;
tmp_i8_index_rhs_ = matlab_index_2d_0(n_w_sum,':',n_S,torch.arange(n_M));
M_k_p_wkM__ = torch.reshape(S_k_p_form__.ravel()[tmp_i8_index_rhs_],mtr((n_w_sum,n_M)))
M_k_q_wkM__ = torch.reshape(S_k_q_form__.ravel()[tmp_i8_index_rhs_],mtr((n_w_sum,n_M)))
for nM in range(n_M):
    nCTF = int(index_nCTF_from_nM_[nM].item());
    M_k_p_wkM__[nM,:] = M_k_p_wkM__[nM,:].ravel()*CTF_inv_k_p_wkC__[nCTF,:].ravel();
    M_k_q_wkM__[nM,:] = M_k_q_wkM__[nM,:].ravel()*CTF_inv_k_p_wkC__[nCTF,:].ravel();
#end;%for nM=0:n_M-1;
#%%%%%%%%;
n_cluster = n_CTF;
index_ncluster_from_nCTF_ = torch.fmod(torch.arange(n_CTF).to(dtype=torch.int32),n_cluster).to(dtype=torch.int32); #%<-- give each nCTF its own cluster. ;
index_ncluster_from_nM_ = index_ncluster_from_nCTF_[index_nCTF_from_nM_];
index_nM_from_ncluster__ = cell(n_cluster);
n_index_nM_from_ncluster_ = torch.zeros(n_cluster).to(dtype=torch.int32);
for ncluster in range(n_cluster):
    index_nM_from_ncluster__[ncluster] = efind(index_ncluster_from_nM_==ncluster);
    n_index_nM_from_ncluster_[ncluster] = numel(index_nM_from_ncluster__[ncluster]);
#end;%for ncluster=0:n_cluster-1;
#%%%%%%%%;
#% design principal-modes. ;
#%%%%%%%%;
tmp_t=tic();
pm_n_UX_rank_max = n_k_p_r-1; #%<-- just to check dimensions.; 
pm_n_UX_rank_c_ = torch.maximum(torch.tensor([1]),torch.minimum(torch.tensor([n_k_p_r-1]),pm_n_UX_rank_max - torch.arange(n_cluster))).to(dtype=torch.int32);
pm_UX_knc___ = torch.zeros(mtr((n_k_p_r,pm_n_UX_rank_max,n_cluster))).to(dtype=torch.float32);
pm_X_weight_rc__ = torch.zeros(mtr((n_k_p_r,n_cluster)));
for ncluster in range(n_cluster):
    pm_n_UX_rank = int(pm_n_UX_rank_c_[ncluster].item());
    index_nM_from_ncluster_ = index_nM_from_ncluster__[ncluster];
    tmp_n_M = numel(index_nM_from_ncluster_);
    assert(tmp_n_M==int(n_index_nM_from_ncluster_[ncluster].item()));
    tmp_i8_index_rhs_ = matlab_index_2d_0(n_w_sum,':',n_M,index_nM_from_ncluster_);
    tmp_M_k_p_wkM__ = torch.reshape(M_k_p_wkM__.ravel()[tmp_i8_index_rhs_],mtr((n_w_sum,tmp_n_M)));
    (
        X_kk__,
        X_weight_r_,
    ) = principled_marching_empirical_cost_matrix_1(
        n_k_p_r,
        k_p_r_,
        weight_2d_k_p_r_,
        n_w_,
        tmp_n_M,
        tmp_M_k_p_wkM__,
    );
    tmp_UX_kn__,tmp_SX_k_,tmp_VX_kn__ = matlab_svds(X_kk__,pm_n_UX_rank);
    assert(size(tmp_UX_kn__,0)==n_k_p_r); assert(size(tmp_UX_kn__,1)==pm_n_UX_rank);
    tmp_i8_index_lhs_ = matlab_index_3d_0(n_k_p_r,':',size(pm_UX_knc___,1),torch.arange(pm_n_UX_rank),n_cluster,ncluster);
    tmp_i8_index_rhs_ = matlab_index_2d_0(n_k_p_r,':',pm_n_UX_rank,torch.arange(pm_n_UX_rank));
    pm_UX_knc___.ravel()[tmp_i8_index_lhs_] = tmp_UX_kn__.ravel()[tmp_i8_index_rhs_];
    pm_X_weight_rc__[ncluster,:] = X_weight_r_.ravel();
#end;%for ncluster=0:n_cluster-1;
tmp_t=toc(tmp_t);
if (flag_verbose>0): disp(sprintf(' %% pm_UX_knc___: %0.3fs',tmp_t)); #end;
#%%%%%%%%;
#% Now calculate the inner-products. ;
#%%%%%%%%;
tmp_t=tic();
parameter = {'type':'parameter'};
parameter['flag_gpu'] = flag_gpu;
(
    parameter,
    Z_SM_tfpm__,
    UX_CTF_S_l2_SM_tfpm__,
    UX_T_M_l2_SM_tfpm__,
    X_SM_tfpm__,
    delta_x_SM_tfpm__,
    delta_y_SM_tfpm__,
    gamma_z_SM_tfpm__,
    index_sub_SM_tfpm__,
    index_nM_from_ncluster__,
    n_index_nM_from_ncluster_,
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
    M_k_p_wkM__,
    n_cluster,
    index_ncluster_from_nCTF_,
    pm_n_UX_rank_c_,
    pm_UX_knc___,
    pm_X_weight_rc__,
    FTK,
)[:11];
tmp_t=toc(tmp_t);
if (flag_verbose>0): disp(sprintf(' %% X_SM_tfpm__: %0.3fs',tmp_t)); #end;
#%%%%%%%%;
#% Calculate true landscape of innerproducts for the same set of translations. ;
#%%%%%%%%;
UX_CTF_S_l2_SM_quad__ = torch.zeros(mtr((n_S,n_M))).to(dtype=torch.float32);
UX_T_M_l2_SM_quad__ = torch.zeros(mtr((n_S,n_M))).to(dtype=torch.float32);
X_SM_form__ = torch.zeros(mtr((n_S,n_M))).to(dtype=torch.float32);
X_SM_quad__ = torch.zeros(mtr((n_S,n_M))).to(dtype=torch.float32);
for nS in range(n_S):
    viewing_azimu_b = viewing_azimu_b_S_[nS].item();
    cb = np.cos(+viewing_azimu_b); sb = np.sin(+viewing_azimu_b);
    Rz = torch.reshape(torch.tensor([ +cb , +sb , 0 , -sb , +cb , 0 , 0 , 0 , 1 ]).to(dtype=torch.float32),mtr((3,3))); #%<-- rotation about the positive z-axis. ;
    viewing_polar_a = viewing_polar_a_S_[nS].item();
    ca = np.cos(+viewing_polar_a); sa = np.sin(+viewing_polar_a);
    Ry = torch.reshape(torch.tensor([ +ca , 0 , -sa , 0 , 1 , 0 , +sa , 0 , +ca ]).to(dtype=torch.float32),mtr((3,3))); #%<-- rotation about the positive y-axis. ;
    delta_temp_0_ = mmvm( Ry.T , mmvm( Rz.T , delta_orig_ ) );
    for nM in range(n_M):
        nCTF = int(index_nCTF_from_nM_[nM].item());
        CTF_k_p_wk_ = CTF_k_p_wkC__[nCTF,:].ravel();
        CTF_inv_k_p_wk_ = CTF_inv_k_p_wkC__[nCTF,:].ravel();
        image_viewing_azimu_b = viewing_azimu_b_S_[nM].item();
        cb = np.cos(+image_viewing_azimu_b); sb = np.sin(+image_viewing_azimu_b);
        Rz = torch.reshape(torch.tensor([ +cb , +sb , 0 , -sb , +cb , 0 , 0 , 0 , 1 ]).to(dtype=torch.float32),mtr((3,3))); #%<-- rotation about the positive z-axis. ;
        image_viewing_polar_a = viewing_polar_a_S_[nM].item();
        ca = np.cos(+image_viewing_polar_a); sa = np.sin(+image_viewing_polar_a);
        Ry = torch.reshape(torch.tensor([ +ca , 0 , -sa , 0 , 1 , 0 , +sa , 0 , +ca ]).to(dtype=torch.float32),mtr((3,3))); #%<-- rotation about the positive y-axis. ;
        delta_temp_1_ = mmvm( Ry.T , mmvm( Rz.T , delta_orig_ ) );
        #%%%%;
        gamma_z = gamma_z_SM_tfpm__[nM,nS].item();#%gamma_z = 2*pi*nw/np.maximum(1,n_w_max);
        cc = np.cos(+gamma_z); sc = np.sin(+gamma_z);
        R2 = torch.reshape(torch.tensor([+cc,+sc,-sc,+cc]),mtr((2,2))).to(dtype=torch.float32);
        delta_x = delta_x_SM_tfpm__[nM,nS].item();
        delta_y = delta_y_SM_tfpm__[nM,nS].item();
        delta_temp_0b_ = mmvm( R2 , delta_temp_0_[:2] ).ravel(); #%<-- rotate delta_temp_0_ by +gamma_z = rotate k by -gamma_z = rotate S_0 by +gamma_z. ;
        delta_temp_1b_ = delta_temp_1_[:2] - torch.tensor([delta_x,delta_y]); #%<-- translate delta_temp_1_ by -delta_ = multiply S_1 by exp(-2*pi*i*dot(k_,delta_)). ;
        #%%%%;
        S_k_p_form_ = torch.exp(+2*pi*i*(pole_k_c_0_*delta_temp_0b_[0].item() + pole_k_c_1_*delta_temp_0b_[1].item())).to(dtype=torch.complex64);
        S_k_p_temp_ = S_k_p_form_*CTF_k_p_wk_;
        M_k_p_form_ = torch.exp(+2*pi*i*(pole_k_c_0_*delta_temp_1b_[0].item() + pole_k_c_1_*delta_temp_1b_[1].item())).to(dtype=torch.complex64);
        M_k_p_temp_ = M_k_p_form_*CTF_inv_k_p_wk_;
        #%%%%;
        S_l2_quad = np.real(np.sqrt(torch.sum(mmvm( torch.reshape(torch.conj(S_k_p_temp_)*S_k_p_temp_,mtr((n_w_max,n_k_p_r))) , weight_2d_k_p_r_.to(dtype=torch.complex64) )).item()/np.maximum(1,n_w_max)));
        M_l2_quad = np.real(np.sqrt(torch.sum(mmvm( torch.reshape(torch.conj(M_k_p_temp_)*M_k_p_temp_,mtr((n_w_max,n_k_p_r))) , weight_2d_k_p_r_.to(dtype=torch.complex64) )).item()/np.maximum(1,n_w_max)));
        #%%%%;
        X_form = h2d(2*pi*k_p_r_max*fnorm(delta_temp_0b_ - delta_temp_1b_)).item()/(2*pi)**2 * (pi*k_p_r_max**2); #%<-- note sign of translation. ;
        X_SM_form__[nM,nS] = np.real(X_form) / np.maximum(1e-12,S_l2_quad*M_l2_quad) ;
        #%%%%;
        X_quad = torch.sum(mmvm( torch.reshape(torch.conj(S_k_p_temp_)*M_k_p_temp_,mtr((n_w_max,n_k_p_r))) , weight_2d_k_p_r_.to(dtype=torch.complex64) )).item()/np.maximum(1,n_w_max);
        X_SM_quad__[nM,nS] = np.real(X_quad) / np.maximum(1e-12,S_l2_quad*M_l2_quad) ;
        UX_CTF_S_l2_SM_quad__[nM,nS] = S_l2_quad**2;
        UX_T_M_l2_SM_quad__[nM,nS] = M_l2_quad**2;
        #%%%%;
    #end;%for nM=0:n_M-1;
#end;%for nS=0:n_S-1;
#%%%%%%%%;
fnorm_disp(flag_verbose,'X_SM_form__',X_SM_form__,'X_SM_tfpm__',X_SM_tfpm__);
fnorm_disp(flag_verbose,'X_SM_quad__',X_SM_quad__,'X_SM_tfpm__',X_SM_tfpm__);
fnorm_disp(flag_verbose,'UX_CTF_S_l2_SM_quad__',UX_CTF_S_l2_SM_quad__,'UX_CTF_S_l2_SM_tfpm__',UX_CTF_S_l2_SM_tfpm__);
fnorm_disp(flag_verbose,'UX_T_M_l2_SM_quad__',UX_T_M_l2_SM_quad__,'UX_T_M_l2_SM_tfpm__',UX_T_M_l2_SM_tfpm__);
#%%%%%%%%;

#%%%%%%%%;
if (flag_verbose>0): disp(sprintf(' %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;')); #end;
if (flag_verbose>0): disp(sprintf(' %% A more complicated test still assuming isotropic CTF: ;')); #end;
if (flag_verbose>0): disp(sprintf(' %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;')); #end;
#%%%%%%%%;
n_2 = 2;
#%%%%;
n_source_S = 7; delta_source_2sS___ = 0.10*(2*torch.rand(mtr((n_2,n_source_S,n_S)))-1).to(dtype=torch.float32);
for nS in range(n_S):
    S_k_p_wkS__[nS,:] = torch.zeros(n_w_sum).to(dtype=torch.complex64);
    for nsource_S in range(n_source_S):
        tmp_wk_ = torch.exp(+2*pi*i*(k_c_0_wk_*delta_source_2sS___[nS,nsource_S,0].item() + k_c_1_wk_*delta_source_2sS___[nS,nsource_S,1].item())).to(dtype=torch.complex64);
        S_k_p_wkS__[nS,:] = S_k_p_wkS__[nS,:] + tmp_wk_;
    #end;%for nsource_S=0:n_source_S-1;
#end;%for nS=0:n_S-1;
#%%%%;
n_source_M = 8; delta_source_2sM___ = 0.10*(2*torch.rand(mtr((n_2,n_source_M,n_M)))-1).to(dtype=torch.float32);
for nM in range(n_M):
    M_k_p_wkM__[nM,:] = torch.zeros(n_w_sum).to(dtype=torch.complex64);
    for nsource_M in range(n_source_M):
        tmp_wk_ = torch.exp(+2*pi*i*(k_c_0_wk_*delta_source_2sM___[nM,nsource_M,0].item() + k_c_1_wk_*delta_source_2sM___[nM,nsource_M,1].item())).to(dtype=torch.complex64);
        M_k_p_wkM__[nM,:] = M_k_p_wkM__[nM,:].ravel() + tmp_wk_;
    #end;%for nsource_M=0:n_source_M-1;
#end;%for nM=0:n_M-1;
#%%%%;
n_source_C = 9; source_sC__ = torch.rand(mtr((n_source_C,n_CTF))).to(dtype=torch.float32);
for nCTF in range(n_CTF):
    CTF_k_p_wkC__[nCTF,:] = torch.zeros(n_w_sum).to(dtype=torch.float32);
    for nsource_C in range(n_source_C):
        tmp_wk_ = source_sC__[nCTF,nsource_C].item() * torch.cos(2*pi*nsource_C*k_p_r_wk_/np.maximum(1e-12,k_p_r_max));
        CTF_k_p_wkC__[nCTF,:] = CTF_k_p_wkC__[nCTF,:].ravel() + tmp_wk_;
    #end;%for nsource_C=0:n_source_C-1;
#end;%for nCTF=0:n_CTF-1;
CTF_k_p_r_kC__ = torch.reshape(torch.mean(torch.reshape(CTF_k_p_wkC__,mtr((n_w_max,n_k_p_r,n_CTF))),2-0),mtr((n_k_p_r,n_CTF)));
#%%%%%%%%;
#% recalculate principal-modes. ;
#%%%%%%%%;
tmp_t=tic();
#%pm_n_UX_rank = n_k_p_r-1; #%<-- just to check dimensions.; 
pm_n_UX_rank_max = int(np.minimum(n_k_p_r-1,5)); #%<-- to check accuracy with lossy projection. ;
pm_n_UX_rank_c_ = torch.maximum(torch.tensor([1]),torch.minimum(torch.tensor([n_k_p_r-1]),pm_n_UX_rank_max - torch.arange(n_cluster))).to(dtype=torch.int32);
pm_UX_knc___ = torch.zeros(mtr((n_k_p_r,pm_n_UX_rank_max,n_cluster))).to(dtype=torch.float32);
pm_X_weight_rc__ = torch.zeros(mtr((n_k_p_r,n_cluster))).to(dtype=torch.float32);
for ncluster in range(n_cluster):
    pm_n_UX_rank = int(pm_n_UX_rank_c_[ncluster]);
    index_nM_from_ncluster_ = index_nM_from_ncluster__[ncluster];
    tmp_n_M = numel(index_nM_from_ncluster_);
    assert(tmp_n_M==n_index_nM_from_ncluster_[ncluster].item());
    tmp_i8_index_rhs_ = matlab_index_2d_0(n_w_sum,':',n_M,index_nM_from_ncluster_);
    tmp_M_k_p_wkM__ = torch.reshape(M_k_p_wkM__.ravel()[tmp_i8_index_rhs_],mtr((n_w_sum,tmp_n_M)));
    (
        X_kk__,
        X_weight_r_,
    ) = principled_marching_empirical_cost_matrix_1(
        n_k_p_r,
        k_p_r_,
        weight_2d_k_p_r_,
        n_w_,
        tmp_n_M,
        tmp_M_k_p_wkM__,
    );
    tmp_UX_kn__,tmp_SX_k_,tmp_VX_kn__ = matlab_svds(X_kk__,pm_n_UX_rank);
    assert(size(tmp_UX_kn__,0)==n_k_p_r); assert(size(tmp_UX_kn__,1)==pm_n_UX_rank);
    tmp_i8_index_lhs_ = matlab_index_3d_0(n_k_p_r,':',size(pm_UX_knc___,1),torch.arange(pm_n_UX_rank),n_cluster,ncluster);
    tmp_i8_index_rhs_ = matlab_index_2d_0(n_k_p_r,':',pm_n_UX_rank,torch.arange(pm_n_UX_rank));
    pm_UX_knc___.ravel()[tmp_i8_index_lhs_] = tmp_UX_kn__.ravel()[tmp_i8_index_rhs_];
    pm_X_weight_rc__[ncluster,:] = X_weight_r_.ravel();
#end;%for ncluster=0:n_cluster-1;
tmp_t=toc(tmp_t);
if (flag_verbose>0): disp(sprintf(' %% pm_UX_knc___: %0.3fs',tmp_t)); #end;
#%%%%%%%%;
#% Now calculate the inner-products. ;
#%%%%%%%%;
tmp_t=tic();
parameter = {'type':'parameter'};
parameter['flag_gpu'] = flag_gpu;
(
    parameter,
    Z_SM_tfpm__,
    UX_CTF_S_l2_SM_tfpm__,
    UX_T_M_l2_SM_tfpm__,
    X_SM_tfpm__,
    delta_x_SM_tfpm__,
    delta_y_SM_tfpm__,
    gamma_z_SM_tfpm__,
    index_sub_SM_tfpm__,
    index_nM_from_ncluster__,
    n_index_nM_from_ncluster_,
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
    M_k_p_wkM__,
    n_cluster,
    index_ncluster_from_nCTF_,
    pm_n_UX_rank_c_,
    pm_UX_knc___,
    pm_X_weight_rc__,
    FTK,
)[:11];
tmp_t=toc(tmp_t);
if (flag_verbose>0): disp(sprintf(' %% X_SM_tfpm__: %0.3fs',tmp_t)); #end;
#%%%%%%%%;
#% Calculate true landscape of innerproducts for the same set of translations. ;
#%%%%%%%%;
UX_CTF_S_l2_SM_quad__ = torch.zeros(mtr((n_S,n_M))).to(dtype=torch.float32);
UX_T_M_l2_SM_quad__ = torch.zeros(mtr((n_S,n_M))).to(dtype=torch.float32);
X_SM_quad__ = torch.zeros(mtr((n_S,n_M))).to(dtype=torch.float32);
Z_SM_quad__ = torch.zeros(mtr((n_S,n_M))).to(dtype=torch.float32);
UX_T_M_l2_SM_quad__ = torch.zeros(mtr((n_S,n_M))).to(dtype=torch.float32);
for nS in range(n_S):
    S_k_p_wk_ = S_k_p_wkS__[nS,:].ravel();
    for nM in range(n_M):
        M_k_p_wk_ = M_k_p_wkM__[nM,:].ravel();
        ncluster = int(index_ncluster_from_nM_[nM].item());
        pm_n_UX_rank = int(pm_n_UX_rank_c_[ncluster].item()); pm_n_w_max = n_w_max; pm_n_w_sum = pm_n_w_max*pm_n_UX_rank;
        tmp_i8_index_rhs_ = matlab_index_3d_0(n_k_p_r,':',size(pm_UX_knc___,1),torch.arange(pm_n_UX_rank),n_cluster,ncluster);
        pm_UX_kn__ = torch.reshape(pm_UX_knc___.ravel()[tmp_i8_index_rhs_],mtr((n_k_p_r,pm_n_UX_rank)));
        pm_X_weight_r_ = pm_X_weight_rc__[ncluster,:].ravel();
        pm_wUX_kn__ = torch.reshape(pm_X_weight_r_,mtr((n_k_p_r,1))) * pm_UX_kn__ ;
        nCTF = int(index_nCTF_from_nM_[nM].item());
        CTF_k_p_r_k_ = CTF_k_p_r_kC__[nCTF,:].ravel();
        CTF_k_p_wk_ = CTF_k_p_wkC__[nCTF,:].ravel();
        gamma_z = gamma_z_SM_tfpm__[nM,nS].item();
        delta_x = delta_x_SM_tfpm__[nM,nS].item();
        delta_y = delta_y_SM_tfpm__[nM,nS].item();
        RS_k_p_wk_ = rotate_p_to_p_fftw(n_k_p_r,n_w_,n_w_sum,S_k_p_wk_,+gamma_z);
        CTF_RS_k_p_wk_ = RS_k_p_wk_*CTF_k_p_wk_;
        UX_CTF_RS_k_p_wn_ = mmmm( torch.reshape(CTF_RS_k_p_wk_,mtr((n_w_max,n_k_p_r))) , pm_wUX_kn__.to(dtype=torch.complex64) ).ravel(); assert(numel(UX_CTF_RS_k_p_wn_)==pm_n_w_sum);
        UX_CTF_RS_l2 = torch.sum(torch.conj(UX_CTF_RS_k_p_wn_)*UX_CTF_RS_k_p_wn_).item() / np.maximum(1,pm_n_w_max) ;
        TM_k_p_wk_ = transf_p_to_p(n_k_p_r,k_p_r_,n_w_,n_w_sum,M_k_p_wk_,+delta_x,+delta_y);
        UX_TM_k_p_wn_ = mmmm( torch.reshape(TM_k_p_wk_,mtr((n_w_max,n_k_p_r))) , pm_wUX_kn__.to(dtype=torch.complex64) ).ravel(); assert(numel(UX_TM_k_p_wn_)==pm_n_w_sum);
        UX_TM_l2 = torch.sum(torch.conj(UX_TM_k_p_wn_)*UX_TM_k_p_wn_).item() / np.maximum(1,pm_n_w_max) ;
        UX_CTF_RS_k_p_UX_TM_k_p = torch.sum(torch.conj(UX_CTF_RS_k_p_wn_)*UX_TM_k_p_wn_).item() / np.maximum(1,pm_n_w_max) ;
        Z_quad = UX_CTF_RS_k_p_UX_TM_k_p;
        X_quad = Z_quad / np.maximum(1e-12,np.sqrt(UX_CTF_RS_l2)) / np.maximum(1e-12,np.sqrt(UX_TM_l2));
        Z_SM_quad__[nM,nS] = np.real(Z_quad);
        X_SM_quad__[nM,nS] = np.real(X_quad);
        UX_CTF_S_l2_SM_quad__[nM,nS] = float(UX_CTF_RS_l2);
        UX_T_M_l2_SM_quad__[nM,nS] = float(UX_TM_l2);
    #%%%%;
    #end;%for nM=0:n_M-1;
#end;%for nS=0:n_S-1;
#%%%%%%%%;
fnorm_disp(flag_verbose,'Z_SM_quad__',Z_SM_quad__,'Z_SM_tfpm__',Z_SM_tfpm__);
fnorm_disp(flag_verbose,'X_SM_quad__',X_SM_quad__,'X_SM_tfpm__',X_SM_tfpm__);
fnorm_disp(flag_verbose,'UX_CTF_S_l2_SM_quad__',UX_CTF_S_l2_SM_quad__,'UX_CTF_S_l2_SM_tfpm__',UX_CTF_S_l2_SM_tfpm__);
fnorm_disp(flag_verbose,'UX_T_M_l2_SM_quad__',UX_T_M_l2_SM_quad__,'UX_T_M_l2_SM_tfpm__',UX_T_M_l2_SM_tfpm__);
#%%%%%%%%;

#%%%%%%%%;
if (flag_verbose>0): disp(sprintf(' %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;')); #end;
if (flag_verbose>0): disp(sprintf(' %% repeat test with precalculation: still assuming isotropic CTF: ;')); #end;
if (flag_verbose>0): disp(sprintf(' %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;')); #end;
#%%%%%%%%;
tmp_t=tic();
if 'index_nM_to_update_' not in locals(): index_nM_to_update_=None; #end;
if 'M_k_q_wkM__' not in locals(): M_k_q_wkM__=None; #end;
if 'UX_T_M_l2_dM__' not in locals(): UX_T_M_l2_dM__=None; #end;
if 'UX_M_l2_M_' not in locals(): UX_M_l2_M_=None; #end;
if 'svd_V_UX_M_lwnM____' not in locals(): svd_V_UX_M_lwnM____=None; #end;
if 'index_nS_to_update_' not in locals(): index_nS_to_update_=None; #end;
if 'UX_CTF_S_k_q_wnS__' not in locals(): UX_CTF_S_k_q_wnS__=None; #end;
if 'UX_CTF_S_l2_S_' not in locals(): UX_CTF_S_l2_S_=None; #end;
parameter = {'type':'parameter'};
parameter['flag_gpu'] = flag_gpu;
index_nM_to_update_ = torch.arange(n_M).to(dtype=torch.int32);
index_nS_to_update_ = torch.arange(n_S).to(dtype=torch.int32);
parameter['flag_precompute_M_k_q_wkM__'] = int(1*1);
parameter['flag_precompute_UX_T_M_l2_dM__'] = int(1*1);
parameter['flag_precompute_UX_M_l2_M_'] = int(1*1);
parameter['flag_precompute_svd_V_UX_M_lwnM____'] = int(1*1);
parameter['flag_precompute_UX_CTF_S_k_q_wnS__'] = int(1*1);
parameter['flag_precompute_UX_CTF_S_l2_S_'] = int(1*1);
(
    parameter,
    tmp_Z_SM_tfpm__,
    tmp_UX_CTF_S_l2_SM_tfpm__,
    tmp_UX_T_M_l2_SM_tfpm__,
    tmp_X_SM_tfpm__,
    tmp_delta_x_SM_tfpm__,
    tmp_delta_y_SM_tfpm__,
    tmp_gamma_z_SM_tfpm__,
    tmp_index_sub_SM_tfpm__,
    tmp_index_nM_from_ncluster__,
    tmp_n_index_nM_from_ncluster_,
    tmp_M_k_q_wkM__,
    tmp_UX_T_M_l2_dM__,
    tmp_UX_M_l2_M_,
    tmp_svd_V_UX_M_lwnM____,
    tmp_UX_CTF_S_k_q_wnS__,
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
    M_k_p_wkM__,
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
tmp_t=toc(tmp_t);
if (flag_verbose>0): disp(sprintf(' %% X_SM_tfpm__: %0.3fs',tmp_t)); #end;
#%%%%%%%%;
fnorm_disp(flag_verbose,'Z_SM_tfpm__',Z_SM_tfpm__,'tmp_Z_SM_tfpm__',tmp_Z_SM_tfpm__,' %%<-- should be zero');
fnorm_disp(flag_verbose,'UX_CTF_S_l2_SM_tfpm__',UX_CTF_S_l2_SM_tfpm__,'tmp_UX_CTF_S_l2_SM_tfpm__',tmp_UX_CTF_S_l2_SM_tfpm__,' %%<-- should be zero');
fnorm_disp(flag_verbose,'UX_T_M_l2_SM_tfpm__',UX_T_M_l2_SM_tfpm__,'tmp_UX_T_M_l2_SM_tfpm__',tmp_UX_T_M_l2_SM_tfpm__,' %%<-- should be zero');
fnorm_disp(flag_verbose,'X_SM_tfpm__',X_SM_tfpm__,'tmp_X_SM_tfpm__',tmp_X_SM_tfpm__,' %%<-- should be zero');
fnorm_disp(flag_verbose,'delta_x_SM_tfpm__',delta_x_SM_tfpm__,'tmp_delta_x_SM_tfpm__',tmp_delta_x_SM_tfpm__,' %%<-- should be zero');
fnorm_disp(flag_verbose,'delta_y_SM_tfpm__',delta_y_SM_tfpm__,'tmp_delta_y_SM_tfpm__',tmp_delta_y_SM_tfpm__,' %%<-- should be zero');
fnorm_disp(flag_verbose,'gamma_z_SM_tfpm__',gamma_z_SM_tfpm__,'tmp_gamma_z_SM_tfpm__',tmp_gamma_z_SM_tfpm__,' %%<-- should be zero');
fnorm_disp(flag_verbose,'index_sub_SM_tfpm__.to(dtype=torch.float32)',index_sub_SM_tfpm__.to(dtype=torch.float32),'tmp_index_sub_SM_tfpm__.to(dtype=torch.float32)',tmp_index_sub_SM_tfpm__.to(dtype=torch.float32),' %%<-- should be zero');
#%%%%%%%%;
parameter_timing_printf(parameter);
