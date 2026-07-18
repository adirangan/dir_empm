from dir_empm.dir_matlab_macros import * ;
from dir_empm.sample_shell_6 import sample_shell_6 ;
from dir_empm.sample_sphere_7 import sample_sphere_7 ;
from dir_empm.get_weight_3d_1 import get_weight_3d_1 ;
from dir_empm.get_weight_2d_2 import get_weight_2d_2 ;
from dir_empm.convert_k_p_to_spharm_uniform_over_n_k_p_r_5 import convert_k_p_to_spharm_uniform_over_n_k_p_r_5 ;
from dir_empm.get_weight_Y_0 import get_weight_Y_0 ;
from dir_empm.plane_wave_expansion_1 import plane_wave_expansion_1 ;
from dir_empm.plane_wave_expansion_2 import plane_wave_expansion_2 ;
from dir_empm.local_yk__from_yk_ import local_yk__from_yk_ ;
from dir_empm.pm_template_3 import pm_template_3 ;
from dir_empm.transf_p_to_p import transf_p_to_p ;
from dir_empm.rotate_p_to_p_fftw import rotate_p_to_p_fftw ;
from dir_empm.qbp_uniform_over_n_k_p_r_10 import qbp_uniform_over_n_k_p_r_10 ;

flag_verbose=1; flag_disp=1;nf=0;
tolerance_machine = 1e-5;

if (flag_verbose>0): disp(sprintf(' %% testing qbp_uniform_over_n_k_p_r_10')); #end;
n_1 = 1; n_2 = 2; n_3 = 3;
k_eq_d_double = 1.0;
k_int = 48;
viewing_k_eq_d_double = 1.0;
n_w_int = 1.0;
n_source = 256;

#%%%%%%%%;
#% Now set up k-quadrature on sphere. ;
#%%%%%%%%;
k_p_r_max = k_int/(2*pi); k_eq_d = k_eq_d_double/(2*pi); str_L = 'L';
flag_uniform_over_n_k_p_r = 1; flag_uniform_over_polar_a = 0; #%<-- This is set to match test_ssnll_from_a_k_Y_12 ;
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
    _,
    _,
    _,
    _,
    n_polar_a_k_,
    polar_a_ka__,
    n_azimu_b_ka__,
) = sample_sphere_7(
    0*flag_verbose,
    k_p_r_max,
    k_eq_d,
    str_L,
    flag_uniform_over_n_k_p_r,
    flag_uniform_over_polar_a,
)[:20]; #%<-- sum(weight_3d_k_p_r_)*(4*pi) = (4/3)*pi*k_p_r_max^3 --> sum(weight_3d_k_p_r_) = (1/3)*k_p_r_max^3 ;
#%%%%%%%%;
#% extract outer shell for visualization. ;
#%%%%%%%%;
index_outer_shell_ = torch.arange(n_qk_csum_[n_k_p_r-1],n_qk_csum_[n_k_p_r-1+1]).to(dtype=torch.int32);
n_outer_shell = numel(index_outer_shell_);
k_outer_shell_r_q_ = k_p_r_qk_[index_outer_shell_].ravel();
assert(torch.std(k_outer_shell_r_q_).item()< tolerance_machine);
k_outer_shell_azimu_b_q_ = k_p_azimu_b_qk_[index_outer_shell_].ravel();
k_outer_shell_polar_a_q_ = k_p_polar_a_qk_[index_outer_shell_].ravel();
#%%%%%%%%;
#% Set up k-quadrature on disc. ;
#%%%%%%%%;
n_w_int = 1;
l_max_upb = matlab_scalar_round(2*pi*k_p_r_max);
n_w_max = n_w_int*2*(l_max_upb+1);
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
n_w_max = int(torch.max(n_w_).item());
n_w_sum = int(torch.sum(n_w_).item());
n_w_csum_ = cumsum_0(n_w_);
#%%%%%%%%;

#%%%%%%%%;
#% Now set up volume with plane-wave sources. ;
#%%%%%%%%;
delta_a_c_3s__ = torch.zeros(mtr((n_3,n_source))).to(dtype=torch.float32);
for nsource in range(n_source):
    tmp_t = nsource/np.maximum(1,n_source);
    tmp_r = 0.5*tmp_t**(0.25); tmp_a = 5*pi*tmp_t; tmp_b = 17*pi*tmp_t;
    tmp_x_0 = tmp_r*np.sin(tmp_a)*np.cos(tmp_b); tmp_x_1 = tmp_r*np.sin(tmp_a)*np.sin(tmp_b); tmp_x_2 = tmp_r*np.cos(tmp_a);
    delta_a_c_3s__[nsource,:] = torch.tensor([tmp_x_0,tmp_x_1,tmp_x_2]).to(dtype=torch.float32);
#end;%for nsource=0:n_source-1;
n_source = matlab_size(delta_a_c_3s__,1);
a_k_p_form_qk_ = torch.zeros(n_qk).to(dtype=torch.complex64);
for nsource in range(n_source):
    delta_a_c_ = delta_a_c_3s__[nsource,:].ravel();
    a_k_p_form_qk_ = a_k_p_form_qk_ + torch.exp(+i*2*pi*(k_c_0_qk_*delta_a_c_[0].item() + k_c_1_qk_*delta_a_c_[1].item() + k_c_2_qk_*delta_a_c_[2].item())).to(dtype=torch.complex64);
#end;%for nsource=0:n_source-1;
disp(sprintf(' %% fnorm(a_k_p_form_qk_): %0.16f',fnorm(a_k_p_form_qk_)));
#%%%%%%%%;

k_p_r_max = 48/(2*pi); k_eq_d = 1.0/(2*pi); str_L = 'L';
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
l_max_max = int(np.minimum(l_max_upb,1+np.ceil(2*pi*k_p_r_[n_k_p_r-1].item())));
n_w_max = n_w_int*2*(l_max_max+1); n_w_0in_ = n_w_max*torch.ones(n_k_p_r).to(dtype=torch.int32);
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
)[:7];
n_w_sum = int(torch.sum(n_w_).item()); n_w_csum_ = cumsum_0(n_w_);
#%%%%%%%%;

#%%%%%%%%;
#% Now convert to a_k_Y_yk_. ;
#%%%%%%%%;
(
    l_max_upb,
    l_max_,
    l_max_max,
    n_m_max,
    m_max_,
    n_y_,
    n_y_max,
    n_y_sum,
    n_y_csum_,
    Y_l_val_yk_,
    Y_m_val_yk_,
    Y_k_val_yk_,
    weight_Y_yk_,
) = get_weight_Y_0(
    flag_verbose,
    n_k_p_r,
    k_p_r_,
    k_p_r_max,
    weight_3d_k_p_r_,
)[:13];
#%%%%%%%%;
tmp_t=tic();
(
    a_k_Y_quad_yk_,
) = convert_k_p_to_spharm_uniform_over_n_k_p_r_5(
    flag_verbose,
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
    a_k_p_form_qk_,
)[:1];
tmp_t = toc(tmp_t); disp(sprintf(' %% convert_k_p_to_spharm_uniform_over_n_k_p_r_5 time %0.2fs',tmp_t));
#%%%%%%%%;
tmp_t=tic();
a_k_Y_form_yk_ = torch.zeros(n_y_sum).to(dtype=torch.complex64);
_,a_k_Y_form_yk_ = plane_wave_expansion_2(None,n_k_p_r,k_p_r_,n_source,delta_a_c_3s__,l_max_)[:2];
tmp_t = toc(tmp_t); disp(sprintf(' %% plane_wave_expansion_2 time %0.2fs',tmp_t));
fnorm_disp(flag_verbose,'a_k_Y_form_yk_',a_k_Y_form_yk_,'a_k_Y_quad_yk_',a_k_Y_quad_yk_,' %%<-- should be small');
#%%%%%%%%;

#%%%%%%%%;
#% select some viewing-angles and templates. ;
#%%%%%%%%;
tmp_t=tic();
viewing_euler_k_eq_d = viewing_k_eq_d_double*1.0/max(1e-12,k_p_r_max);
template_inplane_k_eq_d = -1; flag_uniform_over_polar_a = 0;
(
    template_wkS___,
    n_w,
    n_S,
    viewing_azimu_b_S_,
    viewing_polar_a_S_,
    viewing_weight_S_,
    n_viewing_polar_a,
    viewing_polar_a_,
    n_viewing_azimu_b_,
) = pm_template_3(
    flag_verbose,
    l_max_max,
    n_k_p_r,
    torch.reshape(local_yk__from_yk_(n_k_p_r,l_max_,a_k_Y_form_yk_),mtr((n_y_max,n_k_p_r))),
    viewing_euler_k_eq_d,
    template_inplane_k_eq_d,
    n_w_max,
)[:9];
S_k_p_wkS__ = torch.reshape(template_wkS___,mtr((n_w_sum,n_S)));
tmp_t = toc(tmp_t); disp(sprintf(' %% convert_k_p_to_x_c_1 time %0.2fs',tmp_t));
#%%%%%%%%;

#%%%%%%%%;
#% define rotations in 2d and 3d. ;
#%%%%%%%%;

def R2(gamma_z):
    R2__ = torch.zeros(mtr((n_2,n_2))).to(dtype=torch.float32);
    na=0;
    #R2__.ravel()[na] = +np.cos(gamma_z); na=na+1;
    #R2__.ravel()[na] = +np.sin(gamma_z); na=na+1;
    #R2__.ravel()[na] = -np.sin(gamma_z); na=na+1;
    #R2__.ravel()[na] = +np.cos(gamma_z); na=na+1;
    R2__ = matlab_assign_lhs_from_rhs_(R2__,na, +np.cos(gamma_z)); na=na+1;
    R2__ = matlab_assign_lhs_from_rhs_(R2__,na, +np.sin(gamma_z)); na=na+1;
    R2__ = matlab_assign_lhs_from_rhs_(R2__,na, -np.sin(gamma_z)); na=na+1;
    R2__ = matlab_assign_lhs_from_rhs_(R2__,na, +np.cos(gamma_z)); na=na+1;
    assert(na==4);
    return(R2__);
#end;def;

def Rz(azimu_b):
    Rz__ = torch.zeros(mtr((n_3,n_3))).to(dtype=torch.float32);
    na=0;
    #Rz__.ravel()[na] = +np.cos(azimu_b); na=na+1;
    #Rz__.ravel()[na] = +np.sin(azimu_b); na=na+1;
    #Rz__.ravel()[na] =                0; na=na+1;
    #Rz__.ravel()[na] = -np.sin(azimu_b); na=na+1;
    #Rz__.ravel()[na] = +np.cos(azimu_b); na=na+1;
    #Rz__.ravel()[na] =                0; na=na+1;
    #Rz__.ravel()[na] =                0; na=na+1;
    #Rz__.ravel()[na] =                0; na=na+1;
    #Rz__.ravel()[na] =                1; na=na+1;
    Rz__ = matlab_assign_lhs_from_rhs_(Rz__,na, +np.cos(azimu_b)); na=na+1;
    Rz__ = matlab_assign_lhs_from_rhs_(Rz__,na, +np.sin(azimu_b)); na=na+1;
    Rz__ = matlab_assign_lhs_from_rhs_(Rz__,na,                0); na=na+1;
    Rz__ = matlab_assign_lhs_from_rhs_(Rz__,na, -np.sin(azimu_b)); na=na+1;
    Rz__ = matlab_assign_lhs_from_rhs_(Rz__,na, +np.cos(azimu_b)); na=na+1;
    Rz__ = matlab_assign_lhs_from_rhs_(Rz__,na,                0); na=na+1;
    Rz__ = matlab_assign_lhs_from_rhs_(Rz__,na,                0); na=na+1;
    Rz__ = matlab_assign_lhs_from_rhs_(Rz__,na,                0); na=na+1;
    Rz__ = matlab_assign_lhs_from_rhs_(Rz__,na,                1); na=na+1;
    assert(na==9);
    return(Rz__);
#end;def;

def Ry(polar_a):
    Ry__ = torch.zeros(mtr((n_3,n_3))).to(dtype=torch.float32);
    na=0;
    #Ry__.ravel()[na] = +np.cos(polar_a); na=na+1;
    #Ry__.ravel()[na] =                0; na=na+1;
    #Ry__.ravel()[na] = -np.sin(polar_a); na=na+1;
    #Ry__.ravel()[na] =                0; na=na+1;
    #Ry__.ravel()[na] =                1; na=na+1;
    #Ry__.ravel()[na] =                0; na=na+1;
    #Ry__.ravel()[na] = +np.sin(polar_a); na=na+1;
    #Ry__.ravel()[na] =                0; na=na+1;
    #Ry__.ravel()[na] = +np.cos(polar_a); na=na+1;
    Ry__ = matlab_assign_lhs_from_rhs_(Ry__,na, +np.cos(polar_a)); na=na+1;
    Ry__ = matlab_assign_lhs_from_rhs_(Ry__,na,                0); na=na+1;
    Ry__ = matlab_assign_lhs_from_rhs_(Ry__,na, -np.sin(polar_a)); na=na+1;
    Ry__ = matlab_assign_lhs_from_rhs_(Ry__,na,                0); na=na+1;
    Ry__ = matlab_assign_lhs_from_rhs_(Ry__,na,                1); na=na+1;
    Ry__ = matlab_assign_lhs_from_rhs_(Ry__,na,                0); na=na+1;
    Ry__ = matlab_assign_lhs_from_rhs_(Ry__,na, +np.sin(polar_a)); na=na+1;
    Ry__ = matlab_assign_lhs_from_rhs_(Ry__,na,                0); na=na+1;
    Ry__ = matlab_assign_lhs_from_rhs_(Ry__,na, +np.cos(polar_a)); na=na+1;
    assert(na==9);
    return(Ry__);
#end;def;

#%%%%%%%%;
#% Now test a few templates analytically. ;
#%%%%%%%%;
n_test = 8;
for ntest in range(n_test):
    nS = int(np.maximum(0,np.minimum(n_S-1,np.floor(n_S*ntest/max(1,n_test)))));
    tmp_azimu_b = viewing_azimu_b_S_[nS].item();
    tmp_polar_a = viewing_polar_a_S_[nS].item();
    tmp_gamma_z = 0.0;
    tmp_R__ = mmmm( Rz(-tmp_gamma_z) , mmmm( Ry(-tmp_polar_a) , Rz(-tmp_azimu_b) ) );
    R_k_p_wk_ = torch.zeros(n_w_sum).to(dtype=torch.float32);
    for nsource in range(n_source):
        tmp_delta_ = mmvm( tmp_R__ , delta_a_c_3s__[nsource,:].ravel() );
        R_k_p_wk_ = R_k_p_wk_ + torch.exp(+i*2*pi*(k_c_0_wk_*tmp_delta_[0].item() + k_c_1_wk_*tmp_delta_[1].item()));
    #end;%for nsource=0:n_source-1;
    S_k_p_wk_ = S_k_p_wkS__[nS,:].ravel();
    fnorm_disp(flag_verbose,'S_k_p_wk_',S_k_p_wk_,'R_k_p_wk_',R_k_p_wk_,' %%<-- should be small');
#end;%for ntest=0:n_test-1;
#%%%%%%%%;

#%%%%;
#% Now we generate a selection of synthetic-images, copied from the templates, ;
#% as well as a collection of anisotropic planar-CTF-functions. ;
#%%%%;
n_M = n_S*2;
index_nS_from_nM_ = torch.maximum(torch.tensor([0]),torch.minimum(torch.tensor([n_S-1]),torch.floor(torch.fmod(torch.arange(n_M),n_S)))).to(dtype=torch.int32);
euler_polar_a_M_ = viewing_polar_a_S_.ravel()[index_nS_from_nM_];
euler_azimu_b_M_ = viewing_azimu_b_S_.ravel()[index_nS_from_nM_];
euler_gamma_z_M_ = 2*pi*torch.arange(n_M).to(dtype=torch.float32)/np.maximum(1,n_M) + pi/2;
image_delta_x_M_ = +0.05*(torch.fmod(torch.arange(n_M).to(dtype=torch.float32),11)-5)/5;
image_delta_y_M_ = +0.07*(torch.fmod(torch.arange(n_M).to(dtype=torch.float32),15)-7)/7;
n_CTF = 8; #%<-- some number of CTF-functions. ;
CTF_phi_C_ = torch.zeros(n_CTF).to(dtype=torch.float32);
CTF_k_p_wkC__ = torch.zeros(mtr((n_w_sum,n_CTF))).to(dtype=torch.float32);
for nCTF in range(n_CTF):
    CTF_phi = 2*pi*nCTF/np.maximum(1,n_CTF) + pi/2;
    CTF_phi_C_[nCTF] = CTF_phi;
    CTF_k_p_wk_ = 2*k_p_r_wk_ * torch.cos(k_p_w_wk_ - CTF_phi);
    CTF_k_p_wkC__[nCTF,:] = CTF_k_p_wk_;
#end;%for nCTF=0:n_CTF-1;
index_nCTF_from_nM_ = torch.fmod(torch.arange(n_M).to(dtype=torch.float32),n_CTF).to(dtype=torch.int32);
tmp_q_ = torch.concatenate((torch.arange(0,+n_w_max/2-1+1),torch.arange(-n_w_max/2,-1+1)),0).to(dtype=torch.int32);
M_k_p_wkM__ = torch.zeros(mtr((n_w_sum,n_M))).to(dtype=torch.complex64);
tmp_i8_index_rhs_wkC_ = matlab_index_2d_0(n_w_sum,':',n_CTF,index_nCTF_from_nM_);
tmp_i8_index_rhs_wkS_ = matlab_index_2d_0(n_w_sum,':',n_S,index_nS_from_nM_);
M_k_p_wkM__ = torch.reshape(transf_p_to_p(n_k_p_r,k_p_r_,n_w_,n_w_sum,torch.reshape(CTF_k_p_wkC__.ravel()[tmp_i8_index_rhs_wkC_],mtr((n_w_sum,n_M))) * torch.reshape(rotate_p_to_p_fftw(n_k_p_r,n_w_,n_w_sum,torch.reshape(S_k_p_wkS__.ravel()[tmp_i8_index_rhs_wkS_],mtr((n_w_sum,n_M))),+euler_gamma_z_M_),mtr((n_w_sum,n_M))),-image_delta_x_M_,-image_delta_y_M_),mtr((n_w_sum,n_M))).to(dtype=torch.complex64);
#%%%%%%%%;

#%%%%%%%%;
#% Now we compare reconstruction. ;
#%%%%%%%%;
a_k_Y_relerror = lambda a_k_Y_test_yk_: np.sqrt( torch.sum(torch.abs(a_k_Y_form_yk_ - a_k_Y_test_yk_)**2 * weight_Y_yk_).item() /np.maximum(1e-12,torch.sum(torch.abs(a_k_Y_form_yk_)**2 * weight_Y_yk_).item()) ) ;
#%%%%%%%%;
tmp_t=tic();
qbp_eps = tolerance_machine;
(
    a_k_Y_qbpu_yk_,
) = qbp_uniform_over_n_k_p_r_10(
    qbp_eps,
    n_k_p_r,
    k_p_r_,
    l_max_,
    n_w_,
    n_M,
    M_k_p_wkM__,
    index_nCTF_from_nM_,
    CTF_k_p_wkC__,
    euler_polar_a_M_,
    euler_azimu_b_M_,
    euler_gamma_z_M_,
    +image_delta_x_M_,
    +image_delta_y_M_,
)[:1];
tmp_t = toc(tmp_t); disp(sprintf(' %% qbp_uniform_over_n_k_p_r_10 time %0.2fs',tmp_t));
fnorm_disp(flag_verbose,'a_k_Y_form_yk_',a_k_Y_form_yk_,'a_k_Y_qbpu_yk_',a_k_Y_qbpu_yk_);
if (flag_verbose>0): disp(sprintf(' %% a_k_Y_form_yk_ vs a_k_Y_qbpu_yk_: volumetric-relative-error: %0.16f',a_k_Y_relerror(a_k_Y_qbpu_yk_))); #end;
#%%%%%%%%;
