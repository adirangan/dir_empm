from dir_empm.dir_matlab_macros import * ;
from dir_empm.cg_rhs_2 import cg_rhs_2 ;
from dir_empm.sample_sphere_7 import sample_sphere_7 ;
from dir_empm.sample_shell_6 import sample_shell_6 ;
from dir_empm.get_weight_2d_2 import get_weight_2d_2 ;
from dir_empm.rotate_p_to_p_fftw import rotate_p_to_p_fftw ;
from dir_empm.xxnufft3d3 import xxnufft3d3 ;
from dir_empm.interp_k_p_to_x_c_xxnufft import interp_k_p_to_x_c_xxnufft ;
from dir_empm.tfh_FTK_2 import tfh_FTK_2 ;
from dir_empm.transf_p_to_p import transf_p_to_p ;
from dir_empm.I_PxxP import I_PxxP ;
from dir_empm.plane_wave_expansion_1 import plane_wave_expansion_1 ;
from dir_empm.convert_k_p_to_spharm_4 import convert_k_p_to_spharm_4 ;
from dir_empm.convert_k_p_to_spharm_uniform_over_n_k_p_r_5 import convert_k_p_to_spharm_uniform_over_n_k_p_r_5 ;
from dir_empm.convert_spharm_to_k_p_4 import convert_spharm_to_k_p_4 ;
from dir_empm.convert_spharm_to_k_p_uniform_over_n_k_p_r_5 import convert_spharm_to_k_p_uniform_over_n_k_p_r_5 ;
from dir_empm.local_yk__from_yk_ import local_yk__from_yk_ ;
from dir_empm.local_yk_from_yk__ import local_yk_from_yk__ ;
from dir_empm.pm_template_2 import pm_template_2 ;
from dir_empm.h2d import h2d ;
from dir_empm.h3d import h3d ;
from dir_empm.interp_p_to_q import interp_p_to_q ;
from dir_empm.I_xPPx_0 import I_xPPx_0 ;
from dir_empm.principled_marching_empirical_cost_matrix_1 import principled_marching_empirical_cost_matrix_1 ;
from dir_empm.tpmh_VUXM_lwnM____3 import tpmh_VUXM_lwnM____3 ;
from dir_empm.tfpmh_UX_T_M_l2_dM__1 import tfpmh_UX_T_M_l2_dM__1 ;

str_thisfunction = 'test_transforms_8_dr' ;
flag_verbose = 1 ; # verbosity level.

r'''
k_int = 16; %<-- highest frequency (2*pi*k_p_r_max), watch out for aliasing! ;
k_eq_d_double = 1.0; %<-- prefactor for k_eq_d, determines density of sampling in frequency-space. ;
template_k_eq_d_double = 0.5; %<-- prefactor for template_k_eq_d, determines density of viewing-angles on the sphere. ;
n_w_int = 1.0; %<-- prefactor for n_w_max, determines the number of distinct angles (i.e., n_gamma_z) used in frequency-space 2d-polar-grid. ;
n_x_c = max(64,2*k_int); %<-- the number of 'pixels' on a side in the real-space cartesian-grid. ;
'''

k_int = 16  # highest frequency (2*pi*k_p_r_max), watch out for aliasing!
k_eq_d_double = 1.0  # prefactor for k_eq_d, determines density of sampling in frequency-space.
template_k_eq_d_double = 0.5  # prefactor for template_k_eq_d, determines density of viewing-angles on the sphere.
n_w_int = 1.0  # prefactor for n_w_max, determines the number of distinct angles (i.e., n_gamma_z) used in frequency-space 2d-polar-grid.
n_x_c = max(64, 2 * k_int)  # the number of 'pixels' on a side in the real-space cartesian-grid.

r'''
%%%%%%%%;
% Define spatial grid. ;
%%%%%%%%;
half_diameter_x_c = 1.0d0;
diameter_x_c = 2.0d0*half_diameter_x_c;
x_p_r_max = half_diameter_x_c;
x_c_0_ = linspace(-x_p_r_max,+x_p_r_max,n_x_c); d_x_0 = mean(diff(x_c_0_));
x_c_1_ = linspace(-x_p_r_max,+x_p_r_max,n_x_c); d_x_1 = mean(diff(x_c_1_));
x_c_2_ = linspace(-x_p_r_max,+x_p_r_max,n_x_c); d_x_2 = mean(diff(x_c_2_));
[x_c_0___,x_c_1___,x_c_2___] = ndgrid(x_c_0_,x_c_1_,x_c_2_); n_xxx_c = n_x_c^3;
weight_xxx_c = d_x_0 * d_x_1 * d_x_2 ;
'''

# Define spatial grid.
half_diameter_x_c = 1.0;
diameter_x_c = 2.0 * half_diameter_x_c;
x_p_r_max = half_diameter_x_c;
x_c_0_ = torch.linspace(-x_p_r_max, +x_p_r_max, n_x_c).to(torch.float32); d_x_0 = x_c_0_[1].item()-x_c_0_[0].item();
x_c_1_ = torch.linspace(-x_p_r_max, +x_p_r_max, n_x_c).to(torch.float32); d_x_1 = x_c_1_[1].item()-x_c_1_[0].item();
x_c_2_ = torch.linspace(-x_p_r_max, +x_p_r_max, n_x_c).to(torch.float32); d_x_2 = x_c_2_[1].item()-x_c_2_[0].item();
x_c_2___, x_c_1___, x_c_0___ = torch.meshgrid(x_c_2_, x_c_1_, x_c_0_, indexing='ij') ; n_xxx_c = n_x_c ** 3 ; #<-- reversed to match matlab. ;
weight_xxx_c = d_x_0 * d_x_1 * d_x_2 ;

r'''
%%%%%%%%;
% Now set up k-quadrature on sphere. ;
%%%%%%%%;
k_p_r_max = k_int/(2*pi); k_eq_d = k_eq_d_double/(2*pi); str_L = 'L';
flag_uniform_over_n_k_p_r = 1; %<-- we use the same discretization on each shell. ;
flag_uniform_over_polar_a = 0; %<-- however, we allow for different discretizations on each latitude. ;
[ ...
 n_qk ...
,n_qk_csum_ ...
,k_p_r_qk_ ...
,k_p_azimu_b_qk_ ...
,k_p_polar_a_qk_ ...
,weight_3d_k_p_qk_ ...
,weight_shell_qk_ ...
,n_k_p_r ...
,k_p_r_ ...
,weight_3d_k_p_r_ ...
,k_c_0_qk_ ...
,k_c_1_qk_ ...
,k_c_2_qk_ ...
,~ ...
,~ ...
,~ ...
,~ ...
,n_polar_a_k_ ...
,polar_a_ka__ ...
,n_azimu_b_ka__ ...
] = ...
sample_sphere_7( ...
 0*flag_verbose ...
,k_p_r_max ...
,k_eq_d ...
,str_L ...
,flag_uniform_over_n_k_p_r ...
,flag_uniform_over_polar_a ...
) ;
'''

# Set up k-quadrature on sphere.
k_p_r_max = k_int / (2 * pi);
k_eq_d = k_eq_d_double / (2 * pi);
str_L = 'L';
flag_uniform_over_n_k_p_r = 1;  # use same discretization on each shell
flag_uniform_over_polar_a = 0; # allow different discretizations on each latitude

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
    0 * flag_verbose,
    k_p_r_max,
    k_eq_d,
    str_L,
    flag_uniform_over_n_k_p_r,
    flag_uniform_over_polar_a,
) ;

r'''
%%%%%%%%;
% For eventual reconstruction we generate quadrature on shell. ;
%%%%%%%%;
qref_k_eq_d = k_eq_d/max(1e-12,k_p_r_max);
[ ...
 qref_n_shell ...
,qref_azimu_b_shell_ ...
,qref_polar_a_shell_ ...
,qref_weight_shell_ ...
,qref_k_c_0_shell_ ...
,qref_k_c_1_shell_ ...
,qref_k_c_2_shell_ ...
,qref_n_polar_a ...
,qref_polar_a_ ...
,qref_n_azimu_b_ ...
] = ...
sample_shell_6( ...
 1.0 ...
,qref_k_eq_d ...
,str_L ...
,flag_uniform_over_polar_a ...
);
%%%%%%%%;
'''

# For eventual reconstruction, generate quadrature on shell.
qref_k_eq_d = k_eq_d / max(1e-12, k_p_r_max)
(
    qref_n_shell,
    qref_azimu_b_shell_,
    qref_polar_a_shell_,
    qref_weight_shell_,
    qref_k_c_0_shell_,
    qref_k_c_1_shell_,
    qref_k_c_2_shell_,
    qref_n_polar_a,
    qref_polar_a_,
    qref_n_azimu_b_,
) = sample_shell_6(
    1.0,
    qref_k_eq_d,
    str_L,
    flag_uniform_over_polar_a,
) ;

r'''
tmp_index_ = [n_qk_csum_(end-1):n_qk_csum_(end)-1];
fnorm_disp(flag_verbose,'k_p_azimu_b_qk_(1+tmp_index_)',k_p_azimu_b_qk_(1+tmp_index_),'qref_azimu_b_shell_',qref_azimu_b_shell_,' %%<-- should be zero');
fnorm_disp(flag_verbose,'k_p_polar_a_qk_(1+tmp_index_)',k_p_polar_a_qk_(1+tmp_index_),'qref_polar_a_shell_',qref_polar_a_shell_,' %%<-- should be zero');
'''

tmp_index_ = torch.arange(n_qk_csum_[-2], n_qk_csum_[-1]).to(torch.int32);
fnorm_disp(flag_verbose,'k_p_azimu_b_qk_[tmp_index_]',k_p_azimu_b_qk_[tmp_index_],'qref_azimu_b_shell_',qref_azimu_b_shell_,' %%<-- should be zero') ;
fnorm_disp(flag_verbose,'k_p_polar_a_qk_[tmp_index_]',k_p_polar_a_qk_[tmp_index_],'qref_polar_a_shell_',qref_polar_a_shell_,' %%<-- should be zero') ;

r'''
%%%%%%%%;
% Now define two functions on the sphere, ;
% each a sum of a few plane-waves. ;
%%%%%%%%;
delta_a_c_3s__ = [  ...
 +1.5 , -0.5 ...
;-0.5 , -1.5 ...
;+0.3 , +2.0 ...
] / 2 / k_p_r_max ;
delta_b_c_3s__ = [  ...
 -0.5 , +0.8 ...
;-1.0 , +0.2 ...
;+1.2 , -0.7 ...
] / 2 / k_p_r_max ;
n_source = size(delta_a_c_3s__,2);
a_k_p_form_ = zeros(n_qk,1);
b_k_p_form_ = zeros(n_qk,1);
for nsource=0:n_source-1;
delta_a_c_ = delta_a_c_3s__(:,1+nsource);
a_k_p_form_ = a_k_p_form_ + exp(+i*2*pi*(k_c_0_qk_*delta_a_c_(1+0) + k_c_1_qk_*delta_a_c_(1+1) + k_c_2_qk_*delta_a_c_(1+2)));
delta_b_c_ = delta_b_c_3s__(:,1+nsource);
b_k_p_form_ = b_k_p_form_ + exp(+i*2*pi*(k_c_0_qk_*delta_b_c_(1+0) + k_c_1_qk_*delta_b_c_(1+1) + k_c_2_qk_*delta_b_c_(1+2)));
end;%for nsource=0:n_source-1;
%%%%%%%%;
% Define frequency-space cartesian-grid. ;
%%%%%%%%;
n_k_c = n_x_c + 2; %<-- just to check dimensions. ;
half_diameter_k_c = k_p_r_max;
diameter_k_c = 2.0d0*half_diameter_k_c;
k_p_r_max = half_diameter_k_c;
k_c_0_ = linspace(-k_p_r_max,+k_p_r_max,n_k_c);
k_c_1_ = linspace(-k_p_r_max,+k_p_r_max,n_k_c);
k_c_2_ = linspace(-k_p_r_max,+k_p_r_max,n_k_c);
[k_c_0___,k_c_1___,k_c_2___] = ndgrid(k_c_0_,k_c_1_,k_c_2_); n_kkk_c = n_k_c^3;
weight_kkk_c = (2*k_p_r_max/n_k_c)^3;
%%%%%%%%;
a_k_c_form___ = zeros(n_k_c,n_k_c,n_k_c);
b_k_c_form___ = zeros(n_k_c,n_k_c,n_k_c);
for nsource=0:n_source-1;
delta_a_c_ = delta_a_c_3s__(:,1+nsource);
a_k_c_form___ = a_k_c_form___ + exp(+i*2*pi*(k_c_0___*delta_a_c_(1+0) + k_c_1___*delta_a_c_(1+1) + k_c_2___*delta_a_c_(1+2)));
delta_b_c_ = delta_b_c_3s__(:,1+nsource);
b_k_c_form___ = b_k_c_form___ + exp(+i*2*pi*(k_c_0___*delta_b_c_(1+0) + k_c_1___*delta_b_c_(1+1) + k_c_2___*delta_b_c_(1+2)));
end;%for nsource=0:n_source-1;
%%%%%%%%;
'''


# Now define two functions on the sphere, each a sum of a few plane-waves.
delta_a_c_3s__ = torch.transpose(
    torch.tensor([
        [ +1.5, -0.5 ],
        [ -0.5, -1.5 ],
        [ +0.3, +2.0 ]
    ]) / (2 * k_p_r_max) 
, 1 , 0 ).to(torch.float32);
delta_b_c_3s__ = torch.transpose(
    torch.tensor([
        [ -0.5, +0.8 ],
        [ -1.0, +0.2 ],
        [ +1.2, -0.7 ]
    ]) / (2 * k_p_r_max) 
, 1 , 0 ).to(torch.float32);
n_source = size(delta_a_c_3s__,1);
a_k_p_form_ = torch.zeros(n_qk).to(dtype=torch.complex64);
b_k_p_form_ = torch.zeros(n_qk).to(dtype=torch.complex64);
for nsource in range(n_source):
    delta_a_c_ = delta_a_c_3s__[nsource,:];
    a_k_p_form_ += torch.exp(
        i * 2 * pi * (
            k_c_0_qk_ * delta_a_c_[0] +
            k_c_1_qk_ * delta_a_c_[1] +
            k_c_2_qk_ * delta_a_c_[2]
        )
    );
    delta_b_c_ = delta_b_c_3s__[nsource,:]
    b_k_p_form_ += torch.exp(
        i * 2 * pi * (
            k_c_0_qk_ * delta_b_c_[0] +
            k_c_1_qk_ * delta_b_c_[1] +
            k_c_2_qk_ * delta_b_c_[2]
        )
    );
#end;%for;

# Define frequency-space cartesian-grid.
n_k_c = n_x_c + 2;  # just to check dimensions
half_diameter_k_c = k_p_r_max;
diameter_k_c = 2.0 * half_diameter_k_c;
k_p_r_max = half_diameter_k_c;
k_c_0_ = torch.linspace(-k_p_r_max, +k_p_r_max, n_k_c).to(torch.float32);
k_c_1_ = torch.linspace(-k_p_r_max, +k_p_r_max, n_k_c).to(torch.float32);
k_c_2_ = torch.linspace(-k_p_r_max, +k_p_r_max, n_k_c).to(torch.float32);
k_c_2___, k_c_1___, k_c_0___ = torch.meshgrid(k_c_2_, k_c_1_, k_c_0_, indexing='ij') ; #<-- reversed to match matlab. ;
n_kkk_c = n_k_c ** 3 ;
weight_kkk_c = (2 * k_p_r_max / n_k_c) ** 3 ;

a_k_c_form___ = torch.zeros((n_k_c, n_k_c, n_k_c)).to(dtype=torch.complex64);
b_k_c_form___ = torch.zeros((n_k_c, n_k_c, n_k_c)).to(dtype=torch.complex64);
for nsource in range(n_source):
    delta_a_c_ = delta_a_c_3s__[nsource,:];
    a_k_c_form___ += torch.exp(
        i * 2 * pi * (
            k_c_0___ * delta_a_c_[0] +
            k_c_1___ * delta_a_c_[1] +
            k_c_2___ * delta_a_c_[2]
        )
    );
    delta_b_c_ = delta_b_c_3s__[nsource,:]
    b_k_c_form___ += torch.exp(
        i * 2 * pi * (
            k_c_0___ * delta_b_c_[0] +
            k_c_1___ * delta_b_c_[1] +
            k_c_2___ * delta_b_c_[2]
        )
    );
#end;%for;

r'''
%%%%%%%%;
% Now test k-quadrature on sphere. ;
%%%%%%%%;
I_a_quad = sum(a_k_p_form_.*weight_3d_k_p_qk_);
I_b_quad = sum(b_k_p_form_.*weight_3d_k_p_qk_);
I_a_form = 0;
I_b_form = 0;
for nsource=0:n_source-1;
delta_a_c_ = delta_a_c_3s__(:,1+nsource);
delta_b_c_ = delta_b_c_3s__(:,1+nsource);
tmp_kd = 2*pi*k_p_r_max*fnorm(delta_a_c_);
I_a_form = I_a_form + h3d_(tmp_kd)*k_p_r_max^3;
tmp_kd = 2*pi*k_p_r_max*fnorm(delta_b_c_);
I_b_form = I_b_form + h3d_(tmp_kd)*k_p_r_max^3;
end;%for nsource=0:n_source-1;
fnorm_disp(flag_verbose,'I_a_form',I_a_form,'I_a_quad',I_a_quad,' %%<-- should be <1e-6');
fnorm_disp(flag_verbose,'I_b_form',I_b_form,'I_b_quad',I_b_quad,' %%<-- should be <1e-6');
%%%%%%%%;
a_k_p_l2_quad = sum(conj(a_k_p_form_).*a_k_p_form_.*weight_3d_k_p_qk_);
a_k_p_l2_form = 0;
for nsource0=0:n_source-1;
for nsource1=0:n_source-1;
delta_a_c_0_ = delta_a_c_3s__(:,1+nsource0);
delta_a_c_1_ = delta_a_c_3s__(:,1+nsource1);
tmp_kd = 2*pi*k_p_r_max*fnorm(delta_a_c_0_ - delta_a_c_1_);
tmp_h3d = 4*pi/3; if abs(tmp_kd)>1e-12; tmp_h3d = h3d_(tmp_kd); end;
a_k_p_l2_form = a_k_p_l2_form + tmp_h3d*k_p_r_max^3;
end;%for nsource1=0:n_source-1;
end;%for nsource0=0:n_source-1;
fnorm_disp(flag_verbose,'a_k_p_l2_form',a_k_p_l2_form,'a_k_p_l2_quad',a_k_p_l2_quad,' %%<-- should be <1e-6');
%%%%%%%%;
b_k_p_l2_quad = sum(conj(b_k_p_form_).*b_k_p_form_.*weight_3d_k_p_qk_);
b_k_p_l2_form = 0;
for nsource0=0:n_source-1;
for nsource1=0:n_source-1;
delta_b_c_0_ = delta_b_c_3s__(:,1+nsource0);
delta_b_c_1_ = delta_b_c_3s__(:,1+nsource1);
tmp_kd = 2*pi*k_p_r_max*fnorm(delta_b_c_0_ - delta_b_c_1_);
tmp_h3d = 4*pi/3; if abs(tmp_kd)>1e-12; tmp_h3d = h3d_(tmp_kd); end;
b_k_p_l2_form = b_k_p_l2_form + tmp_h3d*k_p_r_max^3;
end;%for nsource1=0:n_source-1;
end;%for nsource0=0:n_source-1;
fnorm_disp(flag_verbose,'b_k_p_l2_form',b_k_p_l2_form,'b_k_p_l2_quad',b_k_p_l2_quad,' %%<-- should be <1e-6');
%%%%%%%%;
'''

# Now test k-quadrature on sphere.
I_a_quad = torch.sum(a_k_p_form_ * weight_3d_k_p_qk_).item();
I_b_quad = torch.sum(b_k_p_form_ * weight_3d_k_p_qk_).item();

I_a_form = 0;
I_b_form = 0;
for nsource in range(n_source):
    delta_a_c_ = delta_a_c_3s__[nsource,:];
    delta_b_c_ = delta_b_c_3s__[nsource,:];
    tmp_kd = 2 * pi * k_p_r_max * fnorm(delta_a_c_);
    I_a_form += h3d(tmp_kd).item() * k_p_r_max ** 3 ;
    tmp_kd = 2 * pi * k_p_r_max * fnorm(delta_b_c_);
    I_b_form += h3d(tmp_kd).item() * k_p_r_max ** 3 ;
#end;%for;

fnorm_disp(flag_verbose,'I_a_form',I_a_form,'I_a_quad',I_a_quad,' %%<-- should be <1e-6');
fnorm_disp(flag_verbose,'I_b_form',I_b_form,'I_b_quad',I_b_quad,' %%<-- should be <1e-6');

a_k_p_l2_quad = torch.sum(torch.conj(a_k_p_form_) * a_k_p_form_ * weight_3d_k_p_qk_).item();
a_k_p_l2_form = 0;
for nsource0 in range(n_source):
    for nsource1 in range(n_source):
        delta_a_c_0_ = delta_a_c_3s__[nsource0,:];
        delta_a_c_1_ = delta_a_c_3s__[nsource1,:];
        tmp_kd = 2 * pi * k_p_r_max * fnorm(delta_a_c_0_ - delta_a_c_1_);
        tmp_h3d = 4 * pi / 3 if np.abs(tmp_kd) <= 1e-12 else h3d(tmp_kd).item();
        a_k_p_l2_form += tmp_h3d * k_p_r_max ** 3 ;
    #end;%for nsource1;
#end;%for nsource0;

fnorm_disp(flag_verbose,'a_k_p_l2_form',a_k_p_l2_form,'a_k_p_l2_quad',a_k_p_l2_quad,' %%<-- should be <1e-6');

b_k_p_l2_quad = torch.sum(torch.conj(b_k_p_form_) * b_k_p_form_ * weight_3d_k_p_qk_).item();
b_k_p_l2_form = 0 ;
for nsource0 in range(n_source):
    for nsource1 in range(n_source):
        delta_b_c_0_ = delta_b_c_3s__[nsource0,:] ;
        delta_b_c_1_ = delta_b_c_3s__[nsource1,:] ;
        tmp_kd = 2 * pi * k_p_r_max * fnorm(delta_b_c_0_ - delta_b_c_1_) ;
        tmp_h3d = 4 * pi / 3 if np.abs(tmp_kd) <= 1e-12 else h3d(tmp_kd).item() ;
        b_k_p_l2_form += tmp_h3d * k_p_r_max ** 3 ;
    #end;%for;
#end;%for;

fnorm_disp(flag_verbose,'b_k_p_l2_form',b_k_p_l2_form,'b_k_p_l2_quad',b_k_p_l2_quad,' %%<-- should be <1e-6');

r'''
%%%%%%%%;
% Using the k-quadrature on the sphere we can determine the real-space functions a_x_c_form___ and b_x_c_form___ analytically. ;
%%%%%%%%;
a_x_c_form___ = zeros(n_x_c,n_x_c,n_x_c);
for nsource=0:n_source-1;
delta_a_c_ = delta_a_c_3s__(:,1+nsource);
tmp_kd___ = ...
  2*pi*k_p_r_max ...
  *sqrt( ...
       + (delta_a_c_(1+0) + x_c_0___).^2 ...
       + (delta_a_c_(1+1) + x_c_1___).^2 ...
       + (delta_a_c_(1+2) + x_c_2___).^2 ...
       ) ...
  ;
tmp_h3d___ = h3d_(tmp_kd___);
tmp_index_ = efind(abs(tmp_kd___)<=1e-12);
tmp_h3d___(1+tmp_index_) = 4*pi/3;
a_x_c_form___ = a_x_c_form___ + tmp_h3d___*k_p_r_max.^3 ;
end;%for nsource=0:n_source-1;
%%%%%%%%;
b_x_c_form___ = zeros(n_x_c,n_x_c,n_x_c);
for nsource=0:n_source-1;
delta_b_c_ = delta_b_c_3s__(:,1+nsource);
tmp_kd___ = ...
  2*pi*k_p_r_max ...
  *sqrt( ...
       + (delta_b_c_(1+0) + x_c_0___).^2 ...
       + (delta_b_c_(1+1) + x_c_1___).^2 ...
       + (delta_b_c_(1+2) + x_c_2___).^2 ...
       ) ...
  ;
tmp_h3d___ = h3d_(tmp_kd___);
tmp_index_ = efind(abs(tmp_kd___)<=1e-12);
tmp_h3d___(1+tmp_index_) = 4*pi/3;
b_x_c_form___ = b_x_c_form___ + tmp_h3d___*k_p_r_max.^3 ;
end;%for nsource=0:n_source-1;
%%%%%%%%;
'''

# Using the k-quadrature on the sphere we can determine the real-space functions a_x_c_form___ and b_x_c_form___ analytically.
a_x_c_form___ = torch.zeros((n_x_c, n_x_c, n_x_c)).to(dtype=torch.float32);
for nsource in range(n_source):
    delta_a_c_ = delta_a_c_3s__[nsource,:];
    tmp_kd___ = 2*pi*k_p_r_max*torch.sqrt( + (delta_a_c_[0].item() + x_c_0___)**2 + (delta_a_c_[1].item() + x_c_1___)**2 + (delta_a_c_[2].item() + x_c_2___)**2 ) ;
    tmp_h3d___ = h3d(tmp_kd___) ;
    tmp_h3d___[torch.abs(tmp_kd___)<=1e-12] = 4*pi/3 ;
    a_x_c_form___ += tmp_h3d___*k_p_r_max**3 ;
#end;%for nsource in range(n_source);
a_x_c_l2_quad = torch.sum(torch.abs(a_x_c_form___) ** 2 , (0,1,2) ).item() * weight_xxx_c;
print(f" %% Note l2-loss: a_x_c_l2_quad {a_x_c_l2_quad:+0.6f} vs a_k_p_l2_form {a_k_p_l2_form:+0.6f}");
b_x_c_form___ = torch.zeros((n_x_c, n_x_c, n_x_c)).to(dtype=torch.float32);
for nsource in range(n_source):
    delta_b_c_ = delta_b_c_3s__[nsource,:];
    tmp_kd___ = 2*pi*k_p_r_max*torch.sqrt( + (delta_b_c_[0].item() + x_c_0___)**2 + (delta_b_c_[1].item() + x_c_1___)**2 + (delta_b_c_[2].item() + x_c_2___)**2 ) ;
    tmp_h3d___ = h3d(tmp_kd___) ; tmp_h3d___[torch.abs(tmp_kd___)<=1e-12] = 4*pi/3 ;
    b_x_c_form___ += tmp_h3d___*k_p_r_max**3 ;
#end;%for nsource in range(n_source);
b_x_c_l2_quad = torch.sum(torch.abs(b_x_c_form___) ** 2 , (0,1,2) ).item() * weight_xxx_c;
print(f" %% Note l2-loss: b_x_c_l2_quad {b_x_c_l2_quad:+0.6f} vs b_k_p_l2_form {b_k_p_l2_form:+0.6f}");

# Note that, due to the l2-loss,
# we can not expect to reconstruct a_k_p_form_ from a_x_c_l2_quad_ on the limited real-space cartesian-grid.
# (i.e., the high-frequency-components will be lost).

r'''
%%%%%%%%;
eta = pi/k_p_r_max; tmp_t = tic;
a_x_c_quad___ = reshape(xxnufft3d3(n_qk,2*pi*k_c_0_qk_*eta,2*pi*k_c_1_qk_*eta,2*pi*k_c_2_qk_*eta,a_k_p_form_.*weight_3d_k_p_qk_,+1,1e-12,n_xxx_c,x_c_0___(:)/eta,x_c_1___(:)/eta,x_c_2___(:)/eta),[n_x_c,n_x_c,n_x_c]);
tmp_t = toc(tmp_t); disp(sprintf(' %% xxnufft3d3: a_x_c_quad___: time %0.6fs',tmp_t));
fnorm_disp(flag_verbose,'a_x_c_form___',a_x_c_form___,'a_x_c_quad___',a_x_c_quad___);
%%%%;
eta = pi/x_p_r_max; tmp_t = tic;
a_k_p_quad_ = xxnufft3d3(n_xxx_c,x_c_0___(:)*eta,x_c_1___(:)*eta,x_c_2___(:)*eta,a_x_c_form___(:).*weight_xxx_c(:),-1,1e-12,n_qk,2*pi*k_c_0_qk_/eta,2*pi*k_c_1_qk_/eta,2*pi*k_c_2_qk_/eta);
tmp_t = toc(tmp_t); disp(sprintf(' %% xxnufft3d3: a_k_p_quad_: time %0.6fs',tmp_t));
fnorm_disp(flag_verbose,'a_k_p_form_',a_k_p_form_,'a_k_p_quad_',a_k_p_quad_,' %%<-- can be large (bandlimited)');
%%%%%%%%;
eta = pi/k_p_r_max; tmp_t = tic;
b_x_c_quad___ = reshape(xxnufft3d3(n_qk,2*pi*k_c_0_qk_*eta,2*pi*k_c_1_qk_*eta,2*pi*k_c_2_qk_*eta,b_k_p_form_.*weight_3d_k_p_qk_,+1,1e-12,n_xxx_c,x_c_0___(:)/eta,x_c_1___(:)/eta,x_c_2___(:)/eta),[n_x_c,n_x_c,n_x_c]);
tmp_t = toc(tmp_t); disp(sprintf(' %% xxnufft3d3: b_x_c_quad___: time %0.6fs',tmp_t));
fnorm_disp(flag_verbose,'b_x_c_form___',b_x_c_form___,'b_x_c_quad___',b_x_c_quad___);
%%%%;
eta = pi/x_p_r_max; tmp_t = tic;
b_k_p_quad_ = xxnufft3d3(n_xxx_c,x_c_0___(:)*eta,x_c_1___(:)*eta,x_c_2___(:)*eta,b_x_c_form___(:).*weight_xxx_c(:),-1,1e-12,n_qk,2*pi*k_c_0_qk_/eta,2*pi*k_c_1_qk_/eta,2*pi*k_c_2_qk_/eta);
tmp_t = toc(tmp_t); disp(sprintf(' %% xxnufft3d3: b_k_p_quad_: time %0.6fs',tmp_t));
fnorm_disp(flag_verbose,'b_k_p_form_',b_k_p_form_,'b_k_p_quad_',b_k_p_quad_,' %%<-- can be large (bandlimited)');
%%%%%%%%;
'''

eta = pi / k_p_r_max ;
tmp_t = tic();
a_x_c_quad___ = xxnufft3d3(
    n_qk,
    2 * pi * k_c_0_qk_ * eta,
    2 * pi * k_c_1_qk_ * eta,
    2 * pi * k_c_2_qk_ * eta,
    a_k_p_form_ * weight_3d_k_p_qk_,
    +1,
    1e-12,
    n_xxx_c,
    x_c_0___.ravel() / eta,
    x_c_1___.ravel() / eta,
    x_c_2___.ravel() / eta
).reshape((n_x_c, n_x_c, n_x_c)) ;
tmp_t = toc(tmp_t);
if flag_verbose: print(f' %% xxnufft3d3: a_x_c_quad___: {tmp_t:0.6f}s');
fnorm_disp(flag_verbose,'a_x_c_form___',a_x_c_form___,'a_x_c_quad___',a_x_c_quad___,'');

eta = pi / x_p_r_max ;
tmp_t = tic();
a_k_p_quad_ = xxnufft3d3(
    n_xxx_c,
    x_c_0___.ravel() * eta,
    x_c_1___.ravel() * eta,
    x_c_2___.ravel() * eta,
    a_x_c_form___.ravel() * weight_xxx_c,
    -1,
    1e-12,
    n_qk,
    2 * pi * k_c_0_qk_ / eta,
    2 * pi * k_c_1_qk_ / eta,
    2 * pi * k_c_2_qk_ / eta
);
tmp_t = toc(tmp_t);
if flag_verbose: print(f' %% xxnufft3d3: a_k_p_quad_: {tmp_t:0.6f}s');
fnorm_disp(flag_verbose,'a_k_p_form_',a_k_p_form_,'a_k_p_quad_',a_k_p_quad_,' %%<-- can be large (bandlimited)');

eta = pi / k_p_r_max ;
tmp_t = tic();
b_x_c_quad___ = xxnufft3d3(
    n_qk,
    2 * pi * k_c_0_qk_ * eta,
    2 * pi * k_c_1_qk_ * eta,
    2 * pi * k_c_2_qk_ * eta,
    b_k_p_form_ * weight_3d_k_p_qk_,
    +1,
    1e-12,
    n_xxx_c,
    x_c_0___.ravel() / eta,
    x_c_1___.ravel() / eta,
    x_c_2___.ravel() / eta
).reshape((n_x_c, n_x_c, n_x_c));
tmp_t = toc(tmp_t);
if flag_verbose: print(f' %% xxnufft3d3: b_x_c_quad___: {tmp_t:0.6f}s');
fnorm_disp(flag_verbose,'b_x_c_form___',b_x_c_form___,'b_x_c_quad___',b_x_c_quad___,'');

eta = pi / x_p_r_max ;
tmp_t = tic();
b_k_p_quad_ = xxnufft3d3(
    n_xxx_c,
    x_c_0___.ravel() * eta,
    x_c_1___.ravel() * eta,
    x_c_2___.ravel() * eta,
    b_x_c_form___.ravel() * weight_xxx_c,
    -1,
    1e-12,
    n_qk,
    2 * pi * k_c_0_qk_ / eta,
    2 * pi * k_c_1_qk_ / eta,
    2 * pi * k_c_2_qk_ / eta
);
tmp_t = toc(tmp_t);
if flag_verbose: print(f' %% xxnufft3d3: b_k_p_quad_: {tmp_t:0.6f}s');
fnorm_disp(flag_verbose,'b_k_p_form_',b_k_p_form_,'b_k_p_quad_',b_k_p_quad_,' %%<-- can be large (bandlimited)');

r'''
%%%%%%%%;
% Now set up and test polar-quadrature-weights on disk. ;
%%%%%%%%;
l_max_upb = round(2*pi*k_p_r_max);
l_max_max = min(l_max_upb,1+ceil(2*pi*k_p_r_(end)));
n_w_max = n_w_int*2*(l_max_max+1); n_w_0in_ = n_w_max*ones(n_k_p_r,1);
[ ...
 n_w_ ...
,weight_2d_k_p_r_ ...
,weight_2d_wk_ ...
,k_p_r_wk_ ...
,k_p_w_wk_ ...
,k_c_0_wk_ ...
,k_c_1_wk_ ...
] = ...
get_weight_2d_2( ...
 0*flag_verbose ...
,n_k_p_r ...
,k_p_r_ ...
,k_p_r_max ...
,-1 ...
,n_w_0in_ ...
,weight_3d_k_p_r_ ...
);
n_w_sum = sum(n_w_); n_w_csum_ = cumsum([0;n_w_]);
n_gamma_z = n_w_max;
gamma_z_ = transpose(linspace(0,2*pi,n_gamma_z+1)); gamma_z_ = gamma_z_(1:n_gamma_z);
%%%%%%%%;
'''

# Set up and test polar-quadrature-weights on disk.
l_max_upb = matlab_scalar_round(2 * pi * k_p_r_max) ;
l_max_max = min(l_max_upb, 1 + int(np.ceil(2 * pi * k_p_r_[-1].item()))) ;
n_w_max = int(n_w_int * 2 * (l_max_max + 1)) ;
n_w_0in_ = torch.full( [n_k_p_r] , n_w_max ).to(torch.int32);
(
    n_w_,
    weight_2d_k_p_r_,
    weight_2d_wk_,
    k_p_r_wk_,
    k_p_w_wk_,
    k_c_0_wk_,
    k_c_1_wk_,
) = get_weight_2d_2(
    0 * flag_verbose,
    n_k_p_r,
    k_p_r_,
    k_p_r_max,
    -1,
    n_w_0in_,
    weight_3d_k_p_r_,
);

n_w_sum = int(torch.sum(n_w_).item()) ;
n_w_csum_ = cumsum_0(n_w_);
n_gamma_z = n_w_max ;
gamma_z_ = torch.linspace(0, 2 * pi, n_gamma_z + 1).to(torch.float32)[:n_gamma_z] ;

r'''
%%%%%%%%;
tmp_S_delta_x_c_ = 0.85*[cos(pi/4);sin(pi/4)]/max(1e-12,k_p_r_max);
tmp_S_phi = +1*pi/5;
tmp_T_delta_x_c_ = 1.35*[cos(-pi/3);sin(-pi/3)]/max(1e-12,k_p_r_max);
tmp_T_phi = -3*pi/7;
tmp_S_k_p_wk_ = exp(+2*pi*i*(k_c_0_wk_*tmp_S_delta_x_c_(1+0) + k_c_1_wk_*tmp_S_delta_x_c_(1+1)));
tmp_plane_S_k_p_wk_ = 2*k_p_r_wk_.*cos(k_p_w_wk_ - tmp_S_phi);
tmp_T_k_p_wk_ = exp(+2*pi*i*(k_c_0_wk_*tmp_T_delta_x_c_(1+0) + k_c_1_wk_*tmp_T_delta_x_c_(1+1)));
tmp_plane_T_k_p_wk_ = 2*k_p_r_wk_.*cos(k_p_w_wk_ - tmp_T_phi);
tmp_S_x_c_xx_ = real( interp_k_p_to_x_c_xxnufft(n_x_c,diameter_x_c,n_x_c,diameter_x_c,n_k_p_r,k_p_r_,n_w_,tmp_S_k_p_wk_.*weight_2d_wk_*(2*pi)^2)*sqrt(n_x_c*n_x_c) * n_w_sum );
tmp_T_x_c_xx_ = real( interp_k_p_to_x_c_xxnufft(n_x_c,diameter_x_c,n_x_c,diameter_x_c,n_k_p_r,k_p_r_,n_w_,tmp_T_k_p_wk_.*weight_2d_wk_*(2*pi)^2)*sqrt(n_x_c*n_x_c) * n_w_sum );
'''

tmp_S_delta_x_c_ = 0.85 * torch.tensor([np.cos(pi / 4), np.sin(pi / 4)]).to(torch.float32) / max(1e-12, k_p_r_max) ;
tmp_S_phi = +1 * pi / 5 ;
tmp_T_delta_x_c_ = 1.35 * torch.tensor([np.cos(-pi / 3), np.sin(-pi / 3)]).to(torch.float32) / max(1e-12, k_p_r_max) ;
tmp_T_phi = -3 * pi / 7 ;

tmp_S_k_p_wk_ = torch.exp(
    2 * pi * i * (
        k_c_0_wk_ * tmp_S_delta_x_c_[0] +
        k_c_1_wk_ * tmp_S_delta_x_c_[1]
    )
) ;
tmp_plane_S_k_p_wk_ = 2 * k_p_r_wk_ * torch.cos(k_p_w_wk_ - tmp_S_phi) ;

tmp_T_k_p_wk_ = torch.exp(
    2 * pi * i * (
        k_c_0_wk_ * tmp_T_delta_x_c_[0] +
        k_c_1_wk_ * tmp_T_delta_x_c_[1]
    )
) ;
tmp_plane_T_k_p_wk_ = 2 * k_p_r_wk_ * torch.cos(k_p_w_wk_ - tmp_T_phi) ;

tmp_S_x_c_xx_ = torch.real(
    interp_k_p_to_x_c_xxnufft(
        n_x_c, diameter_x_c,
        n_x_c, diameter_x_c,
        n_k_p_r, k_p_r_,
        n_w_, tmp_S_k_p_wk_ * weight_2d_wk_ * (2 * pi) ** 2
    ) * np.sqrt(n_x_c * n_x_c) * n_w_sum
) ;
tmp_T_x_c_xx_ = torch.real(
    interp_k_p_to_x_c_xxnufft(
        n_x_c, diameter_x_c,
        n_x_c, diameter_x_c,
        n_k_p_r, k_p_r_,
        n_w_, tmp_T_k_p_wk_ * weight_2d_wk_ * (2 * pi) ** 2
    ) * np.sqrt(n_x_c * n_x_c) * n_w_sum
) ;

r'''
I_quad = sum(conj(tmp_plane_S_k_p_wk_.*tmp_S_k_p_wk_).*(tmp_plane_T_k_p_wk_.*tmp_T_k_p_wk_).*weight_2d_wk_)*(2*pi)^2;
I_form = I_xPPx_0(k_p_r_max,tmp_S_phi,tmp_S_delta_x_c_,tmp_T_phi,tmp_T_delta_x_c_);
fnorm_disp(flag_verbose,'I_form',I_form,'I_quad',I_quad,' %%<-- should be <1e-6');
'''

I_quad = torch.sum(
    torch.conj(tmp_plane_S_k_p_wk_ * tmp_S_k_p_wk_) *
    (tmp_plane_T_k_p_wk_ * tmp_T_k_p_wk_) *
    weight_2d_wk_
 ).item() * (2 * pi) ** 2 ;

r'''
tmp_S_delta_x_c = frobnorm(tmp_S_delta_x_c_); tmp_S_omega = atan2(tmp_S_delta_x_c_(1+1),tmp_S_delta_x_c_(1+0));
tmp_T_delta_x_c = frobnorm(tmp_T_delta_x_c_); tmp_T_omega = atan2(tmp_T_delta_x_c_(1+1),tmp_T_delta_x_c_(1+0));
I_form = I_PxxP(k_p_r_max, tmp_T_phi, tmp_T_delta_x_c, tmp_T_omega, tmp_S_phi, tmp_S_delta_x_c, tmp_S_omega);
'''

tmp_S_delta_x_c = fnorm(tmp_S_delta_x_c_) ;
tmp_S_omega = np.arctan2(tmp_S_delta_x_c_[1].item(), tmp_S_delta_x_c_[0].item()) ;
tmp_T_delta_x_c = fnorm(tmp_T_delta_x_c_) ;
tmp_T_omega = np.arctan2(tmp_T_delta_x_c_[1].item(), tmp_T_delta_x_c_[0].item()) ;
I_form = I_PxxP(
    k_p_r_max,
    tmp_T_phi,
    tmp_T_delta_x_c,
    tmp_T_omega,
    tmp_S_phi,
    tmp_S_delta_x_c,
    tmp_S_omega
) ;

fnorm_disp(flag_verbose, 'I_form', I_form, 'I_quad', I_quad, ' %%<-- should be <1e-6') ;

r'''
%%%%%%%%;
% Now set up spherical-harmonics. ;
%%%%%%%%;
l_max_ = zeros(n_k_p_r,1);
for nk_p_r=0:n_k_p_r-1;
l_max_(1+nk_p_r) = max(0,min(l_max_upb,1+ceil(2*pi*k_p_r_(1+nk_p_r))));
end;%for nk_p_r=0:n_k_p_r-1;
n_y_ = (l_max_+1).^2;
n_y_max = max(n_y_);
n_y_sum = sum(n_y_);
n_y_csum_ = cumsum([0;n_y_]);
l_max_max = max(l_max_); dWtdkd__l_max_max = 2*l_max_max;
m_max_ = -l_max_max : +l_max_max;
n_m_max = length(m_max_);
Y_l_val_ = zeros(n_y_sum,1);
Y_m_val_ = zeros(n_y_sum,1);
Y_k_val_ = zeros(n_y_sum,1);
for nk_p_r=0:n_k_p_r-1;
l_max = l_max_(1+nk_p_r);
tmp_l_val_ = zeros(n_y_(1+nk_p_r),1);
tmp_m_val_ = zeros(n_y_(1+nk_p_r),1);
na=0; 
for l_val=0:l_max;
for m_val=-l_val:+l_val;
tmp_l_val_(1+na) = l_val;
tmp_m_val_(1+na) = m_val;
na=na+1;
end;%for m_val=-l_val:+l_val;
end;%for l_val=0:l_max;
tmp_index_ = n_y_csum_(1+nk_p_r) + (0:n_y_(1+nk_p_r)-1);
Y_l_val_(1+tmp_index_) = tmp_l_val_;
Y_m_val_(1+tmp_index_) = tmp_m_val_;
Y_k_val_(1+tmp_index_) = k_p_r_(1+nk_p_r);
end;%for nk_p_r=0:n_k_p_r-1;
weight_Y_ = zeros(n_y_sum,1);
for nk_p_r=0:n_k_p_r-1;
tmp_index_ = n_y_csum_(1+nk_p_r) + (0:n_y_(1+nk_p_r)-1);
weight_Y_(1+tmp_index_) = weight_3d_k_p_r_(1+nk_p_r);
end;%for nk_p_r=0:n_k_p_r-1;
'''

l_max_ = torch.zeros(n_k_p_r).to(torch.int32);
for nk_p_r in range(n_k_p_r):
    l_max_[nk_p_r] = max(0,min(l_max_upb,1+np.ceil(2*pi*k_p_r_[nk_p_r].item())));
#end;%for nk_p_r=0:n_k_p_r-1;
n_y_ = (l_max_+1)**2;
n_y_max = int(torch.max(n_y_).item());
n_y_sum = int(torch.sum(n_y_).item());
n_y_csum_ = cumsum_0(n_y_);
l_max_max = int(torch.max(l_max_).item()); dWtdkd__l_max_max = 2*l_max_max;
m_max_ = torch.arange(-l_max_max,+l_max_max+1).to(torch.int32);
n_m_max = numel(m_max_);
Y_l_val_ = torch.zeros(n_y_sum).to(torch.int32);
Y_m_val_ = torch.zeros(n_y_sum).to(torch.int32);
Y_k_val_ = torch.zeros(n_y_sum).to(torch.float32);
for nk_p_r in range(n_k_p_r):
    l_max = int(l_max_[nk_p_r].item());
    tmp_l_val_ = torch.zeros(int(n_y_[nk_p_r].item())).to(torch.int32);
    tmp_m_val_ = torch.zeros(int(n_y_[nk_p_r].item())).to(torch.int32);
    na=0; 
    for l_val in range(l_max+1):
        for m_val in range(-l_val,+l_val+1):
            tmp_l_val_[na] = l_val;
            tmp_m_val_[na] = m_val;
            na=na+1;
        #end;%for m_val=-l_val:+l_val;
    #end;%for l_val=0:l_max;
    tmp_i8_index_lhs_ = int(n_y_csum_[nk_p_r].item()) + torch.arange(int(n_y_[nk_p_r].item())).to(torch.int32);
    Y_l_val_[tmp_i8_index_lhs_] = tmp_l_val_;
    Y_m_val_[tmp_i8_index_lhs_] = tmp_m_val_;
    Y_k_val_[tmp_i8_index_lhs_] = k_p_r_[nk_p_r].item();
#end;%for nk_p_r=0:n_k_p_r-1;
weight_Y_ = torch.zeros(n_y_sum).to(torch.float32);
for nk_p_r in range(n_k_p_r):
    tmp_i8_index_lhs_ = int(n_y_csum_[nk_p_r].item()) + torch.arange(int(n_y_[nk_p_r].item())).to(torch.int32);
    weight_Y_[tmp_i8_index_lhs_] = weight_3d_k_p_r_[nk_p_r].item();
#end;%for nk_p_r=0:n_k_p_r-1;

r'''
%%%%%%%%;
a_k_Y_form_ = zeros(n_y_sum,1);
for nsource=0:n_source-1;
a_k_Y_form_ = a_k_Y_form_ + plane_wave_expansion_1(n_k_p_r,k_p_r_,delta_a_c_3s__(:,1+nsource),l_max_);
end;%for nsource=0:n_source-1;
tmp_t = tic;
if ~exist('Ylm_uklma___','var'); Ylm_uklma___ = []; end;
if ~exist('k_p_azimu_b_sub_uka__','var'); k_p_azimu_b_sub_uka__ = []; end;
if ~exist('k_p_polar_a_sub_uka__','var'); k_p_polar_a_sub_uka__ = []; end;
if ~exist('l_max_uk_','var'); l_max_uk_ = []; end;
if ~exist('index_nu_n_k_per_shell_from_nk_p_r_','var'); index_nu_n_k_per_shell_from_nk_p_r_ = []; end;
if ~exist('index_k_per_shell_uka__','var'); index_k_per_shell_uka__ = []; end;
[ ...
 a_k_Y_quad_ ...
,Ylm_uklma___ ...
,k_p_azimu_b_sub_uka__ ...
,k_p_polar_a_sub_uka__ ...
,l_max_uk_ ...
,index_nu_n_k_per_shell_from_nk_p_r_ ...
,index_k_per_shell_uka__ ...
] = ...
convert_k_p_to_spharm_4( ...
 0*flag_verbose ...
,n_qk ...
,n_qk_csum_ ...
,k_p_r_qk_ ...
,k_p_azimu_b_qk_ ...
,k_p_polar_a_qk_ ...
,weight_3d_k_p_qk_ ...
,weight_shell_qk_ ...
,n_k_p_r ...
,k_p_r_ ...
,weight_3d_k_p_r_ ...
,l_max_ ...
,a_k_p_form_ ...
,Ylm_uklma___ ...
,k_p_azimu_b_sub_uka__ ...
,k_p_polar_a_sub_uka__ ...
,l_max_uk_ ...
,index_nu_n_k_per_shell_from_nk_p_r_ ...
,index_k_per_shell_uka__ ...
);
tmp_t = toc(tmp_t); disp(sprintf(' %% a_k_Y_quad_ time %0.2fs',tmp_t));
fnorm_disp(flag_verbose,'a_k_Y_form_',a_k_Y_form_,'a_k_Y_quad_',a_k_Y_quad_,' %%<-- should be <1e-2');
'''

a_k_Y_form_ = torch.zeros(n_y_sum).to(dtype=torch.complex64);
for nsource in range(n_source):
    a_k_Y_form_ = a_k_Y_form_ + plane_wave_expansion_1(n_k_p_r,k_p_r_,delta_a_c_3s__[nsource,:],l_max_);
#end;%for nsource=0:n_source-1;
tmp_t = tic();
if 'Ylm_uklma___' not in locals(): Ylm_uklma___ = None ;
if 'k_p_azimu_b_sub_uka__' not in locals(): k_p_azimu_b_sub_uka__ = None ;
if 'k_p_polar_a_sub_uka__' not in locals(): k_p_polar_a_sub_uka__ = None ;
if 'l_max_uk_' not in locals(): l_max_uk_ = None ;
if 'index_nu_n_k_per_shell_from_nk_p_r_' not in locals(): index_nu_n_k_per_shell_from_nk_p_r_ = None ;
if 'index_k_per_shell_uka__' not in locals(): index_k_per_shell_uka__ = None ;
(
    a_k_Y_quad_,
    Ylm_uklma___,
    k_p_azimu_b_sub_uka__,
    k_p_polar_a_sub_uka__,
    l_max_uk_,
    index_nu_n_k_per_shell_from_nk_p_r_,
    index_k_per_shell_uka__,
) = convert_k_p_to_spharm_4(
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
    a_k_p_form_,
    Ylm_uklma___,
    k_p_azimu_b_sub_uka__,
    k_p_polar_a_sub_uka__,
    l_max_uk_,
    index_nu_n_k_per_shell_from_nk_p_r_,
    index_k_per_shell_uka__,
);
tmp_t = toc(tmp_t); print(f' %% a_k_Y_quad_ time {tmp_t:.2f}s');
fnorm_disp(flag_verbose,'a_k_Y_form_',a_k_Y_form_,'a_k_Y_quad_',a_k_Y_quad_,' %%<-- should be <1e-2');

r'''
tmp_t = tic;
[ ...
 a_k_p_quad_ ...
] = ...
convert_spharm_to_k_p_4( ...
 0*flag_verbose ...
,n_qk ...
,n_qk_csum_ ...
,k_p_r_qk_ ...
,k_p_azimu_b_qk_ ...
,k_p_polar_a_qk_ ...
,weight_3d_k_p_qk_ ...
,weight_shell_qk_ ...
,n_k_p_r ...
,k_p_r_ ...
,weight_3d_k_p_r_ ...
,l_max_ ...
,a_k_Y_form_ ...
,Ylm_uklma___ ...
,k_p_azimu_b_sub_uka__ ...
,k_p_polar_a_sub_uka__ ...
,l_max_uk_ ...
,index_nu_n_k_per_shell_from_nk_p_r_ ...
,index_k_per_shell_uka__ ...
);
tmp_t = toc(tmp_t); disp(sprintf(' %% a_k_p_quad_ time %0.2fs',tmp_t));
fnorm_disp(flag_verbose,'a_k_p_form_',a_k_p_form_,'a_k_p_quad_',a_k_p_quad_,' %%<-- should be <1e-2');
'''

tmp_t = tic();
(
 a_k_p_quad_,
) = convert_spharm_to_k_p_4(
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
    a_k_Y_form_,
    Ylm_uklma___,
    k_p_azimu_b_sub_uka__,
    k_p_polar_a_sub_uka__,
    l_max_uk_,
    index_nu_n_k_per_shell_from_nk_p_r_,
    index_k_per_shell_uka__,
)[:1];
tmp_t = toc(tmp_t); print(f' %% a_k_p_quad_ time {tmp_t:.2f}s');
fnorm_disp(flag_verbose,'a_k_p_form_',a_k_p_form_,'a_k_p_quad_',a_k_p_quad_,' %%<-- should be <1e-2');

r'''
tmp_t = tic;
[ ...
 a_k_p_reco_ ...
] = ...
convert_spharm_to_k_p_4( ...
 0*flag_verbose ...
,n_qk ...
,n_qk_csum_ ...
,k_p_r_qk_ ...
,k_p_azimu_b_qk_ ...
,k_p_polar_a_qk_ ...
,weight_3d_k_p_qk_ ...
,weight_shell_qk_ ...
,n_k_p_r ...
,k_p_r_ ...
,weight_3d_k_p_r_ ...
,l_max_ ...
,a_k_Y_quad_ ...
,Ylm_uklma___ ...
,k_p_azimu_b_sub_uka__ ...
,k_p_polar_a_sub_uka__ ...
,l_max_uk_ ...
,index_nu_n_k_per_shell_from_nk_p_r_ ...
,index_k_per_shell_uka__ ...
);
tmp_t = toc(tmp_t); disp(sprintf(' %% a_k_p_reco_ time %0.2fs',tmp_t));
fnorm_disp(flag_verbose,'a_k_p_form_',a_k_p_form_,'a_k_p_reco_',a_k_p_reco_,' %%<-- should be <1e-2');
fnorm_disp(flag_verbose,'a_k_p_quad_',a_k_p_quad_,'a_k_p_reco_',a_k_p_reco_,' %%<-- should be <1e-2');
'''

tmp_t = tic();
(
    a_k_p_reco_,
) = convert_spharm_to_k_p_4(
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
    a_k_Y_quad_,
    Ylm_uklma___,
    k_p_azimu_b_sub_uka__,
    k_p_polar_a_sub_uka__,
    l_max_uk_,
    index_nu_n_k_per_shell_from_nk_p_r_,
    index_k_per_shell_uka__,
)[:1];
tmp_t = toc(tmp_t); print(f' %% a_k_p_reco_ time {tmp_t:.2f}s');
fnorm_disp(flag_verbose,'a_k_p_form_',a_k_p_form_,'a_k_p_reco_',a_k_p_reco_,' %%<-- should be <1e-2');
fnorm_disp(flag_verbose,'a_k_p_quad_',a_k_p_quad_,'a_k_p_reco_',a_k_p_reco_,' %%<-- should be <1e-2');

r'''
tmp_t = tic;
[ ...
 a_k_Y_reco_ ...
] = ...
convert_k_p_to_spharm_4( ...
 0*flag_verbose ...
,n_qk ...
,n_qk_csum_ ...
,k_p_r_qk_ ...
,k_p_azimu_b_qk_ ...
,k_p_polar_a_qk_ ...
,weight_3d_k_p_qk_ ...
,weight_shell_qk_ ...
,n_k_p_r ...
,k_p_r_ ...
,weight_3d_k_p_r_ ...
,l_max_ ...
,a_k_p_quad_ ...
,Ylm_uklma___ ...
,k_p_azimu_b_sub_uka__ ...
,k_p_polar_a_sub_uka__ ...
,l_max_uk_ ...
,index_nu_n_k_per_shell_from_nk_p_r_ ...
,index_k_per_shell_uka__ ...
);
tmp_t = toc(tmp_t); disp(sprintf(' %% a_k_Y_quad_ time %0.2fs',tmp_t));
fnorm_disp(flag_verbose,'a_k_Y_form_',a_k_Y_form_,'a_k_Y_reco_',a_k_Y_reco_,' %%<-- should be <1e-2');
fnorm_disp(flag_verbose,'a_k_Y_quad_',a_k_Y_quad_,'a_k_Y_reco_',a_k_Y_reco_,' %%<-- should be <1e-2');
'''

tmp_t = tic();
(
    a_k_Y_reco_,
) = convert_k_p_to_spharm_4(
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
    a_k_p_quad_,
    Ylm_uklma___,
    k_p_azimu_b_sub_uka__,
    k_p_polar_a_sub_uka__,
    l_max_uk_,
    index_nu_n_k_per_shell_from_nk_p_r_,
    index_k_per_shell_uka__,
)[:1];
tmp_t = toc(tmp_t); print(f' %% a_k_Y_reco_ time {tmp_t:.2f}s');
fnorm_disp(flag_verbose,'a_k_Y_form_',a_k_Y_form_,'a_k_Y_reco_',a_k_Y_reco_,' %%<-- should be <1e-2');
fnorm_disp(flag_verbose,'a_k_Y_quad_',a_k_Y_quad_,'a_k_Y_reco_',a_k_Y_reco_,' %%<-- should be <1e-2');

r'''
a_k_Y_form_yk__ = local_yk__from_yk_(n_k_p_r,l_max_,a_k_Y_form_);
a_k_Y_form_yk___ = zeros(1+l_max_max,n_m_max,n_k_p_r);
for nk_p_r=0:n_k_p_r-1;
tmp_index_ = n_y_csum_(1+nk_p_r) + (0:n_y_(1+nk_p_r)-1);
tmp_a_k_Y_form_lm_ = a_k_Y_form_(1+tmp_index_);
tmp_a_k_Y_form_lm__ = zeros(1+l_max_max,n_m_max);
l_max = l_max_(1+nk_p_r);
na=0;
for l_val=0:l_max;
for m_val=-l_val:+l_val;
assert(na==l_val*(l_val+1)+m_val);
tmp_a_k_Y_form_lm__(1+l_val,1+l_max_max+m_val) = tmp_a_k_Y_form_lm_(1+na);
na=na+1;
end;%for m_val=-l_val:+l_val;
end;%for l_val=0:l_max;
a_k_Y_form_yk___(:,:,1+nk_p_r) = tmp_a_k_Y_form_lm__;
end;%for nk_p_r=0:n_k_p_r-1;
%%%%;
flag_check=1;
if flag_check;
na=0;
for nk_p_r=0:n_k_p_r-1;
l_max=l_max_(1+nk_p_r);
for l_val=0:l_max;
for m_val=-l_val:+l_val;
assert(a_k_Y_form_(1+n_y_csum_(1+nk_p_r)+l_val*(l_val+1)+m_val)==a_k_Y_form_yk__(1+l_val*(l_val+1)+m_val,1+nk_p_r));
assert(a_k_Y_form_(1+n_y_csum_(1+nk_p_r)+l_val*(l_val+1)+m_val)==a_k_Y_form_yk___(1+l_val,1+l_max_max+m_val,1+nk_p_r));
na=na+1;
end;%for m_val=-l_val:+l_val;
end;%for l_val=0:l_max;
end;%for nk_p_r=0:n_k_p_r-1;
assert(na==n_y_sum);
end;%if flag_check;
%%%%%%%%;
a_k_Y_form_lkm___ = permute(a_k_Y_form_yk___,[1,3,2]);
%%%%%%%%;
a_k_Y_l2_quad = sum(conj(a_k_Y_form_yk__).*a_k_Y_form_yk__*reshape(weight_3d_k_p_r_,[n_k_p_r,1]));
disp(sprintf(' %% a_k_Y_l2_quad %+0.6f a_k_p_l2_form %+0.6f',a_k_Y_l2_quad,a_k_p_l2_form));
'''

a_k_Y_form_yk__ = local_yk__from_yk_(n_k_p_r,l_max_,a_k_Y_form_);
a_k_Y_form_yk___ = torch.zeros(mtr((1+l_max_max,n_m_max,n_k_p_r))).to(dtype=torch.complex64);
for nk_p_r in range(n_k_p_r):
    tmp_index_ = int(n_y_csum_[nk_p_r].item()) + torch.arange(int(n_y_[nk_p_r].item())).to(torch.int32) ;
    tmp_a_k_Y_form_lm_ = a_k_Y_form_[tmp_index_];
    tmp_a_k_Y_form_lm__ = torch.zeros(mtr((1+l_max_max,n_m_max))).to(dtype=torch.complex64);
    l_max = int(l_max_[nk_p_r].item());
    na=0;
    for l_val in range(l_max+1):
        for m_val in range(-l_val,+l_val+1):
            assert(na==l_val*(l_val+1)+m_val);
            tmp_i8_index_lhs_ = matlab_index_2d_0(1+l_max_max,l_val,n_m_max,l_max_max+m_val);
            tmp_a_k_Y_form_lm__.ravel()[tmp_i8_index_lhs_] = tmp_a_k_Y_form_lm_[na].item();
            na=na+1;
        #end;%for m_val=-l_val:+l_val;
    #end;%for l_val=0:l_max;
    tmp_i8_index_lhs_ = matlab_index_3d_0(1+l_max_max,':',n_m_max,':',n_k_p_r,nk_p_r);
    a_k_Y_form_yk___.ravel()[tmp_i8_index_lhs_] = tmp_a_k_Y_form_lm__.ravel();
#end;%for nk_p_r=0:n_k_p_r-1;
#%%%%;
flag_check=1;
if flag_check:
    na=0;
    for nk_p_r in range(n_k_p_r):
        l_max=int(l_max_[nk_p_r].item());
        for l_val in range(l_max+1):
            for m_val in range(-l_val,+l_val+1):
                tmp_i8_index_rhs_ = matlab_index_2d_0(n_y_max,l_val*(l_val+1)+m_val,n_k_p_r,nk_p_r);
                tmp_i8_index_lhs_ = torch.tensor([int(n_y_csum_[nk_p_r].item()) + l_val*(l_val+1) + m_val ]).to(torch.int32);
                tmp_lhs = a_k_Y_form_[tmp_i8_index_lhs_].item();
                tmp_rhs = a_k_Y_form_yk__.ravel()[tmp_i8_index_rhs_].item();
                if (flag_verbose>2):
                    print(f' %% nk_p_r {nk_p_r} l_val {l_val} m_val {m_val}');
                    print(f' %% %% tmp_rhs {tmp_rhs} tmp_lhs {tmp_lhs}');
                #end;%if (flag_verbose>2):
                assert(np.abs(tmp_lhs-tmp_rhs)<1e-3);
                tmp_i8_index_rhs_ = matlab_index_3d_0(1+l_max_max,l_val,n_m_max,l_max_max+m_val,n_k_p_r,nk_p_r);
                tmp_rhs = a_k_Y_form_yk___.ravel()[tmp_i8_index_rhs_].item();
                if (flag_verbose>2):
                    print(f' %% %% tmp_rhs {tmp_rhs} tmp_lhs {tmp_lhs}');
                #end;%if (flag_verbose>2):
                assert(np.abs(tmp_lhs-tmp_rhs)<1e-3);
                na=na+1;
            #end;%for m_val=-l_val:+l_val;
        #end;%for l_val=0:l_max;
    #end;%for nk_p_r=0:n_k_p_r-1;
    assert(na==n_y_sum);
#end;%if flag_check;
#%%%%%%%%;
a_k_Y_form_lkm___ = torch.permute(a_k_Y_form_yk___,mtr(mts((0,2,1))));
#%%%%%%%%;
a_k_Y_l2_quad = torch.sum( mmvm( (torch.conj(a_k_Y_form_yk__)*a_k_Y_form_yk__).to(dtype=torch.complex64) , weight_3d_k_p_r_.to(dtype=torch.complex64) ) ).item();
print(f' %% a_k_Y_l2_quad {np.abs(a_k_Y_l2_quad):+0.6f} a_k_p_l2_form {np.abs(a_k_p_l2_form):+0.6f}') ;

r'''
%%%%%%%%;
% test convert_k_p_to_spharm_uniform_over_n_k_p_r_5. ;
%%%%%%%%;
tmp_t = tic;
if ~exist('Ylm_uklma___','var'); Ylm_uklma___ = []; end;
if ~exist('k_p_azimu_b_sub_uka__','var'); k_p_azimu_b_sub_uka__ = []; end;
if ~exist('k_p_polar_a_sub_uka__','var'); k_p_polar_a_sub_uka__ = []; end;
if ~exist('l_max_uk_','var'); l_max_uk_ = []; end;
if ~exist('index_nu_n_k_per_shell_from_nk_p_r_','var'); index_nu_n_k_per_shell_from_nk_p_r_ = []; end;
if ~exist('index_k_per_shell_uka__','var'); index_k_per_shell_uka__ = []; end;
[ ...
 a_k_Y_quad_4_ ...
,Ylm_uklma___ ...
,k_p_azimu_b_sub_uka__ ...
,k_p_polar_a_sub_uka__ ...
,l_max_uk_ ...
,index_nu_n_k_per_shell_from_nk_p_r_ ...
,index_k_per_shell_uka__ ...
] = ...
convert_k_p_to_spharm_4( ...
 0*flag_verbose ...
,n_qk ...
,n_qk_csum_ ...
,k_p_r_qk_ ...
,k_p_azimu_b_qk_ ...
,k_p_polar_a_qk_ ...
,weight_3d_k_p_qk_ ...
,weight_shell_qk_ ...
,n_k_p_r ...
,k_p_r_ ...
,weight_3d_k_p_r_ ...
,l_max_ ...
,a_k_p_form_ ...
,Ylm_uklma___ ...
,k_p_azimu_b_sub_uka__ ...
,k_p_polar_a_sub_uka__ ...
,l_max_uk_ ...
,index_nu_n_k_per_shell_from_nk_p_r_ ...
,index_k_per_shell_uka__ ...
);
tmp_t = toc(tmp_t); disp(sprintf(' %% a_k_Y_quad_4_ time %0.2fs',tmp_t));
%%%%%%%%;
tmp_t = tic;
if ~exist('sqrt_2lp1_','var'); sqrt_2lp1_ = []; end;
if ~exist('sqrt_2mp1_','var'); sqrt_2mp1_ = []; end;
if ~exist('sqrt_rat0_m_','var'); sqrt_rat0_m_ = []; end;
if ~exist('sqrt_rat3_lm__','var'); sqrt_rat3_lm__ = []; end;
if ~exist('sqrt_rat4_lm__','var'); sqrt_rat4_lm__ = []; end;
[ ...
 a_k_Y_quad_5_ ...
,sqrt_2lp1_ ...
,sqrt_2mp1_ ...
,sqrt_rat0_m_ ...
,sqrt_rat3_lm__ ...
,sqrt_rat4_lm__ ...
] = ...
convert_k_p_to_spharm_uniform_over_n_k_p_r_5( ...
 0*flag_verbose ...
,n_qk ...
,n_qk_csum_ ...
,k_p_r_qk_ ...
,k_p_azimu_b_qk_ ...
,k_p_polar_a_qk_ ...
,weight_3d_k_p_qk_ ...
,weight_shell_qk_ ...
,n_k_p_r ...
,k_p_r_ ...
,weight_3d_k_p_r_ ...
,l_max_ ...
,a_k_p_form_ ...
,sqrt_2lp1_ ...
,sqrt_2mp1_ ...
,sqrt_rat0_m_ ...
,sqrt_rat3_lm__ ...
,sqrt_rat4_lm__ ...
);
tmp_t = toc(tmp_t); disp(sprintf(' %% a_k_Y_quad_5_ time %0.2fs',tmp_t));
%%%%%%%%;
fnorm_disp(flag_verbose,'a_k_Y_quad_4_',a_k_Y_quad_4_,'a_k_Y_quad_5_',a_k_Y_quad_5_,' %%<-- should be zero');
%%%%%%%%;
'''

tmp_t = tic();
if 'Ylm_uklma___' not in locals(): Ylm_uklma___ = None ; #end;
if 'k_p_azimu_b_sub_uka__' not in locals(): k_p_azimu_b_sub_uka__ = None ; #end;
if 'k_p_polar_a_sub_uka__' not in locals(): k_p_polar_a_sub_uka__ = None ; #end;
if 'l_max_uk_' not in locals(): l_max_uk_ = None ; #end;
if 'index_nu_n_k_per_shell_from_nk_p_r_' not in locals(): index_nu_n_k_per_shell_from_nk_p_r_ = None ; #end;
if 'index_k_per_shell_uka__' not in locals(): index_k_per_shell_uka__ = None ; #end;
(
    a_k_Y_quad_4_,
    Ylm_uklma___,
    k_p_azimu_b_sub_uka__,
    k_p_polar_a_sub_uka__,
    l_max_uk_,
    index_nu_n_k_per_shell_from_nk_p_r_,
    index_k_per_shell_uka__,
) = convert_k_p_to_spharm_4(
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
    a_k_p_form_,
    Ylm_uklma___,
    k_p_azimu_b_sub_uka__,
    k_p_polar_a_sub_uka__,
    l_max_uk_,
    index_nu_n_k_per_shell_from_nk_p_r_,
    index_k_per_shell_uka__,
);
tmp_t = toc(tmp_t); disp(sprintf(' %% a_k_Y_quad_4_ time %0.2fs',tmp_t));
#%%%%%%%%;
tmp_t = tic();
if 'sqrt_2lp1_' not in locals(): sqrt_2lp1_ = None; #end;
if 'sqrt_2mp1_' not in locals(): sqrt_2mp1_ = None; #end;
if 'sqrt_rat0_m_' not in locals(): sqrt_rat0_m_ = None; #end;
if 'sqrt_rat3_lm__' not in locals(): sqrt_rat3_lm__ = None; #end;
if 'sqrt_rat4_lm__' not in locals(): sqrt_rat4_lm__ = None; #end;
(
    a_k_Y_quad_5_,
    sqrt_2lp1_,
    sqrt_2mp1_,
    sqrt_rat0_m_,
    sqrt_rat3_lm__,
    sqrt_rat4_lm__,
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
    a_k_p_form_,
    sqrt_2lp1_,
    sqrt_2mp1_,
    sqrt_rat0_m_,
    sqrt_rat3_lm__,
    sqrt_rat4_lm__,
);
tmp_t = toc(tmp_t); disp(sprintf(' %% a_k_Y_quad_5_ time %0.2fs',tmp_t));
#%%%%%%%%;
fnorm_disp(flag_verbose,'a_k_Y_quad_4_',a_k_Y_quad_4_,'a_k_Y_quad_5_',a_k_Y_quad_5_,' %%<-- should be zero');
#%%%%%%%%;

tmp_t = tic();
if 'Ylm_uklma___' not in locals(): Ylm_uklma___ = None ; #end;
if 'k_p_azimu_b_sub_uka__' not in locals(): k_p_azimu_b_sub_uka__ = None ; #end;
if 'k_p_polar_a_sub_uka__' not in locals(): k_p_polar_a_sub_uka__ = None ; #end;
if 'l_max_uk_' not in locals(): l_max_uk_ = None ; #end;
if 'index_nu_n_k_per_shell_from_nk_p_r_' not in locals(): index_nu_n_k_per_shell_from_nk_p_r_ = None ; #end;
if 'index_k_per_shell_uka__' not in locals(): index_k_per_shell_uka__ = None ; #end;
(
    a_k_p_quad_4_,
    Ylm_uklma___,
    k_p_azimu_b_sub_uka__,
    k_p_polar_a_sub_uka__,
    l_max_uk_,
    index_nu_n_k_per_shell_from_nk_p_r_,
    index_k_per_shell_uka__,
) = convert_spharm_to_k_p_4(
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
    a_k_Y_form_,
    Ylm_uklma___,
    k_p_azimu_b_sub_uka__,
    k_p_polar_a_sub_uka__,
    l_max_uk_,
    index_nu_n_k_per_shell_from_nk_p_r_,
    index_k_per_shell_uka__,
);
tmp_t = toc(tmp_t); disp(sprintf(' %% a_k_p_quad_4_ time %0.2fs',tmp_t));
#%%%%%%%%;
tmp_t = tic();
if 'sqrt_2lp1_' not in locals(): sqrt_2lp1_ = None; #end;
if 'sqrt_2mp1_' not in locals(): sqrt_2mp1_ = None; #end;
if 'sqrt_rat0_m_' not in locals(): sqrt_rat0_m_ = None; #end;
if 'sqrt_rat3_lm__' not in locals(): sqrt_rat3_lm__ = None; #end;
if 'sqrt_rat4_lm__' not in locals(): sqrt_rat4_lm__ = None; #end;
(
    a_k_p_quad_5_,
    sqrt_2lp1_,
    sqrt_2mp1_,
    sqrt_rat0_m_,
    sqrt_rat3_lm__,
    sqrt_rat4_lm__,
) = convert_spharm_to_k_p_uniform_over_n_k_p_r_5(
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
    a_k_Y_form_,
    sqrt_2lp1_,
    sqrt_2mp1_,
    sqrt_rat0_m_,
    sqrt_rat3_lm__,
    sqrt_rat4_lm__,
);
tmp_t = toc(tmp_t); disp(sprintf(' %% a_k_p_quad_5_ time %0.2fs',tmp_t));
#%%%%%%%%;
fnorm_disp(flag_verbose,'a_k_p_quad_4_',a_k_p_quad_4_,'a_k_p_quad_5_',a_k_p_quad_5_,' %%<-- should be zero');
#%%%%%%%%;

r'''
%%%%%%%%;
% define rotations in 2d and 3d. ;
%%%%%%%%;
R2 = @(gamma_z) ...
[ +cos(gamma_z) -sin(gamma_z) ; ...
  +sin(gamma_z) +cos(gamma_z) ; ...
] ;
%%%%%%%%;
Rz = @(azimu_b) ...
[ +cos(azimu_b) -sin(azimu_b) 0 ; ...
  +sin(azimu_b) +cos(azimu_b) 0 ; ...
   0             0            1 ; ...
] ;
%%%%%%%%;
Ry = @(polar_a) ...
[ +cos(polar_a) 0 +sin(polar_a) ; ...
   0            1  0            ; ...
  -sin(polar_a) 0 +cos(polar_a) ; ...
];
%%%%%%%%;
'''

def R2(gamma_z):
    R2__ = torch.zeros(mtr((n_2,n_2))).to(dtype=torch.float32);
    na=0;
    R2__.ravel()[na] = +np.cos(gamma_z); na=na+1;
    R2__.ravel()[na] = +np.sin(gamma_z); na=na+1;
    R2__.ravel()[na] = -np.sin(gamma_z); na=na+1;
    R2__.ravel()[na] = +np.cos(gamma_z); na=na+1;
    assert(na==4);
    return(R2__);
#end;def;

def Rz(azimu_b):
    Rz__ = torch.zeros(mtr((n_3,n_3))).to(dtype=torch.float32);
    na=0;
    Rz__.ravel()[na] = +np.cos(azimu_b); na=na+1;
    Rz__.ravel()[na] = +np.sin(azimu_b); na=na+1;
    Rz__.ravel()[na] =                0; na=na+1;
    Rz__.ravel()[na] = -np.sin(azimu_b); na=na+1;
    Rz__.ravel()[na] = +np.cos(azimu_b); na=na+1;
    Rz__.ravel()[na] =                0; na=na+1;
    Rz__.ravel()[na] =                0; na=na+1;
    Rz__.ravel()[na] =                0; na=na+1;
    Rz__.ravel()[na] =                1; na=na+1;
    assert(na==9);
    return(Rz__);
#end;def;

def Ry(polar_a):
    Ry__ = torch.zeros(mtr((n_3,n_3))).to(dtype=torch.float32);
    na=0;
    Ry__.ravel()[na] = +np.cos(polar_a); na=na+1;
    Ry__.ravel()[na] =                0; na=na+1;
    Ry__.ravel()[na] = -np.sin(polar_a); na=na+1;
    Ry__.ravel()[na] =                0; na=na+1;
    Ry__.ravel()[na] =                1; na=na+1;
    Ry__.ravel()[na] =                0; na=na+1;
    Ry__.ravel()[na] = +np.sin(polar_a); na=na+1;
    Ry__.ravel()[na] =                0; na=na+1;
    Ry__.ravel()[na] = +np.cos(polar_a); na=na+1;
    assert(na==9);
    return(Ry__);
#end;def;

r'''
%%%%%%%%;
% Now generate templates from the volumes in spherical-harmonic coordinates. ;
%%%%%%%%;
[ ...
 S_k_p_wkS__ ...
,~ ...
,n_viewing_S ...
,viewing_azimu_b_S_ ...
,viewing_polar_a_S_ ...
,viewing_weight_S_ ...
] = ...
pm_template_2( ...
 0*flag_verbose ...
,l_max_max ...
,n_k_p_r ...
,reshape(a_k_Y_form_yk__,[n_y_max,n_k_p_r]) ...
,template_k_eq_d_double/max(1e-12,k_p_r_max) ...
,-1 ...
,n_w_max ...
);
n_S = n_viewing_S;
S_k_p_wkS__ = reshape(S_k_p_wkS__,[n_w_max*n_k_p_r,n_S]);
S_k_p_l2_quad_S_ = reshape(sum(bsxfun(@times,conj(S_k_p_wkS__).*S_k_p_wkS__,reshape(weight_2d_wk_,[n_w_sum,1])),1),[n_S,1])*(2*pi)^2;
%%%%%%%%;
'''

(
    S_k_p_wkS__,
    _,
    n_viewing_S,
    viewing_azimu_b_S_,
    viewing_polar_a_S_,
    viewing_weight_S_,
) = pm_template_2(
    0*flag_verbose,
    l_max_max,
    n_k_p_r,
    torch.reshape(a_k_Y_form_yk__,mtr((n_y_max,n_k_p_r))),
    template_k_eq_d_double/np.maximum(1e-12,k_p_r_max),
    -1,
    n_w_max,
)[:6];
n_S = n_viewing_S;
S_k_p_wkS__ = torch.reshape(S_k_p_wkS__,mtr((n_w_max*n_k_p_r,n_S)));
S_k_p_l2_quad_S_ = mvmm( weight_2d_wk_ , torch.abs(S_k_p_wkS__)**2 ).ravel() * (2*pi)**2 ;

r'''
%%%%%%%%;
% The 3d-frequency-space locations associated with a particular template can be calculated as follows: ;
% (see get_template_1.m);
%%%%%%%%;
nS = max(0,min(n_S-1,floor(0.2*n_S)));
tmp_viewing_azimu_b = viewing_azimu_b_S_(1+nS);
tmp_viewing_polar_a = viewing_polar_a_S_(1+nS);
tmp_viewing_gamma_z = 0.0;
tmp_cc_ = cos(k_p_w_wk_); tmp_sc_ = sin(k_p_w_wk_);
tmp_cb = cos(tmp_viewing_azimu_b); tmp_sb = sin(tmp_viewing_azimu_b);
tmp_ca = cos(tmp_viewing_polar_a); tmp_sa = sin(tmp_viewing_polar_a);
tmp_k_c_0_wk_ = (+tmp_cb*tmp_ca*tmp_cc_ - tmp_sb*tmp_sc_).*k_p_r_wk_;
tmp_k_c_1_wk_ = (+tmp_sb*tmp_ca*tmp_cc_ + tmp_cb*tmp_sc_).*k_p_r_wk_;
tmp_k_c_2_wk_ = (-tmp_sa*tmp_cc_                        ).*k_p_r_wk_;
tmp_0_k_c_wk3__ = cat(2,tmp_k_c_0_wk_,tmp_k_c_1_wk_,tmp_k_c_2_wk_);
%%%%%%%%;
% which can also be calculated via: ;
% (see imagesc_S_k_p_3d_2.m);
%%%%%%%%;
k_c_wk2__ = cat(2,cos(k_p_w_wk_).*k_p_r_wk_,sin(k_p_w_wk_).*k_p_r_wk_);
k_c_wk3__ = cat(2,k_c_wk2__,zeros(n_w_sum,1));
tmp_1_k_c_wk3__ = k_c_wk3__*transpose(Ry(tmp_viewing_polar_a))*transpose(Rz(tmp_viewing_azimu_b));
%%%%%%%%;
fnorm_disp(flag_verbose,'tmp_0_k_c_wk3__',tmp_0_k_c_wk3__,'tmp_1_k_c_wk3__',tmp_1_k_c_wk3__,' %%<-- should be zero');
%%%%%%%%;
'''

nS = int(np.maximum(0,np.minimum(n_S-1,np.floor(0.2*n_S))));
tmp_viewing_azimu_b = viewing_azimu_b_S_[nS].item();
tmp_viewing_polar_a = viewing_polar_a_S_[nS].item();
tmp_viewing_gamma_z = 0.0;
tmp_cc_ = torch.cos(k_p_w_wk_); tmp_sc_ = torch.sin(k_p_w_wk_);
tmp_cb = np.cos(tmp_viewing_azimu_b); tmp_sb = np.sin(tmp_viewing_azimu_b);
tmp_ca = np.cos(tmp_viewing_polar_a); tmp_sa = np.sin(tmp_viewing_polar_a);
tmp_k_c_0_wk_ = (+tmp_cb*tmp_ca*tmp_cc_ - tmp_sb*tmp_sc_)*k_p_r_wk_;
tmp_k_c_1_wk_ = (+tmp_sb*tmp_ca*tmp_cc_ + tmp_cb*tmp_sc_)*k_p_r_wk_;
tmp_k_c_2_wk_ = (-tmp_sa*tmp_cc_                        )*k_p_r_wk_;
tmp_0_k_c_wk3__ = torch.reshape(torch.concatenate((tmp_k_c_0_wk_.ravel(),tmp_k_c_1_wk_.ravel(),tmp_k_c_2_wk_.ravel()),0),mtr((n_w_sum,n_3)));
#%%%%%%%%;
k_c_wk3__ = torch.reshape( torch.concatenate( ( torch.cos(k_p_w_wk_)*k_p_r_wk_ , torch.sin(k_p_w_wk_)*k_p_r_wk_ , torch.zeros(n_w_sum).to(dtype=torch.float32) ) , 0 ) , mtr((n_w_sum,n_3)) );
tmp_1_k_c_wk3__ = mmmm( k_c_wk3__ , mmmm(Ry(-tmp_viewing_polar_a),Rz(-tmp_viewing_azimu_b)) ) ;
#%%%%%%%%;
fnorm_disp(flag_verbose,'tmp_0_k_c_wk3__',tmp_0_k_c_wk3__,'tmp_1_k_c_wk3__',tmp_1_k_c_wk3__,' %%<-- should be zero');
#%%%%%%%%;

r'''
%%%%%%%%;
% Now step through and reconstitute the templates themselves. ;
%%%%%%%%;
T_k_p_l2_quad_S_ = zeros(n_S,1);
T_k_p_l2_form_S_ = zeros(n_S,1);
T_k_p_wkS__ = zeros(n_w_sum,n_S);
for nS=0:n_S-1;
S_k_p_wk_ = S_k_p_wkS__(:,1+nS);
tmp_azimu_b = viewing_azimu_b_S_(1+nS);
tmp_polar_a = viewing_polar_a_S_(1+nS);
tmp_gamma_z = 0.0;
tmp_R__ = Rz(-tmp_gamma_z)*Ry(-tmp_polar_a)*Rz(-tmp_azimu_b);
T_k_p_wk_ = zeros(n_w_sum,1);
for nsource=0:n_source-1;
tmp_delta_ = tmp_R__*delta_a_c_3s__(:,1+nsource);
T_k_p_wk_ = T_k_p_wk_ + exp(+i*2*pi*(k_c_0_wk_*tmp_delta_(1+0) + k_c_1_wk_*tmp_delta_(1+1)));
end;%for nsource=0:n_source-1;
T_k_p_l2_quad_S_(1+nS) = sum(conj(T_k_p_wk_).*T_k_p_wk_.*weight_2d_wk_)*(2*pi)^2;
for nsource0=0:n_source-1;
for nsource1=0:n_source-1;
tmp_delta_a_c_0_ = tmp_R__*delta_a_c_3s__(:,1+nsource0);
tmp_delta_a_c_1_ = tmp_R__*delta_a_c_3s__(:,1+nsource1);
tmp_kd = 2*pi*k_p_r_max*fnorm(tmp_delta_a_c_0_(1:2) - tmp_delta_a_c_1_(1:2));
tmp_h2d = (2*pi)^2; if abs(tmp_kd)>1e-12; tmp_h2d = h2d_(tmp_kd); end;
T_k_p_l2_form_S_(1+nS) = T_k_p_l2_form_S_(1+nS) + tmp_h2d/(2*pi)^2 * (pi*k_p_r_max^2);
end;%for nsource1=0:n_source-1;
end;%for nsource0=0:n_source-1;
if (flag_disp>1);
if mod(nS,128)==0;
figure(1+nf);nf=nf+1;clf;figmed;
subplot(1,2,1);imagesc_p(n_k_p_r,k_p_r_,n_w_,n_w_sum,real(S_k_p_wk_)); axis image; axisnotick; title('real(S_k_p_wk_)','Interpreter','none');
subplot(1,2,2);imagesc_p(n_k_p_r,k_p_r_,n_w_,n_w_sum,real(T_k_p_wk_)); axis image; axisnotick; title('real(T_k_p_wk_)','Interpreter','none');
sgtitle(sprintf('nS %d/%d',nS,n_S));
end;%if mod(nS,128)==0;
end;%if (flag_disp>1);
T_k_p_wkS__(:,1+nS) = T_k_p_wk_;
end;%for nS=0:n_S-1;
fnorm_disp(flag_verbose,'T_k_p_wkS__',T_k_p_wkS__,'S_k_p_wkS__',S_k_p_wkS__,' %%<-- should be <1e-2');
fnorm_disp(flag_verbose,'T_k_p_l2_form_S_',T_k_p_l2_form_S_,'T_k_p_l2_quad_S_',T_k_p_l2_quad_S_,' %%<-- should be <1e-2');
fnorm_disp(flag_verbose,'T_k_p_l2_form_S_',T_k_p_l2_form_S_,'S_k_p_l2_quad_S_',S_k_p_l2_quad_S_,' %%<-- should be <1e-2');
%%%%%%%%;
'''

T_k_p_l2_quad_S_ = torch.zeros(n_S).to(dtype=torch.complex64);
T_k_p_l2_form_S_ = torch.zeros(n_S).to(dtype=torch.complex64);
T_k_p_wkS__ = torch.zeros(mtr((n_w_sum,n_S))).to(dtype=torch.complex64);
for nS in range(n_S):
    tmp_i8_index_rhs_ = matlab_index_2d_0(n_w_sum,':',n_S,nS);
    S_k_p_wk_ = S_k_p_wkS__.ravel()[tmp_i8_index_rhs_];
    tmp_azimu_b = viewing_azimu_b_S_[nS].item();
    tmp_polar_a = viewing_polar_a_S_[nS].item();
    tmp_gamma_z = 0.0;
    tmp_R__ = mmmm(Rz(-tmp_gamma_z),mmmm(Ry(-tmp_polar_a),Rz(-tmp_azimu_b)));
    T_k_p_wk_ = torch.zeros(n_w_sum).to(dtype=torch.complex64);
    for nsource in range(n_source):
        tmp_delta_ = mmvm(tmp_R__,delta_a_c_3s__[nsource,:].ravel());
        T_k_p_wk_ = T_k_p_wk_ + torch.exp(+i*2*pi*(k_c_0_wk_*tmp_delta_[0].item() + k_c_1_wk_*tmp_delta_[1].item()));
    #end;%for nsource=0:n_source-1;
    T_k_p_l2_quad_S_[nS] = torch.sum(torch.conj(T_k_p_wk_)*T_k_p_wk_*weight_2d_wk_).item()*(2*pi)**2;
    for nsource0 in range(n_source):
        for nsource1 in range(n_source):
            tmp_delta_a_c_0_ = mmvm( tmp_R__ , delta_a_c_3s__[nsource0,:].ravel() );
            tmp_delta_a_c_1_ = mmvm( tmp_R__ , delta_a_c_3s__[nsource1,:].ravel() );
            tmp_kd = 2*pi*k_p_r_max*fnorm(tmp_delta_a_c_0_[0:2] - tmp_delta_a_c_1_[0:2]);
            tmp_h2d = (2*pi)**2;
            if np.abs(tmp_kd)>1e-12: tmp_h2d = h2d(tmp_kd).item();
            T_k_p_l2_form_S_[nS] = T_k_p_l2_form_S_[nS].item() + tmp_h2d/(2*pi)**2 * (pi*k_p_r_max**2);
        #end;%for nsource1=0:n_source-1;
    #end;%for nsource0=0:n_source-1;
    tmp_i8_index_lhs_ = matlab_index_2d_0(n_w_sum,':',n_S,nS);
    T_k_p_wkS__.ravel()[tmp_i8_index_lhs_] = T_k_p_wk_;
#end;%for nS=0:n_S-1;
fnorm_disp(flag_verbose,'T_k_p_wkS__',T_k_p_wkS__,'S_k_p_wkS__',S_k_p_wkS__,' %%<-- should be <1e-2');
fnorm_disp(flag_verbose,'T_k_p_l2_form_S_',T_k_p_l2_form_S_,'T_k_p_l2_quad_S_',T_k_p_l2_quad_S_,' %%<-- should be <1e-2');
fnorm_disp(flag_verbose,'T_k_p_l2_form_S_',T_k_p_l2_form_S_,'S_k_p_l2_quad_S_',S_k_p_l2_quad_S_,' %%<-- should be <1e-2');

#%%%%%%%%;
#% Now skipping to test_transforms_8_dr.py line 1366: ;
#%%%%%%%%;

r'''
%%%%%%%%;
S_k_q_wkS__ = zeros(n_w_sum,n_S);
for nS=0:n_S-1;
S_k_p_wk_ = S_k_p_wkS__(:,1+nS);
S_k_q_wk_ = interp_p_to_q(n_k_p_r,n_w_,n_w_sum,S_k_p_wk_);
S_k_q_wkS__(:,1+nS) = S_k_q_wk_;
end;%for nS=0:n_S-1;
tmp_ = reshape(interp_p_to_q(n_k_p_r,n_w_,n_w_sum,S_k_p_wkS__),[n_w_sum,n_S]); %<-- this accomplishes the same thing. ;
fnorm_disp(flag_verbose,'S_k_q_wkS__',S_k_q_wkS__,'tmp_',tmp_,' %%<-- should be zero');
S_k_q_wSk___ = permute(reshape(S_k_q_wkS__,[n_w_max,n_k_p_r,n_S]),1+[0,2,1]);
%%%%%%%%;
'''

S_k_q_wkS__ = torch.reshape(interp_p_to_q(n_k_p_r,n_w_,n_w_sum,S_k_p_wkS__),mtr((n_w_sum,n_S)));
S_k_q_wSk___ = torch.permute(torch.reshape(S_k_q_wkS__,mtr((n_w_max,n_k_p_r,n_S))),mtr(mts((0,2,1))));

r'''
flag_delta = 1;
%%%%%%%%;
if ~flag_delta; delta_r_max = 0.0/max(1e-12,k_p_r_max); svd_eps = 1e-6; n_delta_v_requested =  1; end;
if  flag_delta; delta_r_max = 0.5/max(1e-12,k_p_r_max); svd_eps = 1e-6; n_delta_v_requested = 128; end;
FTK = ampmh_FTK_1(n_k_p_r,k_p_r_,k_p_r_max,delta_r_max,svd_eps,n_delta_v_requested);
%%%%%%%%;
n_delta_v = FTK.n_delta_v;
'''

flag_delta = 1;
#%%%%%%%%;
if flag_delta==0: delta_r_max = 0.0/np.maximum(1e-12,k_p_r_max); svd_eps = 1e-6; n_delta_v_requested =   1;
if flag_delta!=0: delta_r_max = 0.5/np.maximum(1e-12,k_p_r_max); svd_eps = 1e-6; n_delta_v_requested = 128;
FTK = tfh_FTK_2(n_k_p_r,k_p_r_.to(dtype=torch.float64),float(k_p_r_max),float(delta_r_max),float(svd_eps),n_delta_v_requested);
#%%%%%%%%;
n_delta_v = FTK['n_delta_v'];

r'''
flag_CTF = 1;
%%%%%%%%;
% Now we generate images for the likelihood-calculation. ;
% This test accounts for anisotropic CTF. ;
% We also shift each image by an 'on-grid' displacement. ;
%%%%%%%%;
n_CTF = 3;
CTF_phi_C_ = pi*[+2/3;-1/5;+6/7];
CTF_k_p_wkC__ = zeros(n_w_sum,n_CTF);
for nCTF=0:n_CTF-1;
if ~flag_CTF; CTF_k_p_wkC__(:,1+nCTF) = ones(n_w_sum,1); end;
if  flag_CTF; CTF_k_p_wkC__(:,1+nCTF) = 2*k_p_r_wk_.*cos(k_p_w_wk_ - CTF_phi_C_(1+nCTF)); end;
end;%for nCTF=0:n_CTF-1;
%%%%%%%%;
tmp_t = tic();
n_M = 2*n_S; %<-- pick something bigger than n_S;
n_gamma_z = n_w_max;
gamma_z_ = transpose(linspace(0,2*pi,n_gamma_z+1)); gamma_z_ = gamma_z_(1:n_gamma_z);
M_k_p_wkM__ = zeros(n_w_sum,n_M);
index_nCTF_from_nM_ = zeros(n_M,1);
index_nS_from_nM_ = zeros(n_M,1);
euler_azimu_b_true_M_ = zeros(n_M,1);
euler_polar_a_true_M_ = zeros(n_M,1);
index_nw_from_nM_ = zeros(n_M,1);
euler_gamma_z_true_M_ = zeros(n_M,1);
index_nd_from_nM_ = zeros(n_M,1);
image_delta_x_true_M_ = zeros(n_M,1);
image_delta_y_true_M_ = zeros(n_M,1);
rng(0);
for nM=0:n_M-1;
nCTF = mod(nM,n_CTF);
CTF_k_p_wk_ = CTF_k_p_wkC__(:,1+nCTF);
index_nCTF_from_nM_(1+nM) = nCTF;
nS = max(0,min(n_S-1,mod(nM,n_S)));
index_nS_from_nM_(1+nM) = nS;
S_k_p_wk_ = S_k_p_wkS__(:,1+nS);
azimu_b = viewing_azimu_b_S_(1+nS);
polar_a = viewing_polar_a_S_(1+nS);
index_gamma_z = periodize(nM,0,n_gamma_z);
gamma_z = gamma_z_(1+index_gamma_z);
index_nw_from_nM_(1+nM) = index_gamma_z;
euler_azimu_b_true_M_(1+nM) = azimu_b;
euler_polar_a_true_M_(1+nM) = polar_a;
euler_gamma_z_true_M_(1+nM) = gamma_z;
index_nd_from_nM_(1+nM) = periodize(nM,0,n_delta_v);
delta_x = FTK.delta_x_(1+index_nd_from_nM_(1+nM)); delta_y = FTK.delta_y_(1+index_nd_from_nM_(1+nM));
image_delta_x_true_M_(1+nM) = delta_x;
image_delta_y_true_M_(1+nM) = delta_y;
T_k_p_wk_ = rotate_p_to_p_fftw(n_k_p_r,n_w_,n_w_sum,S_k_p_wk_,+gamma_z); %<-- note here we rotate S_k_p_wk_ by +gamma_z to form M_k_p_wk_. ;
T_k_p_wk_ = transf_p_to_p(n_k_p_r,k_p_r_,n_w_,n_w_sum,T_k_p_wk_,-delta_x,-delta_y); %<-- note here we translate T_k_p_wk_ by -[delta_x,delta_y]. ;
M_k_p_wk_ = CTF_k_p_wk_.*T_k_p_wk_;
M_k_p_wkM__(:,1+nM) = M_k_p_wk_;
M_k_q_wk_ = interp_p_to_q(n_k_p_r,n_w_,n_w_sum,M_k_p_wk_);
M_k_q_wkM__(:,1+nM) = M_k_q_wk_;
end;%for nM=0:n_M-1;
tmp_t = toc(tmp_t); if (flag_verbose); disp(sprintf(' %% M_k_p_wkM__: %0.2fs',tmp_t)); end;
clear nCTF CTF_k_p_wk_ nS azimu_b polar_a index_gamma_z gamma_z S_k_p_wk_ T_k_p_wk_ M_k_p_wk_ M_k_q_wk_ ;
%%%%%%%%;
% quickly test vectorized construction. ;
%%%%%%%%;
N_k_p_wkM__ = zeros(n_w_sum,n_M);
N_k_p_wkM__ = CTF_k_p_wkC__(:,1+index_nCTF_from_nM_).*reshape(transf_p_to_p(n_k_p_r,k_p_r_,n_w_,n_w_sum,rotate_p_to_p_fftw(n_k_p_r,n_w_,n_w_sum,S_k_p_wkS__(:,1+index_nS_from_nM_),gamma_z_(1+index_nw_from_nM_)),-FTK.delta_x_(1+index_nd_from_nM_),-FTK.delta_y_(1+index_nd_from_nM_)),[n_w_sum,n_M]);
N_k_q_wkM__ = reshape(interp_p_to_q(n_k_p_r,n_w_,n_w_sum,N_k_p_wkM__),[n_w_sum,n_M]);
fnorm_disp(flag_verbose,'M_k_p_wkM__',M_k_p_wkM__,'N_k_p_wkM__',N_k_p_wkM__,' %%<-- should be zero');
fnorm_disp(flag_verbose,'M_k_q_wkM__',M_k_q_wkM__,'N_k_q_wkM__',N_k_q_wkM__,' %%<-- should be zero');
%%%%%%%%;
'''

flag_CTF = 1;
n_CTF = 3;
CTF_phi_C_ = pi*torch.tensor([+2/3,-1/5,+6/7]).to(dtype=torch.float32);
CTF_k_p_wkC__ = torch.zeros(mtr((n_w_sum,n_CTF))).to(dtype=torch.float32);
for nCTF in range(n_CTF):
    if flag_CTF==0: CTF_k_p_wkC__[nCTF,:] = torch.ones(n_w_sum).to(dtype=torch.float32);
    if flag_CTF!=0: CTF_k_p_wkC__[nCTF,:] = 2*k_p_r_wk_ * torch.cos(k_p_w_wk_ - CTF_phi_C_[nCTF].item());
#end;%for nCTF=0:n_CTF-1;
#%%%%%%%%;
tmp_t = tic();
n_M = 2*n_S; #%<-- pick something bigger than n_S;
n_gamma_z = n_w_max;
gamma_z_ = torch.linspace(0,2*pi,n_gamma_z+1).to(dtype=torch.float32); gamma_z_ = gamma_z_[0:n_gamma_z];
M_k_p_wkM__ = torch.zeros(mtr((n_w_sum,n_M))).to(dtype=torch.complex64);
index_nCTF_from_nM_ = torch.zeros(n_M).to(dtype=torch.int32);
index_nS_from_nM_ = torch.zeros(n_M).to(dtype=torch.int32);
euler_azimu_b_true_M_ = torch.zeros(n_M).to(dtype=torch.float32);
euler_polar_a_true_M_ = torch.zeros(n_M).to(dtype=torch.float32);
index_nw_from_nM_ = torch.zeros(n_M).to(dtype=torch.int32);
euler_gamma_z_true_M_ = torch.zeros(n_M).to(dtype=torch.float32);
index_nd_from_nM_ = torch.zeros(n_M).to(dtype=torch.int32);
image_delta_x_true_M_ = torch.zeros(n_M).to(dtype=torch.float32);
image_delta_y_true_M_ = torch.zeros(n_M).to(dtype=torch.float32);
for nM in range(n_M):
    nCTF = int(np.mod(nM,n_CTF));
    index_nCTF_from_nM_[nM] = nCTF;
    nS = int(np.maximum(0,np.minimum(n_S-1,np.mod(nM,n_S))));
    index_nS_from_nM_[nM] = nS;
    azimu_b = viewing_azimu_b_S_[nS].item();
    polar_a = viewing_polar_a_S_[nS].item();
    index_gamma_z = periodize(nM,0,n_gamma_z);
    gamma_z = gamma_z_[index_gamma_z];
    index_nw_from_nM_[nM] = index_gamma_z;
    euler_azimu_b_true_M_[nM] = azimu_b;
    euler_polar_a_true_M_[nM] = polar_a;
    euler_gamma_z_true_M_[nM] = gamma_z;
    index_nd_from_nM_[nM] = periodize(nM,0,n_delta_v);
    delta_x = FTK['r8_delta_x_'][index_nd_from_nM_[nM].item()].item(); delta_y = FTK['r8_delta_y_'][index_nd_from_nM_[nM].item()].item();
    image_delta_x_true_M_[nM] = delta_x;
    image_delta_y_true_M_[nM] = delta_y;
#end;%for nM=0:n_M-1;
tmp_t = toc(tmp_t); 
if (flag_verbose): print(f' %% M_k_p_wkM__: {tmp_t:.2f}s');
del nCTF;del nS;del azimu_b;del polar_a;del index_gamma_z;del gamma_z;
#%%%%%%%%;
#% vectorized construction. ;
#%%%%%%%%;
M_k_p_wkM__ = torch.zeros(mtr((n_w_sum,n_M))).to(dtype=torch.complex64);
M_k_p_wkM__ = CTF_k_p_wkC__[index_nCTF_from_nM_,:] * torch.reshape(transf_p_to_p(n_k_p_r,k_p_r_,n_w_,n_w_sum,rotate_p_to_p_fftw(n_k_p_r,n_w_,n_w_sum,S_k_p_wkS__[index_nS_from_nM_,:],gamma_z_[index_nw_from_nM_]),-FTK['r8_delta_x_'][index_nd_from_nM_],-FTK['r8_delta_y_'][index_nd_from_nM_]),mtr((n_w_sum,n_M)));
M_k_q_wkM__ = torch.reshape(interp_p_to_q(n_k_p_r,n_w_,n_w_sum,M_k_p_wkM__),mtr((n_w_sum,n_M)));
#%%%%%%%%;

r'''
%%%%%%%%;
% quickly test inner-product: <R(+gamma_z)*M_k_p_wk_,S_k_p_wk_> ;
%%%%%%%%;
nS = max(0,min(n_S-1,round(n_S*2/3)));
nM = max(0,min(n_M-1,round(n_M*1/5)));
CTF_k_p_wk_ = CTF_k_p_wkC__(:,1+index_nCTF_from_nM_(1+nM));
M_k_p_wk_ = CTF_k_p_wk_.*M_k_p_wkM__(:,1+nM);
M_k_q_wk_ = interp_p_to_q(n_k_p_r,n_w_,n_w_sum,M_k_p_wk_);
S_k_p_wk_ = S_k_p_wkS__(:,1+nS);
S_k_q_wk_ = interp_p_to_q(n_k_p_r,n_w_,n_w_sum,S_k_p_wk_);
tmp_1_w_ = ifft(sum(bsxfun(@times,reshape(conj(M_k_q_wk_).*S_k_q_wk_,[n_w_max,n_k_p_r]),reshape(weight_2d_k_p_r_,[1,n_k_p_r])),2));
tmp_0_w_ = zeros(n_w_max,1);
for nw=0:n_w_max-1;
gamma_z = gamma_z_(1+nw);
tmp_0_w_(1+nw) = sum(conj(rotate_p_to_p_fftw(n_k_p_r,n_w_,n_w_sum,M_k_p_wk_,+gamma_z)).*S_k_p_wk_.*weight_2d_wk_)*(2*pi)^2;
end;%for nw=0:n_w_max-1;
fnorm_disp(flag_verbose,'tmp_0_w_',tmp_0_w_,'tmp_1_w_',tmp_1_w_,' %%<-- should be zero');
%%%%%%%%;
'''

nS = np.maximum(0,np.minimum(n_S-1,matlab_scalar_round(n_S*2/3)));
nM = np.maximum(0,np.minimum(n_M-1,matlab_scalar_round(n_M*1/5)));
CTF_k_p_wk_ = CTF_k_p_wkC__[int(index_nCTF_from_nM_[nM].item()),:].ravel();
M_k_p_wk_ = CTF_k_p_wk_ * M_k_p_wkM__[nM,:].ravel();
M_k_q_wk_ = interp_p_to_q(n_k_p_r,n_w_,n_w_sum,M_k_p_wk_);
S_k_p_wk_ = S_k_p_wkS__[nS,:].ravel();
S_k_q_wk_ = interp_p_to_q(n_k_p_r,n_w_,n_w_sum,S_k_p_wk_);
tmp_1_w_ = torch.fft.ifft( torch.sum( torch.reshape(torch.conj(M_k_q_wk_)*S_k_q_wk_,mtr((n_w_max,n_k_p_r))) * torch.reshape(weight_2d_k_p_r_,mtr((1,n_k_p_r))),1-1) , dim=1-1 ).to(dtype=torch.complex64);
tmp_0_w_ = torch.zeros(n_w_max).to(dtype=torch.complex64);
for nw in range(n_w_max):
    gamma_z = gamma_z_[nw].item();
    tmp_0_w_[nw] = torch.sum(torch.conj(rotate_p_to_p_fftw(n_k_p_r,n_w_,n_w_sum,M_k_p_wk_,+gamma_z)) * S_k_p_wk_ * weight_2d_wk_).item()*(2*pi)**2;
#end;%for nw=0:n_w_max-1;
fnorm_disp(flag_verbose,'tmp_0_w_',tmp_0_w_,'tmp_1_w_',tmp_1_w_,' %%<-- should be zero');
#%%%%%%%%;

r'''

%%%%%%%%;
% Now construct the template-norms: ;
% <CTF_k_p_wk_.*(R(+gamma_z)*S_k_p_wk_),CTF_k_p_wk_.*(R(+gamma_z)*S_k_p_wk_)> ;
% Note that this does not involve collapsing onto principal-modes. ;
%%%%%%%%;
CTF_R_S_l2_wSC_quad___ = zeros(n_w_max,n_S,n_CTF);
SS_k_p_wkS__ = conj(S_k_p_wkS__).*S_k_p_wkS__;
SS_k_q_wkS__ = reshape(interp_p_to_q_block_0(n_k_p_r,n_w_,n_w_sum,SS_k_p_wkS__),[n_w_sum,n_S]);
CC_k_p_wkC__ = conj(CTF_k_p_wkC__).*CTF_k_p_wkC__;
CC_k_q_wkC__ = reshape(interp_p_to_q_block_0(n_k_p_r,n_w_,n_w_sum,CC_k_p_wkC__),[n_w_sum,n_CTF]);
for nCTF=0:n_CTF-1;
CTF_R_S_l2_wSC_quad___(:,:,1+nCTF) = ifft(squeeze(sum(bsxfun(@times,reshape(bsxfun(@times,CC_k_q_wkC__(:,1+nCTF),conj(SS_k_q_wkS__)),[n_w_max,n_k_p_r,n_S]),reshape(weight_2d_k_p_r_,[1,n_k_p_r,1])),1+1)));
end;%for nCTF=0:n_CTF-1;
%%%%%%%%;
% Now check one template. ;
%%%%%%%%;
nS = max(0,min(n_S-1,round(n_S*1/5)));
tmp_azimu_b = viewing_azimu_b_S_(1+nS);
tmp_polar_a = viewing_polar_a_S_(1+nS);
tmp_gamma_z = 0.0;
tmp_R_S__ = Rz(-tmp_gamma_z)*Ry(-tmp_polar_a)*Rz(-tmp_azimu_b);
nCTF = max(0,min(n_CTF-1,round(n_CTF*1/3)));
tmp_CTF_phi = CTF_phi_C_(1+nCTF);
CTF_R_S_l2_w_quad_ = CTF_R_S_l2_wSC_quad___(:,1+nS,1+nCTF);
CTF_R_S_l2_w_qua2_ = zeros(n_w_max,1);
CTF_R_S_l2_w_form_ = zeros(n_w_max,1);
for nw=0:n_w_max-1;
gamma_z = gamma_z_(1+nw);
CTF_R_S_k_p_wk_ = CTF_k_p_wkC__(:,1+nCTF).*rotate_p_to_p_fftw(n_k_p_r,n_w_,n_w_sum,S_k_p_wkS__(:,1+nS),+gamma_z);
CTF_R_S_l2_w_qua2_(1+nw) = sum(conj(CTF_R_S_k_p_wk_).*CTF_R_S_k_p_wk_.*weight_2d_wk_)*(2*pi)^2;
for nsource0=0:n_source-1;
tmp_S_delta_0_3_ = tmp_R_S__*delta_a_c_3s__(:,1+nsource0); tmp_S_delta_0_2_ = tmp_S_delta_0_3_(1:2);
for nsource1=0:n_source-1;
tmp_S_delta_1_3_ = tmp_R_S__*delta_a_c_3s__(:,1+nsource1); tmp_S_delta_1_2_ = tmp_S_delta_1_3_(1:2);
CTF_R_S_l2_w_form_(1+nw) = CTF_R_S_l2_w_form_(1+nw) + I_xPPx_0(k_p_r_max,tmp_CTF_phi-gamma_z,tmp_S_delta_0_2_,tmp_CTF_phi-gamma_z,tmp_S_delta_1_2_);
end;%for nsource0=0:n_source-1;
end;%for nsource1=0:n_source-1;
end;%for nw=0:n_w_max-1;
fnorm_disp(flag_verbose,'CTF_R_S_l2_w_qua2_',CTF_R_S_l2_w_qua2_,'CTF_R_S_l2_w_quad_',CTF_R_S_l2_w_quad_,' %%<-- should be zero');
fnorm_disp(flag_verbose,'CTF_R_S_l2_w_qua2_',CTF_R_S_l2_w_qua2_,'CTF_R_S_l2_w_form_',CTF_R_S_l2_w_form_,' %%<-- should be <1e-6');
fnorm_disp(flag_verbose,'CTF_R_S_l2_w_quad_',CTF_R_S_l2_w_quad_,'CTF_R_S_l2_w_form_',CTF_R_S_l2_w_form_,' %%<-- should be <1e-6');
%%%%%%%%;
'''

CTF_R_S_l2_wSC_quad___ = torch.zeros(mtr((n_w_max,n_S,n_CTF))).to(dtype=torch.float32);
SS_k_p_wkS__ = torch.conj(S_k_p_wkS__) * S_k_p_wkS__;
SS_k_q_wkS__ = torch.reshape(interp_p_to_q(n_k_p_r,n_w_,n_w_sum,SS_k_p_wkS__),mtr((n_w_sum,n_S)));
CC_k_p_wkC__ = torch.conj(CTF_k_p_wkC__) * CTF_k_p_wkC__;
CC_k_q_wkC__ = torch.reshape(interp_p_to_q(n_k_p_r,n_w_,n_w_sum,CC_k_p_wkC__),mtr((n_w_sum,n_CTF)));
for nCTF in range(n_CTF):
    CTF_R_S_l2_wSC_quad___[nCTF,:,:] = torch.real(torch.fft.ifft(torch.reshape(torch.sum(torch.reshape(CC_k_q_wkC__[nCTF,:].ravel()*torch.conj(SS_k_q_wkS__),mtr((n_w_max,n_k_p_r,n_S))) * torch.reshape(weight_2d_k_p_r_,mtr((1,n_k_p_r,1))) , 2-1),mtr((n_w_max,n_S))),dim=1-0));
#end;%for nCTF=0:n_CTF-1;
#%%%%%%%%;
nS = int(np.maximum(0,np.minimum(n_S-1,matlab_scalar_round(n_S*1/5))));
tmp_azimu_b = viewing_azimu_b_S_[nS].item();
tmp_polar_a = viewing_polar_a_S_[nS].item();
tmp_gamma_z = 0.0;
tmp_R_S__ = mmmm( Rz(-tmp_gamma_z) , mmmm( Ry(-tmp_polar_a) , Rz(-tmp_azimu_b) ) );
nCTF = int(np.maximum(0,np.minimum(n_CTF-1,matlab_scalar_round(n_CTF*1/3))));
tmp_CTF_phi = CTF_phi_C_[nCTF].item();
CTF_R_S_l2_w_quad_ = CTF_R_S_l2_wSC_quad___[nCTF,nS,:].ravel();
CTF_R_S_l2_w_qua2_ = torch.zeros(n_w_max).to(dtype=torch.float32);
CTF_R_S_l2_w_form_ = torch.zeros(n_w_max).to(dtype=torch.float32);
for nw in range(n_w_max):
    gamma_z = gamma_z_[nw].item();
    CTF_R_S_k_p_wk_ = CTF_k_p_wkC__[nCTF,:].ravel() * rotate_p_to_p_fftw(n_k_p_r,n_w_,n_w_sum,S_k_p_wkS__[nS,:].ravel(),+gamma_z);
    CTF_R_S_l2_w_qua2_[nw] = np.real(torch.sum(torch.conj(CTF_R_S_k_p_wk_.ravel()) * CTF_R_S_k_p_wk_.ravel() * weight_2d_wk_.ravel()).item()*(2*pi)**2);
    for nsource0 in range(n_source):
        tmp_S_delta_0_3_ = mmvm(tmp_R_S__,delta_a_c_3s__[nsource0,:].ravel()); tmp_S_delta_0_2_ = tmp_S_delta_0_3_[0:2].ravel();
        for nsource1 in range(n_source):
            tmp_S_delta_1_3_ = mmvm(tmp_R_S__,delta_a_c_3s__[nsource1,:].ravel()); tmp_S_delta_1_2_ = tmp_S_delta_1_3_[0:2].ravel();
            CTF_R_S_l2_w_form_[nw] = CTF_R_S_l2_w_form_[nw].item() + I_xPPx_0(k_p_r_max,tmp_CTF_phi-gamma_z,tmp_S_delta_0_2_,tmp_CTF_phi-gamma_z,tmp_S_delta_1_2_);
        #end;%for nsource0=0:n_source-1;
    #end;%for nsource1=0:n_source-1;
#end;%for nw=0:n_w_max-1;
fnorm_disp(flag_verbose,'CTF_R_S_l2_w_qua2_',CTF_R_S_l2_w_qua2_,'CTF_R_S_l2_w_quad_',CTF_R_S_l2_w_quad_,' %%<-- should be zero');
fnorm_disp(flag_verbose,'CTF_R_S_l2_w_qua2_',CTF_R_S_l2_w_qua2_,'CTF_R_S_l2_w_form_',CTF_R_S_l2_w_form_,' %%<-- should be <1e-6');
fnorm_disp(flag_verbose,'CTF_R_S_l2_w_quad_',CTF_R_S_l2_w_quad_,'CTF_R_S_l2_w_form_',CTF_R_S_l2_w_form_,' %%<-- should be <1e-6');
#%%%%%%%%;
dir_base = '/data/rangan' ;
dir_ascii = dir_base + '/dir_cryoem/dir_rangan_python/dir_ascii' ;
fname_ascii = dir_ascii + '/CTF_R_S_l2_w_quad_.ascii' ;
print(f" %% writing {fname_ascii}");
np.savetxt(fname_ascii,CTF_R_S_l2_w_quad_.numpy().ravel());
fname_ascii = dir_ascii + '/CTF_R_S_l2_w_qua2_.ascii' ;
print(f" %% writing {fname_ascii}");
np.savetxt(fname_ascii,CTF_R_S_l2_w_qua2_.numpy().ravel());
fname_ascii = dir_ascii + '/CTF_R_S_l2_w_form_.ascii' ;
print(f" %% writing {fname_ascii}");
np.savetxt(fname_ascii,CTF_R_S_l2_w_form_.numpy().ravel());
#%%%%%%%%;

r'''
flag_pm = 1;
%%%%%%%%%%%%%%%%;
% Now calculate innerproduct Z_dwSM____. ;
% Here we account for anisotropic CTF. ;
% Note that this calculation assumes the radial-projection is nearly lossless. ;
%%%%%%%%%%%%%%%%;
n_UX_rank = n_k_p_r-1; %<-- just to check dimensions. Note that, in principle, this could be different for different nCTF. ; 
pm_n_UX_rank = n_UX_rank; %<-- just to check dimension. ;
Z_dwSM_ampm____ = zeros(n_delta_v,n_w_max,n_S,n_M);
UX_M_l2_M_ = zeros(n_M,1);
UX_T_M_l2_dM__ = zeros(n_delta_v,n_M);
UX_knC___ = zeros(n_k_p_r,n_UX_rank,n_CTF);
X_weight_rC__ = zeros(n_k_p_r,n_CTF);
%%%%%%%%;
for nCTF=0:n_CTF-1;
if (flag_verbose); disp(sprintf(' %% nCTF %d/%d',nCTF,n_CTF)); end;
CTF_k_p_wk_ = CTF_k_p_wkC__(:,1+nCTF);
index_M_sub_ = efind(index_nCTF_from_nM_==nCTF); n_M_sub = numel(index_M_sub_);
%%%%;
% Prepare principal-modes. ;
%%%%;
UX_kn__ = eye(n_k_p_r,n_UX_rank);
X_weight_r_ = sqrt(weight_2d_k_p_r_);
if  flag_pm;
UX_kn__ = zeros(n_k_p_r,n_UX_rank);
X_weight_r_ = zeros(n_k_p_r,1);
[ ...
 X_kk__ ...
,X_weight_r_ ...
] = ...
principled_marching_empirical_cost_matrix_1( ...
 n_k_p_r ...
,k_p_r_ ...
,weight_2d_k_p_r_ ...
,n_w_ ...
,n_M_sub ...
,M_k_p_wkM__(:,1+index_M_sub_) ...
);
[tmp_UX_kn__,tmp_SX_k__,tmp_VX_kn__] = svds(X_kk__,n_UX_rank);
UX_kn__(:,1:n_UX_rank) = tmp_UX_kn__(:,1:n_UX_rank);
end;%if  flag_pm;
UX_knC___(:,:,1+nCTF) = UX_kn__;
X_weight_rC__(:,1+nCTF) = X_weight_r_;
%%%%;
% Prepare quasi-images. ;
%%%%;
tmp_t = tic();
CTF_M_sub_k_p_wkM__ = bsxfun(@times,CTF_k_p_wk_,M_k_p_wkM__(:,1+index_M_sub_));
CTF_M_sub_k_q_wkM__ = zeros(n_w_sum,n_M_sub);
for nM_sub=0:n_M_sub-1;
CTF_M_sub_k_q_wkM__(:,1+nM_sub) = interp_p_to_q(n_k_p_r,n_w_,n_w_sum,CTF_M_sub_k_p_wkM__(:,1+nM_sub));
end;%for nM_sub=0:n_M_sub-1;
M_sub_k_q_wkM__ = M_k_q_wkM__(:,1+index_M_sub_);
svd_VUXCTFM_sub_lwnM____ = tpmh_VUXM_lwnM____3(FTK,n_k_p_r,n_w_,n_M_sub,CTF_M_sub_k_q_wkM__,pm_n_UX_rank,UX_kn__,X_weight_r_);
svd_VUXM_sub_lwnM____ = tpmh_VUXM_lwnM____3(FTK,n_k_p_r,n_w_,n_M_sub,M_sub_k_q_wkM__,pm_n_UX_rank,UX_kn__,X_weight_r_);
tmp_t = toc(tmp_t); if (flag_verbose); disp(sprintf(' %% tmpmh_VUXM_lwnM____3: %0.2fs',tmp_t)); end;
%%%%;
% Now calculate norms of the translated images. ;
%%%%;
tmp_t = tic();
UX_T_M_sub_l2_dM__ = tfpmh_UX_T_M_l2_dM__1(FTK,n_w_,n_M_sub,pm_n_UX_rank,svd_VUXM_sub_lwnM____);
tmp_t = toc(tmp_t); if (flag_verbose); disp(sprintf(' %% tfpmh_UX_T_M_sub_l2_dm__1: %0.2fs',tmp_t)); end;
tmp_index_d0 = intersect(efind(FTK.delta_x_==0),efind(FTK.delta_y_==0)); assert(numel(tmp_index_d0)==1); %<-- should be zero-displacement. ;
UX_M_sub_l2_M_ = reshape(UX_T_M_sub_l2_dM__(1+tmp_index_d0,:),[n_M_sub,1]);
%%%%;
% Prepare UX_S_k_q_wnS__. ;
%%%%;
tmp_t = tic();
UX_S_k_q_wnS__ = reshape(permute(reshape(reshape(S_k_q_wSk___,[n_w_max*n_S,n_k_p_r])*diag(X_weight_r_)*UX_kn__,[n_w_max,n_S,pm_n_UX_rank]),1+[0,2,1]),[n_w_max*pm_n_UX_rank,n_S]);
tmp_t = toc(tmp_t); if (flag_verbose); disp(sprintf(' %% UX_S_k_q_wnS__: %0.2fs',tmp_t)); end;
%%%%;
% Calculate Z_sub_dwSM____. ;
%%%%;
tmp_t = tic();
UX_S_k_q_nSw___ = permute(reshape(UX_S_k_q_wnS__,[n_w_max,n_UX_rank,n_S]),1+[1,2,0]);
tmp_t_sub = tic();
svd_VUXCTFM_sub_nMwl____ = permute(svd_VUXCTFM_sub_lwnM____,1+[2,3,1,0]);
tmp_t_sub = toc(tmp_t_sub); if (flag_verbose); disp(sprintf(' %% svd_VUXCTFM_sub_nMwl____: %0.2fs',tmp_t_sub)); end;
tmp_t_sub = tic();
%svd_SVUXCTFM_sub_SMwl____ = zeros(n_S,n_M_sub,n_w_max,FTK.n_svd_l);
%for nl=0:FTK.n_svd_l-1; for nw=0:n_w_max-1;
%svd_SVUXCTFM_sub_SMwl____(:,:,1+nw,1+nl) = ctranspose(UX_S_k_q_nSw___(:,:,1+nw))*svd_VUXCTFM_sub_nMwl____(:,:,1+nw,1+nl);
%end;end;%for nw=0:n_w_max-1;for nl=0:FTK.n_svd_l-1;
svd_SVUXCTFM_sub_SMwl____ = pagemtimes(pagectranspose(UX_S_k_q_nSw___),svd_VUXCTFM_sub_nMwl____);
tmp_t_sub = toc(tmp_t_sub); if (flag_verbose>1); disp(sprintf(' %% svd_SVUXCTFM_SMwl____: %0.6f',tmp_t_sub)); end;
tmp_t_sub = tic();
%svd_SVUXCTFM_sub_lwSM____ = permute(ifft(permute(svd_SVUXCTFM_sub_SMwl____,1+[2,3,0,1]),[],1)*n_w_max,1+[1,0,2,3]);
svd_SVUXCTFM_sub_lwSM____ = ifft(permute(svd_SVUXCTFM_sub_SMwl____,1+[3,2,0,1]),[],1+1)*n_w_max;
tmp_t_sub = toc(tmp_t_sub); if (flag_verbose>1); disp(sprintf(' %% svd_SVUXCTFM_sub_lwSM____: %0.6f',tmp_t_sub)); end;
tmp_t_sub = tic(); nop=0;
svd_USESVUXCTFM_sub_dwSM____ = reshape(FTK.svd_U_d_expiw_s__*reshape(svd_SVUXCTFM_sub_lwSM____,[FTK.n_svd_l,n_w_max*n_S*n_M_sub]),[FTK.n_delta_v,n_w_max,n_S,n_M_sub]);
%svd_USESVUXCTFM_sub_dwSM____ = reshape(pagemtimes(FTK.svd_U_d_expiw_s__,svd_SVUXCTFM_sub_lwSM____),[FTK.n_delta_v,n_w_max,n_S,n_M_sub]);
tmp_t_sub = toc(tmp_t_sub); if (flag_verbose>1); disp(sprintf(' %% svd_USESVUXCTFM_sub_dwSM____: %0.6f',tmp_t_sub)); end;
tmp_t = toc(tmp_t); if (flag_verbose); disp(sprintf(' %% Z_sub_dwSM____: %0.2fs',tmp_t)); end;
%%%%;
% Store results. ;
%%%%;
UX_M_l2_M_(1+index_M_sub_) = UX_M_sub_l2_M_;
UX_T_M_l2_dM__(:,1+index_M_sub_) = UX_T_M_sub_l2_dM__;
Z_dwSM_ampm____(:,:,:,1+index_M_sub_) = svd_USESVUXCTFM_sub_dwSM____ ;
%%%%;
clear UX_kn__ X_weight_r_ CTF_k_p_wk_ index_M_sub_ UX_T_M_sub_l2_dM__ UX_M_sub_l2_M_ ;
clear UX_S_k_q_wnS__ ;
clear svd_VUXCTFM_sub_lwnM____ svd_VUXCTFM_sub_nMwl____ ;
clear svd_SVUXCTFM_sub_SMwl____ svd_SVUXCTFM_sub_SMwl____ svd_SVUXCTFM_sub_lwSM____ ;
clear svd_USESVUXCTFM_sub_dwSM____ ;
%%%%%%%%;
end;%for nCTF=0:n_CTF-1;
%%%%%%%%;
clear nCTF;
%%%%%%%%%%%%%%%%;
'''

flag_pm = 1;
#%%%%%%%%%%%%%%%%;
device_use = 'cpu'; #<-- batch sizes uncontrolled below, cannot easily push onto cuda. ;
n_UX_rank = n_k_p_r-1; #%<-- just to check dimensions. Note that, in principle, this could be different for different nCTF. ; 
pm_n_UX_rank = n_UX_rank; #%<-- just to check dimension. Note that, in principle, this could be different for different nCTF. ; 
Z_dwSM_ampm____ = torch.zeros(mtr((n_delta_v,n_w_max,n_S,n_M))).to(dtype=torch.complex64);
UX_M_l2_M_ = torch.zeros(n_M).to(dtype=torch.float32);
UX_T_M_l2_dM__ = torch.zeros(mtr((n_delta_v,n_M))).to(dtype=torch.float32);
UX_knC___ = torch.zeros(mtr((n_k_p_r,n_UX_rank,n_CTF))).to(dtype=torch.float32);
X_weight_rC__ = torch.zeros(mtr((n_k_p_r,n_CTF))).to(dtype=torch.float32);
#%%%%%%%%;
for nCTF in range(n_CTF):
    if (flag_verbose): print(f' %% nCTF {nCTF}/{n_CTF}');
    CTF_k_p_wk_ = CTF_k_p_wkC__[nCTF,:].ravel();
    if (flag_verbose>1): print(f' %% nCTF {nCTF}/{n_CTF}, CTF_k_p_wk_: {fnorm(CTF_k_p_wk_)}');
    index_M_sub_ = efind(index_nCTF_from_nM_==nCTF); n_M_sub = numel(index_M_sub_);
    tmp_i8_index_rhs_ = matlab_index_2d_0(n_w_sum,':',n_M,index_M_sub_);
    M_sub_k_p_wkM__ = torch.reshape(M_k_p_wkM__.ravel()[tmp_i8_index_rhs_],mtr((n_w_sum,numel(index_M_sub_))));
    if (flag_verbose>1): print(f' %% nCTF {nCTF}/{n_CTF}, M_sub_k_p_wkM__: {fnorm(M_sub_k_p_wkM__)}');
    M_sub_k_q_wkM__ = interp_p_to_q(n_k_p_r,n_w_,n_w_sum,M_sub_k_p_wkM__);
    if (flag_verbose>1): print(f' %% nCTF {nCTF}/{n_CTF}, M_sub_k_q_wkM__: {fnorm(M_sub_k_q_wkM__)}');
    #%%%%;
    tmp_n = int(np.maximum(n_k_p_r,n_UX_rank)); UX_kn__ = torch.eye(tmp_n).to(dtype=torch.float32); tmp_i8_index_rhs_ = matlab_index_2d_0(tmp_n,torch.arange(n_k_p_r),tmp_n,torch.arange(n_UX_rank)); UX_kn__ = torch.reshape(UX_kn__.ravel()[tmp_i8_index_rhs_],mtr((n_k_p_r,n_UX_rank))).to(dtype=torch.float32);
    X_weight_r_ = torch.sqrt(weight_2d_k_p_r_);
    if (flag_verbose>1): print(f' %% nCTF {nCTF}/{n_CTF}, X_weight_r_: {fnorm(X_weight_r_)}');
    if  flag_pm:
        UX_kn__ = torch.zeros(mtr((n_k_p_r,n_UX_rank))).to(dtype=torch.float32);
        X_weight_r_ = torch.zeros(n_k_p_r).to(dtype=torch.float32);
        (
            X_kk__,
            X_weight_r_,
        ) = principled_marching_empirical_cost_matrix_1(
            n_k_p_r,
            k_p_r_,
            weight_2d_k_p_r_,
            n_w_,
            n_M_sub,
            M_sub_k_p_wkM__,
        );
        tmp_UX_kn__ , tmp_SX_k_ , tmp_VX_kn__ = torch.linalg.svd(X_kk__.T,full_matrices=False); tmp_UX_kn__ = tmp_UX_kn__.T; #%<-- extra transposes to match matlab. ;
        tmp_UX_kn__ = tmp_UX_kn__[0:n_UX_rank,:]; tmp_SX_k_ = tmp_SX_k_[0:n_UX_rank]; tmp_VX_kn__ = tmp_VX_kn__[0:n_UX_rank,:];
        UX_kn__ = tmp_UX_kn__;
    #end;%if  flag_pm;
    if (flag_verbose>1): print(f' %% nCTF {nCTF}/{n_CTF}, UX_kn__: {fnorm(UX_kn__)}');
    UX_knC___[nCTF,:,:] = UX_kn__;
    X_weight_rC__[nCTF,:] = X_weight_r_;
    #%%%%;
    tmp_t = tic();
    CTF_M_sub_k_p_wkM__ = torch.reshape(CTF_k_p_wk_,mtr((n_w_sum,1))) * torch.reshape(M_sub_k_p_wkM__,mtr((n_w_sum,n_M_sub))).to(dtype=torch.complex64);
    CTF_M_sub_k_q_wkM__ = interp_p_to_q(n_k_p_r,n_w_,n_w_sum,CTF_M_sub_k_p_wkM__).to(dtype=torch.complex64);
    svd_VUXCTFM_sub_lwnM____ = tpmh_VUXM_lwnM____3(FTK,n_k_p_r,n_w_,n_M_sub,CTF_M_sub_k_q_wkM__,pm_n_UX_rank,UX_kn__,X_weight_r_).to(dtype=torch.complex64);
    svd_VUXM_sub_lwnM____ = tpmh_VUXM_lwnM____3(FTK,n_k_p_r,n_w_,n_M_sub,M_sub_k_q_wkM__,pm_n_UX_rank,UX_kn__,X_weight_r_).to(dtype=torch.complex64);
    tmp_t = toc(tmp_t);
    if (flag_verbose): print(f' %% tmpmh_VUXM_lwnM____3: {tmp_t:.2f}s');
    if (flag_verbose>1): print(f' %% nCTF {nCTF}/{n_CTF}, CTF_M_sub_k_p_wkM__: {fnorm(CTF_M_sub_k_p_wkM__)}');
    if (flag_verbose>1): print(f' %% nCTF {nCTF}/{n_CTF}, CTF_M_sub_k_q_wkM__: {fnorm(CTF_M_sub_k_q_wkM__)}');
    if (flag_verbose>1): print(f' %% nCTF {nCTF}/{n_CTF}, svd_VUXCTFM_sub_lwnM____: {fnorm(svd_VUXCTFM_sub_lwnM____)}');
    if (flag_verbose>1): print(f' %% nCTF {nCTF}/{n_CTF}, svd_VUXM_sub_lwnM____: {fnorm(svd_VUXM_sub_lwnM____)}');
    #%%%%;
    tmp_t = tic();
    UX_T_M_sub_l2_dM__ = tfpmh_UX_T_M_l2_dM__1(FTK,n_w_,n_M_sub,pm_n_UX_rank,svd_VUXM_sub_lwnM____).to(dtype=torch.float32);
    tmp_t = toc(tmp_t); 
    if (flag_verbose): print(f' %% tfpmh_UX_T_M_sub_l2_dm__1: {tmp_t:.2f}s');
    tmp_index_d0 = intersect_0(efind(torch.abs(FTK['r8_delta_x_'])< 1e-6),efind(torch.abs(FTK['r8_delta_y_'])< 1e-6))[0];
    assert(numel(tmp_index_d0)>=1); #%<-- should be zero-displacement. ;
    if numel(tmp_index_d0)> 1: tmp_index_d0 = tmp_index_d0[0]; #end;
    tmp_i8_index_rhs_ = matlab_index_2d_0(FTK['n_delta_v'],tmp_index_d0,n_M_sub,':');
    UX_M_sub_l2_M_ = UX_T_M_sub_l2_dM__.ravel()[tmp_i8_index_rhs_].ravel(); assert(numel(UX_M_sub_l2_M_)==n_M_sub);
    if (flag_verbose>1): print(f' %% nCTF {nCTF}/{n_CTF}, UX_M_sub_l2_M_: {fnorm(UX_M_sub_l2_M_)}');
    #%%%%;
    tmp_t = tic();
    UX_S_k_q_wnS__ = torch.reshape(torch.permute(torch.reshape(mmmm( torch.reshape(S_k_q_wSk___.to(dtype=torch.complex64),mtr((n_w_max*n_S,n_k_p_r)))*torch.reshape(X_weight_r_.to(dtype=torch.complex64),mtr((1,n_k_p_r))) , UX_kn__.to(dtype=torch.complex64) ),mtr((n_w_max,n_S,pm_n_UX_rank))),mtr(mts((0,2,1)))),mtr((n_w_max*pm_n_UX_rank,n_S))).to(dtype=torch.complex64);
    tmp_t = toc(tmp_t);
    if (flag_verbose): print(f' %% UX_S_k_q_wnS__: {tmp_t:.2f}s');
    if (flag_verbose>1): print(f' %% nCTF {nCTF}/{n_CTF}, UX_S_k_q_wnS__: {fnorm(UX_S_k_q_wnS__)}');
    #%%%%;
    tmp_t = tic();
    UX_S_k_q_nSw___ = torch.permute(torch.reshape(UX_S_k_q_wnS__.to(dtype=torch.complex64),mtr((n_w_max,n_UX_rank,n_S))),mtr(mts((1,2,0)))).to(dtype=torch.complex64);
    tmp_t_sub = tic();
    if (flag_verbose>1): print(f' %% nCTF {nCTF}/{n_CTF}, UX_S_k_q_nSw___: {fnorm(UX_S_k_q_nSw___)}');
    svd_VUXCTFM_sub_nMwl____ = torch.permute(svd_VUXCTFM_sub_lwnM____.to(dtype=torch.complex64),mtr(mts((2,3,1,0)))).to(dtype=torch.complex64);
    tmp_t_sub = toc(tmp_t_sub);
    if (flag_verbose): print(f' %% svd_VUXCTFM_sub_nMwl____: {tmp_t_sub:.2f}s');
    if (flag_verbose>1): print(f' %% nCTF {nCTF}/{n_CTF}, svd_VUXCTFM_sub_nMwl____: {fnorm(svd_VUXCTFM_sub_nMwl____)}');
    tmp_t_sub = tic();
    str_einsum = msr('nSw') + ',' + msr('nMwl') + '->' + msr('SMwl') ;
    svd_SVUXCTFM_sub_SMwl____ = torch.einsum(str_einsum,torch.conj(UX_S_k_q_nSw___).to(dtype=torch.complex64),svd_VUXCTFM_sub_nMwl____.to(dtype=torch.complex64)).to(dtype=torch.complex64);
    tmp_t_sub = toc(tmp_t_sub);
    if (flag_verbose>1): print(f' %% svd_SVUXCTFM_sub_SMwl____: {tmp_t_sub:.6f}s');
    if (flag_verbose>1): print(f' %% nCTF {nCTF}/{n_CTF}, svd_SVUXCTFM_sub_SMwl____: {fnorm(svd_SVUXCTFM_sub_SMwl____)}');
    tmp_t_sub = tic();
    svd_SVUXCTFM_sub_lwSM____ = torch.fft.ifft(torch.permute(svd_SVUXCTFM_sub_SMwl____.to(dtype=torch.complex64),mtr(mts((3,2,0,1)))),dim=3-1).to(dtype=torch.complex64)*n_w_max;
    tmp_t_sub = toc(tmp_t_sub);
    if (flag_verbose>1): print(f' %% svd_SVUXCTFM_sub_lwSM____: {tmp_t_sub:.6f}s');
    if (flag_verbose>1): print(f' %% nCTF {nCTF}/{n_CTF}, svd_SVUXCTFM_sub_lwSM____: {fnorm(svd_SVUXCTFM_sub_lwSM____)}');
    tmp_t_sub = tic();
    svd_USESVUXCTFM_sub_dwSM____ = torch.reshape( mmmm( torch.reshape(FTK['c16_svd_U_d_expiw_s__'].to(dtype=torch.complex64),mtr((FTK['n_delta_v'],FTK['n_svd_l']))) , torch.reshape(svd_SVUXCTFM_sub_lwSM____.to(dtype=torch.complex64),mtr((FTK['n_svd_l'],n_w_max*n_S*n_M_sub))) ),mtr((FTK['n_delta_v'],n_w_max,n_S,n_M_sub))).to(dtype=torch.complex64);
    tmp_t_sub = toc(tmp_t_sub);
    if (flag_verbose>1): print(f' %% svd_USESVUXCTFM_sub_dwSM____: {tmp_t_sub:.6f}s');
    if (flag_verbose>1): print(f' %% nCTF {nCTF}/{n_CTF}, svd_USESVUXCTFM_sub_dwSM____: {fnorm(svd_USESVUXCTFM_sub_dwSM____)}');
    tmp_t = toc(tmp_t);
    if (flag_verbose): print(f' %% Z_sub_dwSM____: {tmp_t:.2f}s');
    #%%%%;
    UX_M_l2_M_[index_M_sub_] = UX_M_sub_l2_M_.ravel();
    tmp_i8_index_lhs_ = matlab_index_2d_0(FTK['n_delta_v'],':',n_M,index_M_sub_);
    UX_T_M_l2_dM__.ravel()[tmp_i8_index_lhs_] = UX_T_M_sub_l2_dM__.ravel();
    tmp_i8_index_lhs_ = matlab_index_4d_0(FTK['n_delta_v'],':',n_w_max,':',n_S,':',n_M,index_M_sub_);
    Z_dwSM_ampm____.ravel()[tmp_i8_index_lhs_] = svd_USESVUXCTFM_sub_dwSM____.ravel() ;
    #%%%%;
    del UX_kn__;del X_weight_r_;del CTF_k_p_wk_;del index_M_sub_;del UX_T_M_sub_l2_dM__;del UX_M_sub_l2_M_;
    del UX_S_k_q_wnS__;
    del svd_VUXCTFM_sub_lwnM____;del svd_VUXCTFM_sub_nMwl____;
    del svd_SVUXCTFM_sub_SMwl____;del svd_SVUXCTFM_sub_lwSM____;
    del svd_USESVUXCTFM_sub_dwSM____;
    #%%%%%%%%;
#end;%for nCTF=0:n_CTF-1;
del nCTF;
#%%%%%%%%%%%%%%%%;
dir_base = '/data/rangan' ;
dir_ascii = dir_base + '/dir_cryoem/dir_rangan_python/dir_ascii' ;
fname_ascii = dir_ascii + '/UX_T_M_l2_dM__.ascii' ;
print(f" %% writing {fname_ascii}");
np.savetxt(fname_ascii,UX_T_M_l2_dM__.numpy().ravel());
fname_ascii = dir_ascii + '/UX_M_l2_M_.ascii' ;
print(f" %% writing {fname_ascii}");
np.savetxt(fname_ascii,UX_M_l2_M_.numpy().ravel());
#%%%%%%%%;

r'''
%%%%%%%%;
% Now re-construct the template-norms, this time limited to radial principal-modes: ;
% <((R(+gamma_z)*CTF_k_p_wk_).*S_k_p_wk_) * wUX_kn__),((R(+gamma_z)*CTF_k_p_wk_).*S_k_p_wk_) * wUX_kn__)> ;
% Note that this does yes involve collapsing onto principal-modes. ;
% Note that this calculation assumes the radial-projection is nearly lossless. ;
%%%%%%%%;
pm_n_k_p_r = pm_n_UX_rank; pm_n_w_max = n_w_max;
pm_n_w_ = pm_n_w_max*ones(pm_n_k_p_r,1);
pm_n_w_sum = pm_n_k_p_r*pm_n_w_max;
UX_CTF_R_S_l2_wSC_quad___ = zeros(n_w_max,n_S,n_CTF);
SS_k_p_wkS__ = conj(S_k_p_wkS__).*S_k_p_wkS__;
SS_k_q_wkS__ = reshape(interp_p_to_q_block_0(n_k_p_r,n_w_,n_w_sum,SS_k_p_wkS__),[n_w_sum,n_S]);
CC_k_p_wkC__ = conj(CTF_k_p_wkC__).*CTF_k_p_wkC__;
CC_k_q_wkC__ = reshape(interp_p_to_q_block_0(n_k_p_r,n_w_,n_w_sum,CC_k_p_wkC__),[n_w_sum,n_CTF]);
for nCTF=0:n_CTF-1;
UX_kn__ = UX_knC___(:,:,1+nCTF);
X_weight_r_ = X_weight_rC__(:,1+nCTF);
wUX_kn__ = diag(X_weight_r_)*UX_kn__;
UX_SS_k_q_wnS__ = reshape(permute(pagemtimes(transpose(wUX_kn__),permute(reshape(SS_k_q_wkS__,[n_w_max,n_k_p_r,n_S]),1+[1,0,2])),1+[1,0,2]),[pm_n_w_sum,n_S]);
UX_CC_k_q_wn_ = reshape(reshape(CC_k_q_wkC__(:,1+nCTF),[n_w_max,n_k_p_r])*wUX_kn__,[pm_n_w_sum,1]);
UX_CTF_R_S_l2_wSC_quad___(:,:,1+nCTF) = ifft(squeeze(sum(reshape(bsxfun(@times,UX_CC_k_q_wn_,conj(UX_SS_k_q_wnS__)),[pm_n_w_max,pm_n_k_p_r,n_S]),1+1)));
clear UX_kn__ X_weight_r_ wUX_kn__ ;
end;%for nCTF=0:n_CTF-1;
%%%%%%%%;
% Now check one template and one CTF. ;
%%%%%%%%;
nS = max(0,min(n_S-1,round(n_S*1/5)));
nCTF = max(0,min(n_CTF-1,round(n_CTF*2/3)));
UX_kn__ = UX_knC___(:,:,1+nCTF); X_weight_r_ = X_weight_rC__(:,1+nCTF); wUX_kn__ = diag(X_weight_r_)*UX_kn__;
CTF_R_S_l2_w_quad_ = CTF_R_S_l2_wSC_quad___(:,1+nS,1+nCTF);
UX_CTF_R_S_l2_w_quad_ = UX_CTF_R_S_l2_wSC_quad___(:,1+nS,1+nCTF);
CTF_R_S_l2_w_qua2_ = zeros(n_w_max,1);
UX_CTF_R_S_l2_w_qua2_ = zeros(n_w_max,1);
for nw=0:n_w_max-1;
gamma_z = gamma_z_(1+nw);
CTF_R_S_k_p_wk_ = CTF_k_p_wkC__(:,1+nCTF).*rotate_p_to_p_fftw(n_k_p_r,n_w_,n_w_sum,S_k_p_wkS__(:,1+nS),+gamma_z);
CTF_R_S_l2_w_qua2_(1+nw) = sum(conj(CTF_R_S_k_p_wk_).*CTF_R_S_k_p_wk_.*weight_2d_wk_)*(2*pi)^2;
UX_CTF_R_S_k_p_wn_ = reshape(reshape(CTF_k_p_wkC__(:,1+nCTF).*rotate_p_to_p_fftw(n_k_p_r,n_w_,n_w_sum,S_k_p_wkS__(:,1+nS),+gamma_z),[n_w_max,n_k_p_r])*wUX_kn__,[pm_n_w_sum,1]);
UX_CTF_R_S_l2_w_qua2_(1+nw) = sum(conj(UX_CTF_R_S_k_p_wn_).*UX_CTF_R_S_k_p_wn_)/max(1,n_w_max);
end;%for nw=0:n_w_max-1;
clear UX_kn__ X_weight_r_ wUX_kn__ UX_CTF_R_S_k_p_wn_ ;
fnorm_disp(flag_verbose,'CTF_R_S_l2_w_qua2_',CTF_R_S_l2_w_qua2_,'CTF_R_S_l2_w_quad_',CTF_R_S_l2_w_quad_,' %%<-- should be zero');
fnorm_disp(flag_verbose,'CTF_R_S_l2_w_qua2_',CTF_R_S_l2_w_qua2_,'UX_CTF_R_S_l2_w_quad_',UX_CTF_R_S_l2_w_quad_,' %%<-- should be zero');
fnorm_disp(flag_verbose,'UX_CTF_R_S_l2_w_qua2_',UX_CTF_R_S_l2_w_qua2_,'CTF_R_S_l2_w_quad_',CTF_R_S_l2_w_quad_,' %%<-- should be zero');
fnorm_disp(flag_verbose,'UX_CTF_R_S_l2_w_qua2_',UX_CTF_R_S_l2_w_qua2_,'UX_CTF_R_S_l2_w_quad_',UX_CTF_R_S_l2_w_quad_,' %%<-- should be zero');
%%%%%%%%;
'''

pm_n_k_p_r = int(pm_n_UX_rank); pm_n_w_max = int(n_w_max);
pm_n_w_ = pm_n_w_max*torch.ones(pm_n_k_p_r).to(dtype=torch.int32);
pm_n_w_sum = int(pm_n_k_p_r*pm_n_w_max);
UX_CTF_R_S_l2_wSC_quad___ = torch.zeros(mtr((n_w_max,n_S,n_CTF))).to(dtype=torch.float32);
SS_k_p_wkS__ = torch.conj(S_k_p_wkS__) * S_k_p_wkS__;
SS_k_q_wkS__ = torch.reshape(interp_p_to_q(n_k_p_r,n_w_,n_w_sum,SS_k_p_wkS__),mtr((n_w_sum,n_S)));
CC_k_p_wkC__ = torch.conj(CTF_k_p_wkC__) * CTF_k_p_wkC__;
CC_k_q_wkC__ = torch.reshape(interp_p_to_q(n_k_p_r,n_w_,n_w_sum,CC_k_p_wkC__),mtr((n_w_sum,n_CTF)));
for nCTF in range(n_CTF):
    UX_kn__ = torch.reshape(UX_knC___[nCTF,:,:],mtr((n_k_p_r,pm_n_k_p_r)));
    X_weight_r_ = X_weight_rC__[nCTF,:].ravel();
    wUX_kn__ = torch.reshape(X_weight_r_,mtr((n_k_p_r,1))) * torch.reshape(UX_kn__,mtr((n_k_p_r,pm_n_k_p_r)));
    str_einsum = msr('kn') + ',' + msr('wkS') + '->' + msr('wnS') ;
    UX_SS_k_q_wnS__ = torch.reshape(torch.einsum(str_einsum,wUX_kn__.to(dtype=torch.complex64),torch.reshape(SS_k_q_wkS__,mtr((n_w_max,n_k_p_r,n_S))).to(dtype=torch.complex64)),mtr((pm_n_w_sum,n_S))).to(dtype=torch.complex64);
    UX_CC_k_q_wn_ = torch.reshape(mmmm( torch.reshape(CC_k_q_wkC__[nCTF,:].to(dtype=torch.complex64),mtr((n_w_max,n_k_p_r))) , wUX_kn__.to(dtype=torch.complex64) ),mtr((pm_n_w_sum,1))).to(dtype=torch.complex64);
    UX_CTF_R_S_l2_wSC_quad___[nCTF,:,:] = torch.real(torch.fft.ifft(torch.reshape(torch.sum(torch.reshape(UX_CC_k_q_wn_.ravel() * torch.conj(UX_SS_k_q_wnS__),mtr((pm_n_w_max,pm_n_k_p_r,n_S))),2-1),mtr((pm_n_w_max,n_S))),dim=1-0)).to(dtype=torch.float32);
    del UX_kn__;del X_weight_r_;del wUX_kn__;
#end;%for nCTF=0:n_CTF-1;
#%%%%%%%%;
nS = int(np.maximum(0,np.minimum(n_S-1,matlab_scalar_round(n_S*1/5))));
nCTF = int(np.maximum(0,np.minimum(n_CTF-1,matlab_scalar_round(n_CTF*2/3))));
UX_kn__ = torch.reshape(UX_knC___[nCTF,:,:],mtr((n_k_p_r,pm_n_k_p_r))); X_weight_r_ = X_weight_rC__[nCTF,:].ravel(); 
wUX_kn__ = torch.reshape(X_weight_r_,mtr((n_k_p_r,1))) * torch.reshape(UX_kn__,mtr((n_k_p_r,pm_n_k_p_r)));
CTF_R_S_l2_w_quad_ = CTF_R_S_l2_wSC_quad___[nCTF,nS,:].ravel();
UX_CTF_R_S_l2_w_quad_ = UX_CTF_R_S_l2_wSC_quad___[nCTF,nS,:].ravel();
CTF_R_S_l2_w_qua2_ = torch.zeros(n_w_max).to(dtype=torch.float32);
UX_CTF_R_S_l2_w_qua2_ = torch.zeros(n_w_max).to(dtype=torch.float32);
for nw in range(n_w_max):
    gamma_z = gamma_z_[nw].item();
    CTF_R_S_k_p_wk_ = CTF_k_p_wkC__[nCTF,:].ravel() * rotate_p_to_p_fftw(n_k_p_r,n_w_,n_w_sum,S_k_p_wkS__[nS,:].ravel(),+gamma_z);
    CTF_R_S_l2_w_qua2_[nw] = np.real(torch.sum(torch.conj(CTF_R_S_k_p_wk_.ravel()) * CTF_R_S_k_p_wk_.ravel() * weight_2d_wk_.ravel()).item()*(2*pi)**2);
    UX_CTF_R_S_k_p_wn_ = mmmm( torch.reshape(CTF_k_p_wkC__[nCTF,:].ravel()*rotate_p_to_p_fftw(n_k_p_r,n_w_,n_w_sum,S_k_p_wkS__[nS,:].ravel(),+gamma_z),mtr((n_w_max,n_k_p_r))).to(dtype=torch.complex64) , wUX_kn__.to(dtype=torch.complex64) ).ravel();
    UX_CTF_R_S_l2_w_qua2_[nw] = np.real(torch.sum(torch.conj(UX_CTF_R_S_k_p_wn_.ravel()) * UX_CTF_R_S_k_p_wn_.ravel()).item()/np.maximum(1,n_w_max));
#end;%for nw=0:n_w_max-1;
del UX_kn__;del X_weight_r_;del wUX_kn__;del UX_CTF_R_S_k_p_wn_;
fnorm_disp(flag_verbose,'CTF_R_S_l2_w_qua2_',CTF_R_S_l2_w_qua2_,'CTF_R_S_l2_w_quad_',CTF_R_S_l2_w_quad_,' %%<-- should be zero');
fnorm_disp(flag_verbose,'CTF_R_S_l2_w_qua2_',CTF_R_S_l2_w_qua2_,'UX_CTF_R_S_l2_w_quad_',UX_CTF_R_S_l2_w_quad_,' %%<-- should be zero');
fnorm_disp(flag_verbose,'UX_CTF_R_S_l2_w_qua2_',UX_CTF_R_S_l2_w_qua2_,'CTF_R_S_l2_w_quad_',CTF_R_S_l2_w_quad_,' %%<-- should be zero');
fnorm_disp(flag_verbose,'UX_CTF_R_S_l2_w_qua2_',UX_CTF_R_S_l2_w_qua2_,'UX_CTF_R_S_l2_w_quad_',UX_CTF_R_S_l2_w_quad_,' %%<-- should be zero');
#%%%%%%%%;

r'''
%%%%%%%%;
% Now estimate landscape of innerproducts across delta_x_ and delta_y_. ;
% Limited to a single image-template pair and a fixed nw. ;
%%%%%%%%;
gamma_z_ = transpose(linspace(0,2*pi,n_gamma_z+1)); gamma_z_ = gamma_z_(1:n_gamma_z);
nS = max(0,min(n_S-1,round(n_S*2/5)));
azimu_b_S = viewing_azimu_b_S_(1+nS);
polar_a_S = viewing_polar_a_S_(1+nS);
gamma_z_S = 0.0;
R_S__ = Rz(-gamma_z_S)*Ry(-polar_a_S)*Rz(-azimu_b_S);
nM = max(0,min(n_M-1,round(n_M*4/5)));
index_nS_from_nM = index_nS_from_nM_(1+nM);
azimu_b_M = euler_azimu_b_true_M_(1+nM);
polar_a_M = euler_polar_a_true_M_(1+nM);
gamma_z_M = euler_gamma_z_true_M_(1+nM);
index_nd_from_nM = index_nd_from_nM_(1+nM);
delta_x_M = image_delta_x_true_M_(1+nM);
delta_y_M = image_delta_y_true_M_(1+nM);
R_M__ = Rz(+gamma_z_M)*Ry(-polar_a_M)*Rz(-azimu_b_M);
nw = max(0,min(n_w_max-1,round(n_w_max*3/5)));
Z_d_ampm_ = Z_dwSM_ampm____(:,1+nw,1+nS,1+nM);
Z_d_quad_ = zeros(n_delta_v,1);
Z_d_form_ = zeros(n_delta_v,1);
S_k_p_wk_ = S_k_p_wkS__(:,1+nS);
M_k_p_wk_ = M_k_p_wkM__(:,1+nM);
nCTF = index_nCTF_from_nM_(1+nM);
CTF_k_p_wk_ = CTF_k_p_wkC__(:,1+nCTF);
CTF_phi = CTF_phi_C_(1+nCTF);
%%%%;
gamma_z = (2*pi*nw)/n_w_max;
RS_k_p_wk_ = rotate_p_to_p_fftw(n_k_p_r,n_w_,n_w_sum,S_k_p_wk_,+gamma_z);
CTF_RS_k_p_wk_ = RS_k_p_wk_.*CTF_k_p_wk_;
CTF_RS_k_p_l2 = sum(conj(CTF_RS_k_p_wk_).*CTF_RS_k_p_wk_.*weight_2d_wk_)*(2*pi)^2;
T0M_k_p_wk_ = transf_p_to_p(n_k_p_r,k_p_r_,n_w_,n_w_sum,M_k_p_wk_,+0.0,+0.0);
T0M_k_p_l2 = sum(conj(T0M_k_p_wk_).*T0M_k_p_wk_.*weight_2d_wk_)*(2*pi)^2;
%%;
for ndelta_v=0:n_delta_v-1;
%%;
delta_x = FTK.delta_x_(1+ndelta_v); delta_y = FTK.delta_y_(1+ndelta_v);
TM_k_p_wk_ = transf_p_to_p(n_k_p_r,k_p_r_,n_w_,n_w_sum,M_k_p_wk_,+delta_x,+delta_y);
CTF_RS_k_p_TM_k_p = sum(conj(CTF_RS_k_p_wk_).*TM_k_p_wk_.*weight_2d_wk_)*(2*pi)^2;
Z_d_quad_(1+ndelta_v) = CTF_RS_k_p_TM_k_p;
%%;
%%%%;
Z_d_form = 0;
for nsource_S=0:n_source-1;
delta_S_3_ = R_S__*delta_a_c_3s__(:,1+nsource_S);
delta_S_2_ = R2(+gamma_z) * delta_S_3_(1:2);
for nsource_M=0:n_source-1;
delta_M_3_ = R_M__*delta_a_c_3s__(:,1+nsource_M);
delta_M_2_ = delta_M_3_(1:2) - [delta_x;delta_y] + [delta_x_M;delta_y_M];
tmp_I = I_xPPx_0(k_p_r_max,CTF_phi,delta_S_2_,CTF_phi,delta_M_2_);
Z_d_form = Z_d_form + tmp_I;
end;%for nsource_M=0:n_source-1;
end;%for nsource_S=0:n_source-1;
Z_d_form_(1+ndelta_v) = Z_d_form;
%%;
end;%for ndelta_v=0:n_delta_v-1;
%%;
fnorm_disp(flag_verbose,'Z_d_quad_',Z_d_quad_,'Z_d_ampm_',Z_d_ampm_,' %%<-- should be <1e-2');
fnorm_disp(flag_verbose,'Z_d_form_',Z_d_form_,'Z_d_ampm_',Z_d_ampm_,' %%<-- should be <1e-2');
fnorm_disp(flag_verbose,'Z_d_form_',Z_d_form_,'Z_d_quad_',Z_d_quad_,' %%<-- should be <1e-2');
%%%%%%%%;
if (flag_disp>0);
figure(1+nf);nf=nf+1;clf;figmed;
p_row = 2; p_col = 3; np=0;
Zlim_ = prctile(real(Z_d_form_),[ 5,95],'all');
subplot_t(p_row,p_col,1+np);np=np+1;cla; imagesc_d_FTK([],FTK.delta_x_,FTK.delta_y_,real(Z_d_form_),Zlim_); axisnotick; axis image; title('real(Z_d_form_)','Interpreter','none');
subplot_t(p_row,p_col,1+np);np=np+1;cla; imagesc_d_FTK([],FTK.delta_x_,FTK.delta_y_,imag(Z_d_form_),Zlim_); axisnotick; axis image; title('imag(Z_d_form_)','Interpreter','none');
subplot_t(p_row,p_col,1+np);np=np+1;cla; imagesc_d_FTK([],FTK.delta_x_,FTK.delta_y_,real(Z_d_quad_),Zlim_); axisnotick; axis image; title('real(Z_d_quad_)','Interpreter','none');
subplot_t(p_row,p_col,1+np);np=np+1;cla; imagesc_d_FTK([],FTK.delta_x_,FTK.delta_y_,imag(Z_d_quad_),Zlim_); axisnotick; axis image; title('imag(Z_d_quad_)','Interpreter','none');
subplot_t(p_row,p_col,1+np);np=np+1;cla; imagesc_d_FTK([],FTK.delta_x_,FTK.delta_y_,real(Z_d_ampm_),Zlim_); axisnotick; axis image; title('real(Z_d_ampm_)','Interpreter','none');
subplot_t(p_row,p_col,1+np);np=np+1;cla; imagesc_d_FTK([],FTK.delta_x_,FTK.delta_y_,imag(Z_d_ampm_),Zlim_); axisnotick; axis image; title('imag(Z_d_ampm_)','Interpreter','none');
fname_fig_pre = sprintf('%s/test_transforms_Z_d_form_FIGA',dir_jpg);
fname_fig_jpg = sprintf('%s.jpg',fname_fig_pre);
if flag_replot | ~exist(fname_fig_jpg,'file');
disp(sprintf(' %% writing %s',fname_fig_pre));
print('-djpeg',fname_fig_jpg);
end;%if flag_replot | ~exist(fname_fig_jpg,'file');
%close(gcf);
end;%if (flag_disp>0);
%%%%%%%%;
'''

#%%%%%%%%;
gamma_z_ = torch.linspace(0,2*pi,n_gamma_z+1).to(dtype=torch.float32); gamma_z_ = gamma_z_[0:n_gamma_z].ravel();
nS = int(np.maximum(0,np.minimum(n_S-1,matlab_scalar_round(n_S*2/5))));
azimu_b_S = viewing_azimu_b_S_[nS].item();
polar_a_S = viewing_polar_a_S_[nS].item();
gamma_z_S = 0.0;
R_S__ = mmmm( Rz(-gamma_z_S) , mmmm( Ry(-polar_a_S) , Rz(-azimu_b_S) ) );
nM = int(np.maximum(0,np.minimum(n_M-1,matlab_scalar_round(n_M*4/5))));
index_nS_from_nM = int(index_nS_from_nM_[nM].item());
azimu_b_M = euler_azimu_b_true_M_[nM].item();
polar_a_M = euler_polar_a_true_M_[nM].item();
gamma_z_M = euler_gamma_z_true_M_[nM].item();
index_nd_from_nM = int(index_nd_from_nM_[nM].item());
delta_x_M = image_delta_x_true_M_[nM].item();
delta_y_M = image_delta_y_true_M_[nM].item();
R_M__ = mmmm( Rz(+gamma_z_M) , mmmm( Ry(-polar_a_M) , Rz(-azimu_b_M) ) );
nw = int(np.maximum(0,np.minimum(n_w_max-1,matlab_scalar_round(n_w_max*3/5))));
tmp_i8_index_rhs_ = matlab_index_4d_0(FTK['n_delta_v'],':',n_w_max,nw,n_S,nS,n_M,nM);
Z_d_ampm_ = Z_dwSM_ampm____.ravel()[tmp_i8_index_rhs_];
Z_d_quad_ = torch.zeros(n_delta_v).to(dtype=torch.complex64);
Z_d_form_ = torch.zeros(n_delta_v).to(dtype=torch.complex64);
S_k_p_wk_ = S_k_p_wkS__[nS,:].ravel();
M_k_p_wk_ = M_k_p_wkM__[nM,:].ravel();
nCTF = int(index_nCTF_from_nM_[nM].item());
CTF_k_p_wk_ = CTF_k_p_wkC__[nCTF,:].ravel();
CTF_phi = CTF_phi_C_[nCTF].item();
#%%%%;
gamma_z = (2*pi*nw)/np.maximum(1,n_w_max);
RS_k_p_wk_ = rotate_p_to_p_fftw(n_k_p_r,n_w_,n_w_sum,S_k_p_wk_,+gamma_z);
CTF_RS_k_p_wk_ = RS_k_p_wk_.ravel()*CTF_k_p_wk_.ravel();
CTF_RS_k_p_l2 = torch.sum(torch.conj(CTF_RS_k_p_wk_.ravel())*CTF_RS_k_p_wk_.ravel()*weight_2d_wk_.ravel()).item()*(2*pi)**2;
T0M_k_p_wk_ = transf_p_to_p(n_k_p_r,k_p_r_,n_w_,n_w_sum,M_k_p_wk_,+0.0,+0.0);
T0M_k_p_l2 = torch.sum(torch.conj(T0M_k_p_wk_.ravel())*T0M_k_p_wk_.ravel()*weight_2d_wk_.ravel()).item()*(2*pi)**2;
#%%;
for ndelta_v in range(n_delta_v):
    #%%;
    delta_x = FTK['r8_delta_x_'][ndelta_v].item(); delta_y = FTK['r8_delta_y_'][ndelta_v].item();
    TM_k_p_wk_ = transf_p_to_p(n_k_p_r,k_p_r_,n_w_,n_w_sum,M_k_p_wk_,+delta_x,+delta_y);
    CTF_RS_k_p_TM_k_p = torch.sum(torch.conj(CTF_RS_k_p_wk_.ravel())*TM_k_p_wk_.ravel()*weight_2d_wk_.ravel()).item()*(2*pi)**2;
    Z_d_quad_[ndelta_v] = CTF_RS_k_p_TM_k_p;
    #%%;
    #%%%%;
    Z_d_form = 0;
    for nsource_S in range(n_source):
        delta_S_3_ = mmvm( R_S__ , delta_a_c_3s__[nsource_S,:].ravel() );
        delta_S_2_ = mmvm( R2(+gamma_z) , delta_S_3_[0:2].ravel() );
        for nsource_M in range(n_source):
            delta_M_3_ = mmvm( R_M__ , delta_a_c_3s__[nsource_M,:].ravel() );
            delta_M_2_ = delta_M_3_[0:2].ravel() - torch.tensor([delta_x,delta_y]).to(dtype=torch.float32) + torch.tensor([delta_x_M,delta_y_M]).to(dtype=torch.float32);
            tmp_I = I_xPPx_0(k_p_r_max,CTF_phi,delta_S_2_,CTF_phi,delta_M_2_);
            Z_d_form = Z_d_form + tmp_I;
        #end;%for nsource_M=0:n_source-1;
    #end;%for nsource_S=0:n_source-1;
    Z_d_form_[ndelta_v] = Z_d_form;
    #%%;
#end;%for ndelta_v=0:n_delta_v-1;
#%%;
fnorm_disp(flag_verbose,'Z_d_quad_',Z_d_quad_,'Z_d_ampm_',Z_d_ampm_,' %%<-- should be <1e-2');
fnorm_disp(flag_verbose,'Z_d_form_',Z_d_form_,'Z_d_ampm_',Z_d_ampm_,' %%<-- should be <1e-2');
fnorm_disp(flag_verbose,'Z_d_form_',Z_d_form_,'Z_d_quad_',Z_d_quad_,' %%<-- should be <1e-2');
#%%%%%%%%;

dir_base = '/data/rangan' ;
dir_ascii = dir_base + '/dir_cryoem/dir_rangan_python/dir_ascii' ;
fname_ascii = dir_ascii + '/Z_d_quad_.ascii' ;
print(f" %% writing {fname_ascii}");
np.savetxt(fname_ascii,Z_d_quad_.numpy().ravel());
fname_ascii = dir_ascii + '/Z_d_form_.ascii' ;
print(f" %% writing {fname_ascii}");
np.savetxt(fname_ascii,Z_d_form_.numpy().ravel());
fname_ascii = dir_ascii + '/Z_d_ampm_.ascii' ;
print(f" %% writing {fname_ascii}");
np.savetxt(fname_ascii,Z_d_ampm_.numpy().ravel());

