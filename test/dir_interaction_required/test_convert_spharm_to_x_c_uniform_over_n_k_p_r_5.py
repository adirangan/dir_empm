from dir_empm.dir_matlab_macros import * ;
from dir_empm.cg_rhs_2 import cg_rhs_2 ;
from dir_empm.sample_sphere_7 import sample_sphere_7 ;
from dir_empm.xxnufft3d3 import xxnufft3d3 ;
from dir_empm.plane_wave_expansion_1 import plane_wave_expansion_1 ;
from dir_empm.convert_k_p_to_spharm_4 import convert_k_p_to_spharm_4 ;
from dir_empm.convert_k_p_to_spharm_uniform_over_n_k_p_r_5 import convert_k_p_to_spharm_uniform_over_n_k_p_r_5 ;
from dir_empm.convert_spharm_to_k_p_4 import convert_spharm_to_k_p_4 ;
from dir_empm.convert_spharm_to_k_p_uniform_over_n_k_p_r_5 import convert_spharm_to_k_p_uniform_over_n_k_p_r_5 ;
from dir_empm.convert_spharm_to_x_c_uniform_over_n_k_p_r_5 import convert_spharm_to_x_c_uniform_over_n_k_p_r_5 ;
from dir_empm.h3d import h3d ;

str_thisfunction = 'test_convert_spharm_to_x_c_uniform_over_n_k_p_r_5.py' ;
flag_verbose = 1 ; # verbosity level.

k_int = 16  # highest frequency (2*pi*k_p_r_max), watch out for aliasing!
k_eq_d_double = 1.0  # prefactor for k_eq_d, determines density of sampling in frequency-space.
template_k_eq_d_double = 0.5  # prefactor for template_k_eq_d, determines density of viewing-angles on the sphere.
n_w_int = 1.0  # prefactor for n_w_max, determines the number of distinct angles (i.e., n_gamma_z) used in frequency-space 2d-polar-grid.
n_x_c = max(64, 2 * k_int)  # the number of 'pixels' on a side in the real-space cartesian-grid.

#%%%%%%%%;
# Define spatial grid (centered).
half_diameter_x_c = 1.0;
diameter_x_c = 2.0 * half_diameter_x_c;
x_p_r_max = half_diameter_x_c;
x_c_0_ = torch.linspace(-x_p_r_max, +x_p_r_max, n_x_c).to(torch.float32); d_x_c_0 = x_c_0_[1].item()-x_c_0_[0].item();
x_c_1_ = torch.linspace(-x_p_r_max, +x_p_r_max, n_x_c).to(torch.float32); d_x_c_1 = x_c_1_[1].item()-x_c_1_[0].item();
x_c_2_ = torch.linspace(-x_p_r_max, +x_p_r_max, n_x_c).to(torch.float32); d_x_c_2 = x_c_2_[1].item()-x_c_2_[0].item();
x_c_2___, x_c_1___, x_c_0___ = torch.meshgrid(x_c_2_, x_c_1_, x_c_0_, indexing='ij') ; n_xxx_c = n_x_c ** 3 ; #<-- reversed to match matlab. ;
weight_xxx_c = d_x_c_0 * d_x_c_1 * d_x_c_2 ;
#%%%%%%%%;
# Define spatial grid (uncentered).
n_x_u = n_x_c;
x_u_0_ = torch.linspace(-x_p_r_max, +x_p_r_max, n_x_u+1).to(torch.float32)[:-1]; d_x_u_0 = x_u_0_[1].item()-x_u_0_[0].item();
x_u_1_ = torch.linspace(-x_p_r_max, +x_p_r_max, n_x_u+1).to(torch.float32)[:-1]; d_x_u_1 = x_u_1_[1].item()-x_u_1_[0].item();
x_u_2_ = torch.linspace(-x_p_r_max, +x_p_r_max, n_x_u+1).to(torch.float32)[:-1]; d_x_u_2 = x_u_2_[1].item()-x_u_2_[0].item();
x_u_2___, x_u_1___, x_u_0___ = torch.meshgrid(x_u_2_, x_u_1_, x_u_0_, indexing='ij') ; n_xxx_u = n_x_u ** 3 ; #<-- reversed to match matlab. ;
weight_xxx_u = d_x_u_0 * d_x_u_1 * d_x_u_2 ;
#%%%%%%%%;

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

# Define frequency-space cartesian-grid (centered).
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

# Define frequency-space volumes analytically (on frequency-space cartesian-grid).
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

# Now test k-quadrature on sphere.
# First calculate integrals numerically. ;
I_a_quad = torch.sum(a_k_p_form_ * weight_3d_k_p_qk_).item();
I_b_quad = torch.sum(b_k_p_form_ * weight_3d_k_p_qk_).item();
# Now calculate integrals analytically. ;
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
# Now compare results. ;
fnorm_disp(flag_verbose,'I_a_form',I_a_form,'I_a_quad',I_a_quad,' %%<-- should be <1e-6');
fnorm_disp(flag_verbose,'I_b_form',I_b_form,'I_b_quad',I_b_quad,' %%<-- should be <1e-6');

# Calculate l2-norm in frequency-space, and compare with formula. ;
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
# Using the k-quadrature on the sphere we can also determine the real-space function a_x_u_form___ analytically.
a_x_u_form___ = torch.zeros((n_x_u, n_x_u, n_x_u)).to(dtype=torch.float32);
for nsource in range(n_source):
    delta_a_c_ = delta_a_c_3s__[nsource,:];
    tmp_kd___ = 2*pi*k_p_r_max*torch.sqrt( + (delta_a_c_[0].item() + x_u_0___)**2 + (delta_a_c_[1].item() + x_u_1___)**2 + (delta_a_c_[2].item() + x_u_2___)**2 ) ;
    tmp_h3d___ = h3d(tmp_kd___) ;
    tmp_h3d___[torch.abs(tmp_kd___)<=1e-12] = 4*pi/3 ;
    a_x_u_form___ += tmp_h3d___*k_p_r_max**3 ;
#end;%for nsource in range(n_source);
a_x_u_l2_quad = torch.sum(torch.abs(a_x_u_form___) ** 2 , (0,1,2) ).item() * weight_xxx_u;
print(f" %% Note l2-loss: a_x_u_l2_quad {a_x_u_l2_quad:+0.6f} vs a_k_p_l2_form {a_k_p_l2_form:+0.6f}");

# Note that, due to the l2-loss,
# we can not expect to reconstruct a_k_p_form_ from a_x_c_l2_quad_ on the limited real-space cartesian-grid.
# (i.e., the high-frequency-components will be lost).

# Here we calculate the reconstruction loss going from a_k_p_form_ back to a_x_c_quad___. ;
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
# Here we calculate the reconstruction loss going from a_x_c_form___ back to a_k_p_quad_. ;
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

# Here we calculate the reconstruction loss going from b_k_p_form_ back to b_x_c_quad___. ;
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
# Here we calculate the reconstruction loss going from b_x_c_form___ back to b_k_p_quad_. ;
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

# Now set up spherical-harmonics. ;
l_max_upb = matlab_scalar_round(2 * pi * k_p_r_max) ;
l_max_max = min(l_max_upb, 1 + int(np.ceil(2 * pi * k_p_r_[-1].item()))) ;
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

# Here we determine the spherical-harmonic representation a_k_Y_form_ analytically. ;
a_k_Y_form_ = torch.zeros(n_y_sum).to(dtype=torch.complex64);
for nsource in range(n_source):
    a_k_Y_form_ = a_k_Y_form_ + plane_wave_expansion_1(n_k_p_r,k_p_r_,delta_a_c_3s__[nsource,:],l_max_);
#end;%for nsource=0:n_source-1;
# Now we use our quadrature to generate a_k_Y_quad_ from the analytical form of a_k_p_form_. ;
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
# And we compare the results. ;
fnorm_disp(flag_verbose,'a_k_Y_form_',a_k_Y_form_,'a_k_Y_quad_',a_k_Y_quad_,' %%<-- should be <1e-2');
# We do the same for the reverse transformation. ;
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
# And we test the reconstruction error (i.e., for a round-trip back to a_k_p_). ;
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
# And we test the reconstruction error (i.e., for a round-trip back to a_k_Y_). ;
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

# Now we compare the results of convert_k_p_to_spharm_4 with convert_k_p_to_spharm_uniform_over_n_k_p_r_5. ;
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

# Now we compare the results of convert_spharm_to_k_p_4 with convert_spharm_to_k_p_uniform_over_n_k_p_r_5. ;
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

# Now we finally test out convert_spharm_to_x_c_uniform_over_n_k_p_r_5 for a centered grid. ;
tmp_t = tic();
flag_u_vs_c = 0; #<-- This implies a centered grid for x_c_. ;
if 'sqrt_2lp1_' not in locals(): sqrt_2lp1_ = None; #end;
if 'sqrt_2mp1_' not in locals(): sqrt_2mp1_ = None; #end;
if 'sqrt_rat0_m_' not in locals(): sqrt_rat0_m_ = None; #end;
if 'sqrt_rat3_lm__' not in locals(): sqrt_rat3_lm__ = None; #end;
if 'sqrt_rat4_lm__' not in locals(): sqrt_rat4_lm__ = None; #end;
(
    a_x_c_quad_6_,
    a_k_p_quad_6_,
    sqrt_2lp1_,
    sqrt_2mp1_,
    sqrt_rat0_m_,
    sqrt_rat3_lm__,
    sqrt_rat4_lm__,
) = convert_spharm_to_x_c_uniform_over_n_k_p_r_5(
    0*flag_verbose,
    k_eq_d,
    n_k_p_r,
    k_p_r_,
    k_p_r_max,
    weight_3d_k_p_r_,
    l_max_,
    a_k_Y_form_,
    half_diameter_x_c,
    n_x_c,
    sqrt_2lp1_,
    sqrt_2mp1_,
    sqrt_rat0_m_,
    sqrt_rat3_lm__,
    sqrt_rat4_lm__,
    flag_u_vs_c,
);
tmp_t = toc(tmp_t); disp(sprintf(' %% a_k_p_quad_6_ time %0.2fs',tmp_t));
#%%%%%%%%;
fnorm_disp(flag_verbose,'a_x_c_form___.ravel()',a_x_c_form___.ravel(),'a_x_c_quad_6_',a_x_c_quad_6_,' %%<-- should be small');
fnorm_disp(flag_verbose,'a_k_p_quad_4_',a_k_p_quad_4_,'a_k_p_quad_6_',a_k_p_quad_6_,' %%<-- should be zero');
#%%%%%%%%;

# Now we finally test out convert_spharm_to_x_c_uniform_over_n_k_p_r_5 for an uncentered grid. ;
tmp_t = tic();
flag_u_vs_c = 1; #<-- This implies an uncentered grid for x_u_. ;
if 'sqrt_2lp1_' not in locals(): sqrt_2lp1_ = None; #end;
if 'sqrt_2mp1_' not in locals(): sqrt_2mp1_ = None; #end;
if 'sqrt_rat0_m_' not in locals(): sqrt_rat0_m_ = None; #end;
if 'sqrt_rat3_lm__' not in locals(): sqrt_rat3_lm__ = None; #end;
if 'sqrt_rat4_lm__' not in locals(): sqrt_rat4_lm__ = None; #end;
(
    a_x_u_quad_7_,
    a_k_p_quad_7_,
    sqrt_2lp1_,
    sqrt_2mp1_,
    sqrt_rat0_m_,
    sqrt_rat3_lm__,
    sqrt_rat4_lm__,
) = convert_spharm_to_x_c_uniform_over_n_k_p_r_5(
    0*flag_verbose,
    k_eq_d,
    n_k_p_r,
    k_p_r_,
    k_p_r_max,
    weight_3d_k_p_r_,
    l_max_,
    a_k_Y_form_,
    half_diameter_x_c,
    n_x_u,
    sqrt_2lp1_,
    sqrt_2mp1_,
    sqrt_rat0_m_,
    sqrt_rat3_lm__,
    sqrt_rat4_lm__,
    flag_u_vs_c,
);
tmp_t = toc(tmp_t); disp(sprintf(' %% a_k_p_quad_7_ time %0.2fs',tmp_t));
#%%%%%%%%;
fnorm_disp(flag_verbose,'a_x_u_form___.ravel()',a_x_u_form___.ravel(),'a_x_u_quad_7_',a_x_u_quad_7_,' %%<-- should be small');
fnorm_disp(flag_verbose,'a_k_p_quad_4_',a_k_p_quad_4_,'a_k_p_quad_7_',a_k_p_quad_7_,' %%<-- should be zero');
#%%%%%%%%;

