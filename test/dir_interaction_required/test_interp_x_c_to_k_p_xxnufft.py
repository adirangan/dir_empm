from dir_empm.dir_matlab_macros import * ;
from dir_empm.get_weight_3d_1 import get_weight_3d_1 ;
from dir_empm.get_weight_2d_2 import get_weight_2d_2 ;
from dir_empm.xxnufft2d3 import xxnufft2d3 ;
from dir_empm.interp_k_p_to_x_c_xxnufft import interp_k_p_to_x_c_xxnufft ;
from dir_empm.interp_x_c_to_k_p_xxnufft import interp_x_c_to_k_p_xxnufft ;
from dir_empm.h2d import h2d ;

str_thisfunction = 'interp_x_c_to_k_p_xxnufft';

flag_verbose=1; flag_disp=1; nf=1;
#%%%%%%%%;
#% Define spatial grid. ;
#%%%%%%%%;
half_diameter_x_u = 1.0; x_p_r_max = half_diameter_x_u;
diameter_x_u = 2*half_diameter_x_u;
n_x_M_u = 128+1;
#%%%%%%%%;
#% first set up 'uniform' (i.e., uncentered) cartesian grid. ;
#%%%%%%%%;
x_u_0_ = -half_diameter_x_u + torch.arange(n_x_M_u).to(dtype=torch.float32)/n_x_M_u*diameter_x_u;
x_u_1_ = -half_diameter_x_u + torch.arange(n_x_M_u).to(dtype=torch.float32)/n_x_M_u*diameter_x_u;
x_u_0__,x_u_1__ = torch.meshgrid(x_u_1_,x_u_0_,indexing='ij'); #<-- order reversed to match matlab. ;
dx_u = x_u_0_[1].item()-x_u_0_[0].item();
n_xx_M_u = n_x_M_u**2;
#%%%%%%%%;
#% next set up 'cartesian' (i.e., centered) cartesian grid. ;
#%%%%%%%%;
half_diameter_x_c = 1.0;
diameter_x_c = 2*half_diameter_x_c;
n_x_M_c = 128;
x_c_0_ = torch.linspace(-x_p_r_max,+x_p_r_max,n_x_M_c).to(dtype=torch.float32);
x_c_1_ = torch.linspace(-x_p_r_max,+x_p_r_max,n_x_M_c).to(dtype=torch.float32);
x_c_0__,x_c_1__ = torch.meshgrid(x_c_1_,x_c_0_,indexing='ij'); #<-- order reversed to match matlab. ;
dx_c = x_c_0_[1].item()-x_c_0_[0].item();
n_xx_M_c = n_x_M_c**2;
#%%%%%%%%;
#% Now set up and test k-quadrature on sphere. ;
#%%%%%%%%;
k_int = 1*48; k_eq_d_double = 1.0; n_w_int = 1;
k_p_r_max = k_int/(2*pi); k_eq_d = k_eq_d_double/(2*pi); str_L = 'L';
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
n_w_max = n_w_int*2*(k_int+1);
n_w_0in_ = n_w_max*torch.ones(n_k_p_r).to(dtype=torch.int32);
(
    n_w_,
    weight_2d_k_p_r_,
    weight_2d_wk_,
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
n_w_sum = int(torch.sum(n_w_).item());
n_w_csum_ = cumsum_0(n_w_);
#%%%%%%%%;
n_source = 8;
rng(0);
delta_a_x__ = torch.zeros(mtr((2,n_source))).to(dtype=torch.float32);
delta_b_x__ = torch.zeros(mtr((2,n_source))).to(dtype=torch.float32);
for nsource in range(n_source):
    delta_a_x_ = 0.125*(2*torch.rand(2).to(dtype=torch.float32)-1);
    delta_a_x__[nsource,:] = delta_a_x_;
    delta_b_x_ = 0.125*(2*torch.rand(2).to(dtype=torch.float32)-1);
    delta_b_x__[nsource,:] = delta_b_x_;
#end;%for nsource in range(n_source):
a_k_p_form_ = torch.zeros(n_w_sum).to(dtype=torch.complex64);
b_k_p_form_ = torch.zeros(n_w_sum).to(dtype=torch.complex64);
for nsource in range(n_source):
    delta_a_x_ = delta_a_x__[nsource,:];
    a_k_p_form_ = a_k_p_form_ + torch.exp(+i*2*pi*(k_c_0_wk_*delta_a_x_[0].item() + k_c_1_wk_*delta_a_x_[1].item()));
    delta_b_x_ = delta_b_x__[nsource,:];
    b_k_p_form_ = b_k_p_form_ + torch.exp(+i*2*pi*(k_c_0_wk_*delta_b_x_[0].item() + k_c_1_wk_*delta_b_x_[1].item()));
#end;%for nsource in range(n_source):
I_a_quad = torch.sum(a_k_p_form_*weight_2d_wk_).item()*(2*pi)**2;
I_b_quad = torch.sum(b_k_p_form_*weight_2d_wk_).item()*(2*pi)**2;
I_a_form = 0;
I_b_form = 0;
for nsource in range(n_source):
    delta_a_x_ = delta_a_x__[nsource,:];
    delta_b_x_ = delta_b_x__[nsource,:];
    tmp_kd = 2*pi*k_p_r_max*fnorm(delta_a_x_);
    I_a_form = I_a_form + h2d(tmp_kd).item()*k_p_r_max**2/(4*pi);
    tmp_kd = 2*pi*k_p_r_max*fnorm(delta_b_x_);
    I_b_form = I_b_form + h2d(tmp_kd).item()*k_p_r_max**2/(4*pi);
#end;%for nsource in range(n_source):
fnorm_disp(flag_verbose,'I_a_form',I_a_form,'I_a_quad',I_a_quad,'%<-- should be small');
fnorm_disp(flag_verbose,'I_b_form',I_b_form,'I_b_quad',I_b_quad,'%<-- should be small');
#%%%%%%%%;
#% Now test xxnufft2d3. ;
#%%%%%%%%;
# For some reason I can not figure out how to call xxnufft2d3 (or rather finufft2d3) with only a single output point. ;
# The np.asarray command in xxnufft2d3 collapses a torch.array with a single element into a float, rather than a 1-element-array. ;
#%%%%%%%%;
torch_zeros_ = torch.zeros(2).to(dtype=torch.float32);
I_a_quad_ = xxnufft2d3(n_w_sum,2*pi*k_c_0_wk_,2*pi*k_c_1_wk_,a_k_p_form_*weight_2d_wk_,+1,1e-12,2,torch_zeros_,torch_zeros_)*(2*pi)**2;
I_a_quad = I_a_quad_[0].item(); #<-- so we do this kludge instead. ;
fnorm_disp(flag_verbose,'I_a_form',I_a_form,'I_a_quad',I_a_quad,'%<-- should be small');
I_b_quad_ = xxnufft2d3(n_w_sum,2*pi*k_c_0_wk_,2*pi*k_c_1_wk_,b_k_p_form_*weight_2d_wk_,+1,1e-12,2,torch_zeros_,torch_zeros_)*(2*pi)**2;
I_b_quad = I_b_quad_[0].item(); #<-- so we do this kludge instead. ;
fnorm_disp(flag_verbose,'I_b_form',I_b_form,'I_b_quad',I_b_quad,'%<-- should be small');
#%%%%%%%%;
a_x_c_form_ = torch.zeros(mtr((n_x_M_c,n_x_M_c))).to(dtype=torch.complex64);
b_x_c_form_ = torch.zeros(mtr((n_x_M_c,n_x_M_c))).to(dtype=torch.complex64);
for nsource in range(n_source):
    delta_a_x_ = delta_a_x__[nsource,:];
    delta_b_x_ = delta_b_x__[nsource,:];
    tmp_kd__ = 2*pi * k_p_r_max * torch.sqrt( (x_c_0__ + delta_a_x_[0].item())**2 + (x_c_1__ + delta_a_x_[1].item())**2 ) ;
    a_x_c_form_ = a_x_c_form_ + h2d(tmp_kd__)*k_p_r_max**2/(4*pi);
    tmp_kd__ = 2*pi * k_p_r_max * torch.sqrt( (x_c_0__ + delta_b_x_[0].item())**2 + (x_c_1__ + delta_b_x_[1].item())**2 ) ;
    b_x_c_form_ = b_x_c_form_ + h2d(tmp_kd__)*k_p_r_max**2/(4*pi);
#end;%for nsource in range(n_source):
#%%%%%%%%;
a_x_c_quad_ = torch.reshape(xxnufft2d3(n_w_sum,2*pi*k_c_0_wk_,2*pi*k_c_1_wk_,a_k_p_form_*weight_2d_wk_,+1,1e-12,n_xx_M_c,x_c_0__.ravel(),x_c_1__.ravel())*(2*pi)**2,mtr((n_x_M_c,n_x_M_c)));
fnorm_disp(flag_verbose,'a_x_c_form_',a_x_c_form_,'a_x_c_quad_',a_x_c_quad_,'%<-- should be small for |x|<=0.1 or so');
b_x_c_quad_ = torch.reshape(xxnufft2d3(n_w_sum,2*pi*k_c_0_wk_,2*pi*k_c_1_wk_,b_k_p_form_*weight_2d_wk_,+1,1e-12,n_xx_M_c,x_c_0__.ravel(),x_c_1__.ravel())*(2*pi)**2,mtr((n_x_M_c,n_x_M_c)));
fnorm_disp(flag_verbose,'b_x_c_form_',b_x_c_form_,'b_x_c_quad_',b_x_c_quad_,'%<-- should be small for |x|<=0.1 or so');
#%%%%%%%%;
a_x_u_form_ = torch.zeros(mtr((n_x_M_u,n_x_M_u))).to(dtype=torch.complex64);
b_x_u_form_ = torch.zeros(mtr((n_x_M_u,n_x_M_u))).to(dtype=torch.complex64);
for nsource in range(n_source):
    delta_a_x_ = delta_a_x__[nsource,:];
    delta_b_x_ = delta_b_x__[nsource,:];
    tmp_kd__ = 2*pi * k_p_r_max * torch.sqrt( (x_u_0__ + delta_a_x_[0].item())**2 + (x_u_1__ + delta_a_x_[1].item())**2 ) ;
    a_x_u_form_ = a_x_u_form_ + h2d(tmp_kd__)*k_p_r_max**2/(4*pi);
    tmp_kd__ = 2*pi * k_p_r_max * torch.sqrt( (x_u_0__ + delta_b_x_[0].item())**2 + (x_u_1__ + delta_b_x_[1].item())**2 ) ;
    b_x_u_form_ = b_x_u_form_ + h2d(tmp_kd__)*k_p_r_max**2/(4*pi);
#end;%for nsource in range(n_source):
#%%%%%%%%;
a_x_u_quad_ = torch.reshape(xxnufft2d3(n_w_sum,2*pi*k_c_0_wk_,2*pi*k_c_1_wk_,a_k_p_form_*weight_2d_wk_,+1,1e-12,n_xx_M_u,x_u_0__.ravel(),x_u_1__.ravel())*(2*pi)**2,mtr((n_x_M_u,n_x_M_u)));
fnorm_disp(flag_verbose,'a_x_u_form_',a_x_u_form_,'a_x_u_quad_',a_x_u_quad_,'%<-- should be small for |x|<=0.1 or so');
b_x_u_quad_ = torch.reshape(xxnufft2d3(n_w_sum,2*pi*k_c_0_wk_,2*pi*k_c_1_wk_,b_k_p_form_*weight_2d_wk_,+1,1e-12,n_xx_M_u,x_u_0__.ravel(),x_u_1__.ravel())*(2*pi)**2,mtr((n_x_M_u,n_x_M_u)));
fnorm_disp(flag_verbose,'b_x_u_form_',b_x_u_form_,'b_x_u_quad_',b_x_u_quad_,'%<-- should be small for |x|<=0.1 or so');
#%%%%%%%%;

#%%%%%%%%;
flag_u_vs_c=0;
a_k_p_form_l2 = torch.sum(torch.abs(a_k_p_form_.ravel())**2*weight_2d_wk_).item()*(2*pi)**2;
a_x_c_quad_ = interp_k_p_to_x_c_xxnufft(n_x_M_c,diameter_x_c,n_x_M_c,diameter_x_c,n_k_p_r,k_p_r_,n_w_,a_k_p_form_*weight_2d_wk_*(2*pi)**2,flag_u_vs_c)*np.sqrt(n_x_M_c**2) * n_w_sum;
a_x_c_quad_l2 = torch.sum(torch.abs(a_x_c_quad_.ravel())**2).item()*dx_c**2;
b_k_p_form_l2 = torch.sum(torch.abs(b_k_p_form_.ravel())**2*weight_2d_wk_).item()*(2*pi)**2;
b_x_c_quad_ = interp_k_p_to_x_c_xxnufft(n_x_M_c,diameter_x_c,n_x_M_c,diameter_x_c,n_k_p_r,k_p_r_,n_w_,b_k_p_form_*weight_2d_wk_*(2*pi)**2,flag_u_vs_c)*np.sqrt(n_x_M_c**2) * n_w_sum;
b_x_c_quad_l2 = torch.sum(torch.abs(b_x_c_quad_.ravel())**2).item()*dx_c**2;
fnorm_disp(flag_verbose,'a_k_p_form_l2',a_k_p_form_l2,'a_x_c_quad_l2',a_x_c_quad_l2,'%<-- should be moderate due to restricted integral');
fnorm_disp(flag_verbose,'b_k_p_form_l2',b_k_p_form_l2,'b_x_c_quad_l2',b_x_c_quad_l2,'%<-- should be moderate due to restricted integral');
#%%%%%%%%;
fnorm_disp(flag_verbose,'a_x_c_form_',a_x_c_form_,'a_x_c_quad_',a_x_c_quad_,'%<-- should be small for |x|<=0.1 or so');
fnorm_disp(flag_verbose,'b_x_c_form_',b_x_c_form_,'b_x_c_quad_',b_x_c_quad_,'%<-- should be small for |x|<=0.1 or so');
a_x_c_form_l2 = torch.sum(torch.abs(a_x_c_form_.ravel())**2).item()*dx_c**2;
b_x_c_form_l2 = torch.sum(torch.abs(b_x_c_form_.ravel())**2).item()*dx_c**2;
a_x_c_quad_l2 = torch.sum(torch.abs(a_x_c_quad_.ravel())**2).item()*dx_c**2;
b_x_c_quad_l2 = torch.sum(torch.abs(b_x_c_quad_.ravel())**2).item()*dx_c**2;
fnorm_disp(flag_verbose,'a_x_c_form_l2',a_x_c_form_l2,'a_x_c_quad_l2',a_x_c_quad_l2,'%<-- should be small');
fnorm_disp(flag_verbose,'b_x_c_form_l2',b_x_c_form_l2,'b_x_c_quad_l2',b_x_c_quad_l2,'%<-- should be small');
#%%%%%%%%;
flag_u_vs_c=1;
a_k_p_form_l2 = torch.sum(torch.abs(a_k_p_form_.ravel())**2*weight_2d_wk_).item()*(2*pi)**2;
a_x_u_quad_ = interp_k_p_to_x_c_xxnufft(n_x_M_u,diameter_x_u,n_x_M_u,diameter_x_u,n_k_p_r,k_p_r_,n_w_,a_k_p_form_*weight_2d_wk_*(2*pi)**2,flag_u_vs_c)*np.sqrt(n_x_M_u**2) * n_w_sum;
a_x_u_quad_l2 = torch.sum(torch.abs(a_x_u_quad_.ravel())**2).item()*dx_u**2;
b_k_p_form_l2 = torch.sum(torch.abs(b_k_p_form_.ravel())**2*weight_2d_wk_).item()*(2*pi)**2;
b_x_u_quad_ = interp_k_p_to_x_c_xxnufft(n_x_M_u,diameter_x_u,n_x_M_u,diameter_x_u,n_k_p_r,k_p_r_,n_w_,b_k_p_form_*weight_2d_wk_*(2*pi)**2,flag_u_vs_c)*np.sqrt(n_x_M_u**2) * n_w_sum;
b_x_u_quad_l2 = torch.sum(torch.abs(b_x_u_quad_.ravel())**2).item()*dx_u**2;
fnorm_disp(flag_verbose,'a_k_p_form_l2',a_k_p_form_l2,'a_x_u_quad_l2',a_x_u_quad_l2,'%<-- should be moderate due to restricted integral');
fnorm_disp(flag_verbose,'b_k_p_form_l2',b_k_p_form_l2,'b_x_u_quad_l2',b_x_u_quad_l2,'%<-- should be moderate due to restricted integral');
#%%%%%%%%;
fnorm_disp(flag_verbose,'a_x_u_form_',a_x_u_form_,'a_x_u_quad_',a_x_u_quad_,'%<-- should be small for |x|<=0.1 or so');
fnorm_disp(flag_verbose,'b_x_u_form_',b_x_u_form_,'b_x_u_quad_',b_x_u_quad_,'%<-- should be small for |x|<=0.1 or so');
a_x_u_form_l2 = torch.sum(torch.abs(a_x_u_form_.ravel())**2).item()*dx_u**2;
b_x_u_form_l2 = torch.sum(torch.abs(b_x_u_form_.ravel())**2).item()*dx_u**2;
a_x_u_quad_l2 = torch.sum(torch.abs(a_x_u_quad_.ravel())**2).item()*dx_u**2;
b_x_u_quad_l2 = torch.sum(torch.abs(b_x_u_quad_.ravel())**2).item()*dx_u**2;
fnorm_disp(flag_verbose,'a_x_u_form_l2',a_x_u_form_l2,'a_x_u_quad_l2',a_x_u_quad_l2,'%<-- should be small');
fnorm_disp(flag_verbose,'b_x_u_form_l2',b_x_u_form_l2,'b_x_u_quad_l2',b_x_u_quad_l2,'%<-- should be small');

#%%%%%%%%;
#% Now test gaussian on centered grid. ;
#%%%%%%%%;
tmp_sigma_x_c = 0.0625;
tmp_sigma_k_p = 1/tmp_sigma_x_c;
tmp_delta_ = torch.zeros(2).to(dtype=torch.float32); tmp_delta_[0] = 0.75*(+0.1); tmp_delta_[1] = 0.75*(-0.2);
tmp_M_x_c_form_ = 1/(np.sqrt(2*pi)*tmp_sigma_x_c)**2 * torch.exp( -( (x_c_0__-tmp_delta_[0].item())**2 + (x_c_1__-tmp_delta_[1].item())**2 ) / (2*tmp_sigma_x_c**2) );
tmp_M_x_c_form_l2 = torch.sum(tmp_M_x_c_form_.ravel()**2).item()*dx_c**2;
disp(sprintf(' %% sum(tmp_M_x_c_form_*dx_c**2,''all'') = %0.16f',torch.sum(tmp_M_x_c_form_.ravel()*dx_c**2).item()));
disp(sprintf(' %% tmp_M_x_c_form_l2 = %0.16f',tmp_M_x_c_form_l2));
flag_u_vs_c = 0;
tmp_M_k_p_quad_ = interp_x_c_to_k_p_xxnufft(n_x_M_c,diameter_x_c,n_x_M_c,diameter_x_c,tmp_M_x_c_form_,n_k_p_r,k_p_r_,n_w_,flag_u_vs_c)*np.sqrt(n_x_M_c**2)*dx_c**2;
tmp_M_k_p_quad_l2 = torch.sum(torch.abs(tmp_M_k_p_quad_)**2 * weight_2d_wk_).item() * (2*pi)**2;
disp(sprintf(' %% tmp_M_k_p_quad_l2 = %0.16f',tmp_M_k_p_quad_l2));
tmp_M_k_p_form_ = torch.zeros(n_w_sum).to(dtype=torch.complex64);
na=0;
for nk_p_r in range(n_k_p_r):
    k_p_r = k_p_r_[nk_p_r].item();
    n_w = int(n_w_[nk_p_r].item());
    for nw in range(n_w):
        k_x_c_0 = k_p_r*np.cos(2*pi*nw/n_w);
        k_x_c_1 = k_p_r*np.sin(2*pi*nw/n_w);
        tmp_M_k_p_form_[na] = np.exp( -( (2*pi*k_x_c_0)**2 + (2*pi*k_x_c_1)**2 ) / (2/tmp_sigma_x_c**2) ) * np.exp( - 2*pi*i*( k_x_c_0*tmp_delta_[0].item() + k_x_c_1*tmp_delta_[1].item() ) );
        na=na+1;
    #end;%for nw in range(n_w):
#end;%for nk_p_r in range(n_k_p_r):
tmp_M_k_p_form_l2 = torch.sum(torch.abs(tmp_M_k_p_form_)**2 * weight_2d_wk_).item() * (2*pi)**2;
disp(sprintf(' %% tmp_M_k_p_form_l2 = %0.16f',tmp_M_k_p_form_l2));
fnorm_disp(flag_verbose,'tmp_M_k_p_form_',tmp_M_k_p_form_,'tmp_M_k_p_quad_',tmp_M_k_p_quad_,'%<-- should be small');
flag_u_vs_c = 0;
tmp_M_x_c_reco_ = interp_k_p_to_x_c_xxnufft(n_x_M_c,diameter_x_c,n_x_M_c,diameter_x_c,n_k_p_r,k_p_r_,n_w_,tmp_M_k_p_quad_*weight_2d_wk_*(2*pi)**2,flag_u_vs_c)*np.sqrt(n_x_M_c**2) * n_w_sum;
tmp_M_x_c_reco_l2 = torch.sum(torch.abs(tmp_M_x_c_reco_)**2).item()*dx_c**2;
disp(sprintf(' %% tmp_M_x_c_reco_l2 = %0.16f',tmp_M_x_c_reco_l2));
fnorm_disp(flag_verbose,'tmp_M_x_c_form_',tmp_M_x_c_form_,'tmp_M_x_c_reco_',tmp_M_x_c_reco_,'%<-- should be moderate (since cartesian quadrature is not good)');

#%%%%%%%%;
#% Now test gaussian on uncentered grid. ;
#%%%%%%%%;
tmp_sigma_x_u = 0.0625 + 0.0125; #%<-- to differentiate from centered version. ;
tmp_sigma_k_p = 1/tmp_sigma_x_u;
tmp_delta_ = torch.zeros(2).to(dtype=torch.float32); tmp_delta_[0] = 0.75*(+0.15); tmp_delta_[1] = 0.75*(-0.18); #%<-- to differentiate from centered version. ;
tmp_M_x_u_form_ = 1/(np.sqrt(2*pi)*tmp_sigma_x_u)**2 * torch.exp( -( (x_u_0__-tmp_delta_[0].item())**2 + (x_u_1__-tmp_delta_[1].item())**2 ) / (2*tmp_sigma_x_u**2) );
tmp_M_x_u_form_l2 = torch.sum(tmp_M_x_u_form_.ravel()**2).item()*dx_u**2;
disp(sprintf(' %% sum(tmp_M_x_u_form_*dx_u**2,''all'') = %0.16f',torch.sum(tmp_M_x_u_form_.ravel()*dx_u**2).item()));
disp(sprintf(' %% tmp_M_x_u_form_l2 = %0.16f',tmp_M_x_u_form_l2));
flag_u_vs_c = 1;
tmp_M_k_p_quad_ = interp_x_c_to_k_p_xxnufft(n_x_M_u,diameter_x_u,n_x_M_u,diameter_x_u,tmp_M_x_u_form_,n_k_p_r,k_p_r_,n_w_,flag_u_vs_c)*np.sqrt(n_x_M_u**2)*dx_u**2;
tmp_M_k_p_quad_l2 = torch.sum(torch.abs(tmp_M_k_p_quad_)**2 * weight_2d_wk_).item() * (2*pi)**2;
disp(sprintf(' %% tmp_M_k_p_quad_l2 = %0.16f',tmp_M_k_p_quad_l2));
tmp_M_k_p_form_ = torch.zeros(n_w_sum).to(dtype=torch.complex64);
na=0;
for nk_p_r in range(n_k_p_r):
    k_p_r = k_p_r_[nk_p_r].item();
    n_w = int(n_w_[nk_p_r].item());
    for nw in range(n_w):
        k_x_c_0 = k_p_r*np.cos(2*pi*nw/n_w);
        k_x_c_1 = k_p_r*np.sin(2*pi*nw/n_w);
        tmp_M_k_p_form_[na] = np.exp( -( (2*pi*k_x_c_0)**2 + (2*pi*k_x_c_1)**2 ) / (2/tmp_sigma_x_u**2) ) * np.exp( - 2*pi*i*( k_x_c_0*tmp_delta_[0].item() + k_x_c_1*tmp_delta_[1].item() ) );
        na=na+1;
    #end;%for nw in range(n_w):
#end;%for nk_p_r in range(n_k_p_r):
tmp_M_k_p_form_l2 = torch.sum(torch.abs(tmp_M_k_p_form_)**2 * weight_2d_wk_).item() * (2*pi)**2;
disp(sprintf(' %% tmp_M_k_p_form_l2 = %0.16f',tmp_M_k_p_form_l2));
fnorm_disp(flag_verbose,'tmp_M_k_p_form_',tmp_M_k_p_form_,'tmp_M_k_p_quad_',tmp_M_k_p_quad_,'%<-- should be small');
flag_u_vs_c = 1;
tmp_M_x_u_reco_ = interp_k_p_to_x_c_xxnufft(n_x_M_u,diameter_x_u,n_x_M_u,diameter_x_u,n_k_p_r,k_p_r_,n_w_,tmp_M_k_p_quad_*weight_2d_wk_*(2*pi)**2,flag_u_vs_c)*np.sqrt(n_x_M_u**2) * n_w_sum;
tmp_M_x_u_reco_l2 = torch.sum(torch.abs(tmp_M_x_u_reco_)**2).item()*dx_u**2;
disp(sprintf(' %% tmp_M_x_u_reco_l2 = %0.16f',tmp_M_x_u_reco_l2));
fnorm_disp(flag_verbose,'tmp_M_x_u_form_',tmp_M_x_u_form_,'tmp_M_x_u_reco_',tmp_M_x_u_reco_,'%<-- should be moderate (since cartesian quadrature is not good)');

