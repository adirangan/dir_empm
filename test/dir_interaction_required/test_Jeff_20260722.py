from dir_empm.dir_matlab_macros.matlab_macros import * ;
from dir_empm.get_weight_3d_1 import get_weight_3d_1 ;
from dir_empm.get_weight_2d_2 import get_weight_2d_2 ;
from dir_empm.interp_k_p_to_x_c_xxnufft import interp_k_p_to_x_c_xxnufft ;
from dir_empm.interp_x_c_to_k_p_xxnufft import interp_x_c_to_k_p_xxnufft ;
from dir_empm.interp_k_p_to_x_p_xxnufft import interp_k_p_to_x_p_xxnufft ;
from dir_empm.interp_x_p_to_k_p_xxnufft import interp_x_p_to_k_p_xxnufft ;
from dir_empm.xxnufft2d3 import xxnufft2d3 ;
import os ; import matplotlib.pyplot as plt ;

flag_verbose=1; nf=0;
flag_disp = 1; nf=0; flag_replot = 1;
str_thisfunction = 'test_Jeff_20260722';

#%%%%%%%%;
#% Set up spatial-grid for testing. ;
#%%%%%%%%;
half_diameter_x_c = 1.0;
diameter_x_c = 2.0*half_diameter_x_c;
#%%%%%%%%;
n_x_M_u = 128;
x_u_0_ = torch.linspace(-half_diameter_x_c,+half_diameter_x_c,n_x_M_u+1).to(dtype=torch.float32); x_u_0_ = x_u_0_[:-1];
x_u_1_ = torch.linspace(-half_diameter_x_c,+half_diameter_x_c,n_x_M_u+1).to(dtype=torch.float32); x_u_1_ = x_u_1_[:-1];
x_u_1__,x_u_0__ = torch.meshgrid(x_u_1_,x_u_0_,indexing='ij'); #<-- reversed to match matlab. ;
dx_u = x_u_0_[1].item() - x_u_0_[0].item();
n_xx_M_u = n_x_M_u**2;

#%%%%%%%%;
#% Now set up polar grids in real-space and fourier-space. ;
#%%%%%%%%;
x_res = 1;
x_int = 48*x_res;
x_eq_d_double = 0.5;
n_t_int = 1;
#%%%%%%%%;
k_res = 1; #%<-- be careful with the aliasing associated with n_x_M_u. ;
k_int = 48*k_res;
k_eq_d_double = 0.5;
n_w_int = n_t_int;
#%%%%%%%%;
x_p_r_max = half_diameter_x_c;
k_p_r_max = k_int/(2*pi);
x_eq_d = x_eq_d_double/(2*pi)/k_p_r_max * 1.05; #%<-- to check dimensions. ;
k_eq_d = k_eq_d_double/(2*pi)/x_p_r_max;
#%%%%%%%%;

#%%%%%%%%;
(
    n_x_p_r,
    x_p_r_,
    weight_3d_x_p_r_,
) = get_weight_3d_1(
    np.maximum(0,flag_verbose-1),
    x_p_r_max,
    x_eq_d,
)[:3];
fnorm_disp(flag_verbose,'sum(weight_3d_x_p_r_)*4*pi',sum_0(weight_3d_x_p_r_)*4*pi,'volume',4/3*pi*x_p_r_max**3,' %<-- should be small');
#%%%%%%%%;
n_t_max = n_t_int*2*(x_int+1) + 2;  #%<-- to check dimensions. ;
n_t_0in_ = n_t_max*torch.ones(n_x_p_r).to(dtype=torch.int32);
#%%%%%%%%;
(
    n_t_,
    weight_2d_x_p_r_,
    weight_2d_x_p_tx_,
    x_p_r_tx_,
    x_p_t_tx_,
    x_u_0_tx_,
    x_u_1_tx_,
) = get_weight_2d_2(
    np.maximum(0,flag_verbose-1),
    n_x_p_r,
    x_p_r_,
    x_p_r_max,
    -1,
    n_t_0in_,
    weight_3d_x_p_r_,
)[:7];
fnorm_disp(flag_verbose,'sum(weight_2d_x_p_r_)',sum_0(weight_2d_x_p_r_),'area',pi*x_p_r_max**2,' %<-- should be small');
n_t_sum = sum_0(n_t_); n_t_csum_ = cumsum_0(n_t_);
#%%%%%%%%;

#%%%%%%%%;
(
    n_k_p_r,
    k_p_r_,
    weight_3d_k_p_r_,
) = get_weight_3d_1(
    np.maximum(0,flag_verbose-1),
    k_p_r_max,
    k_eq_d,
)[:3];
fnorm_disp(flag_verbose,'sum(weight_3d_k_p_r_)*4*pi',sum_0(weight_3d_k_p_r_)*4*pi,'volume',4/3*pi*k_p_r_max**3,' %<-- should be small');
#%%%%%%%%;
#%n_w_max = 12; 
n_w_max = n_w_int*2*(k_int+1);
n_w_0in_ = n_w_max*torch.ones(n_k_p_r).to(dtype=torch.int32);
#%%%%%%%%;
(
    n_w_,
    weight_2d_k_p_r_,
    weight_2d_k_p_wk_,
    k_p_r_wk_,
    k_p_w_wk_,
    k_c_0_wk_,
    k_c_1_wk_,
) = get_weight_2d_2(
    np.maximum(0,flag_verbose-1),
    n_k_p_r,
    k_p_r_,
    k_p_r_max,
    -1,
    n_w_0in_,
    weight_3d_k_p_r_,
)[:7];
fnorm_disp(flag_verbose,'sum(weight_2d_k_p_r_)',sum_0(weight_2d_k_p_r_),'area',pi*k_p_r_max**2,' %<-- should be small');
n_w_sum = sum_0(n_w_); n_w_csum_ = cumsum_0(n_w_);
#%%%%%%%%;

if (flag_verbose>0):
    disp(sprintf(' %% %% %% %% %% %% %% %%'));
    disp(sprintf(' %% (2*pi)*x_p_r_max: %d',(2*pi)*x_p_r_max));
    disp(sprintf(' %% n_x_p_r: %d',n_x_p_r));
    disp(sprintf(' %% n_t_max: %d',n_t_max));
    disp(sprintf(' %% (2*pi)*k_p_r_max: %d',(2*pi)*k_p_r_max));
    disp(sprintf(' %% n_k_p_r: %d',n_k_p_r));
    disp(sprintf(' %% n_w_max: %d',n_w_max));
    disp(sprintf(' %% %% %% %% %% %% %% %%'));
#end;%if (flag_verbose>0);

#%%%%%%%%;
#% Now test gaussian. ;
#%%%%%%%%;
sigma_x_c = 0.0625;
sigma_k_p = 1/sigma_x_c;
delta_ = 0.75*torch.tensor([+0.1,-0.2]).to(dtype=torch.float32);
M_x_u_form_ = 1/(np.sqrt(2*pi)*sigma_x_c)**2 * torch.exp( -( (x_u_0__-delta_[0])**2 + (x_u_1__-delta_[1])**2 ) / (2*sigma_x_c**2) );
M_x_u_form_l2 = sum_0(M_x_u_form_**2)*dx_u**2;
disp(sprintf(' %% sum(M_x_u_form_*dx_u**2,''all'') = %0.16f',sum_0(M_x_u_form_*dx_u**2)));
disp(sprintf(' %% M_x_u_form_l2 = %0.16f',M_x_u_form_l2));
M_x_p_form_ = 1/(np.sqrt(2*pi)*sigma_x_c)**2 * torch.exp( -( (x_u_0_tx_-delta_[0])**2 + (x_u_1_tx_-delta_[1])**2 ) / (2*sigma_x_c**2) );
M_x_p_form_l2 = sum_0(torch.abs(M_x_p_form_)**2 * weight_2d_x_p_tx_) * (2*pi)**2;
disp(sprintf(' %% M_x_p_form_l2 = %0.16f',M_x_p_form_l2));
M_k_p_quad_ = interp_x_c_to_k_p_xxnufft(n_x_M_u,diameter_x_c,n_x_M_u,diameter_x_c,M_x_u_form_,n_k_p_r,k_p_r_,n_w_)*np.sqrt(n_x_M_u**2)*dx_u**2;
M_k_p_quad_l2 = sum_0(torch.abs(M_k_p_quad_)**2 * weight_2d_k_p_wk_) * (2*pi)**2;
disp(sprintf(' %% M_k_p_quad_l2 = %0.16f',M_k_p_quad_l2));
M_k_p_form_ = torch.exp( -( (2*pi*k_c_0_wk_)**2 + (2*pi*k_c_1_wk_)**2 ) / (2/sigma_x_c**2) ) * torch.exp( - 2*pi*i*( k_c_0_wk_*delta_[0] + k_c_1_wk_*delta_[1] ) );
#%%%%%%%%;
#% Jeff code: ;
#%%%%%%%%;
list_J_k_p_form_ = [];
for nk_p_r in range(n_k_p_r):
    k_p_r = k_p_r_[nk_p_r].item() ;
    n_w = int(n_w_[nk_p_r].item()) ;
    for nw in range(n_w):
        k_x_0 = k_p_r * np.cos(2 * pi * nw/n_w) ;
        k_x_1 = k_p_r * np.sin(2 * pi * nw/n_w) ;
        real_part = -1 * sigma_x_c ** 2 * (
            (2 * pi * k_x_0) ** 2 \
            + (2 * pi * k_x_1) ** 2
        ) / 2 ;
        imag_part = -2j * pi * (k_x_0 * delta_[0].item() + k_x_1 * delta_[1].item()) ;
        point = (torch.exp(torch.tensor(real_part)) * torch.exp(torch.tensor(imag_part))).item() ;
        list_J_k_p_form_.append(point) ;
    #end;%for nw in range(n_w):
#end;%for nk_p_r in range(n_k_p_r):
J_k_p_form_ = torch.tensor(list_J_k_p_form_).to(dtype=torch.complex64);
fnorm_disp(flag_verbose,'M_k_p_form_',M_k_p_form_,'J_k_p_form_',J_k_p_form_,' %<-- should be small if no aliasing');
#%%%%%%%%;
M_k_p_form_l2 = sum_0(torch.abs(M_k_p_form_)**2 * weight_2d_k_p_wk_) * (2*pi)**2;
disp(sprintf(' %% M_k_p_form_l2 = %0.16f',M_k_p_form_l2));
fnorm_disp(flag_verbose,'M_k_p_form_',M_k_p_form_,'M_k_p_quad_',M_k_p_quad_,' %<-- should be small if no aliasing');
M_k_p_reco_ = interp_x_p_to_k_p_xxnufft(n_k_p_r,k_p_r_,k_p_r_max,n_w_,weight_2d_k_p_wk_,n_x_p_r,x_p_r_,x_p_r_max,n_t_,weight_2d_x_p_tx_,M_x_p_form_)*(2*pi)**2;
fnorm_disp(flag_verbose,'M_k_p_form_',M_k_p_form_,'M_k_p_reco_',M_k_p_reco_,' %<-- should be small if no aliasing');
M_k_p_reco_l2 = sum_0(torch.abs(M_k_p_reco_)**2 * weight_2d_k_p_wk_) * (2*pi)**2;
disp(sprintf(' %% M_k_p_reco_l2 = %0.16f',M_k_p_reco_l2));
N_k_p_reco_ = torch.zeros(n_w_sum).to(dtype=torch.complex64);
eta = 1.0; #%eta = 1/max(1e-12,sqrt(2*pi*k_p_r_max)); %eta = pi/max(1e-12,x_p_r_max);
N_k_p_reco_ = xxnufft2d3(n_t_sum,x_u_0_tx_*eta,x_u_1_tx_*eta,M_x_p_form_*weight_2d_x_p_tx_,-1,1e-12,n_w_sum,2*pi*k_c_0_wk_/eta,2*pi*k_c_1_wk_/eta) * (2*pi)**2;
fnorm_disp(flag_verbose,'M_k_p_reco_',M_k_p_reco_,'N_k_p_reco_',N_k_p_reco_,' %<-- should be zero');
fnorm_disp(flag_verbose,'M_k_p_form_',M_k_p_form_,'N_k_p_reco_',N_k_p_reco_,' %<-- should be small if no aliasing');
if (flag_verbose>1):
    zarray_printf_margin(N_k_p_reco_,n_w_max,n_k_p_r,' %% N_k_p_reco_: '); disp(sprintf(' %%'));
#end;%if (flag_verbose>1);
M_x_u_reco_ = interp_k_p_to_x_c_xxnufft(n_x_M_u,diameter_x_c,n_x_M_u,diameter_x_c,n_k_p_r,k_p_r_,n_w_,M_k_p_quad_*weight_2d_k_p_wk_*(2*pi)**2)*np.sqrt(n_x_M_u**2) * n_w_sum;
M_x_u_reco_l2 = sum_0(torch.abs(M_x_u_reco_)**2)*dx_u**2;
disp(sprintf(' %% M_x_u_reco_l2 = %0.16f',M_x_u_reco_l2));
fnorm_disp(flag_verbose,'M_x_u_form_',M_x_u_form_,'M_x_u_reco_',M_x_u_reco_,' %<-- should be small if no aliasing');
M_x_p_reco_ = interp_k_p_to_x_p_xxnufft(n_x_p_r,x_p_r_,x_p_r_max,n_t_,weight_2d_x_p_tx_,n_k_p_r,k_p_r_,k_p_r_max,n_w_,weight_2d_k_p_wk_,M_k_p_form_)*(2*pi)**2;
M_x_p_reco_l2 = sum_0(torch.abs(M_x_p_reco_)**2 * weight_2d_x_p_tx_) * (2*pi)**2;
disp(sprintf(' %% M_x_p_reco_l2 = %0.16f',M_x_p_reco_l2));
#%%%%%%%%;

#%%%%%%%%;
disp(sprintf(' %% returning after %s at line %d',str_thisfunction,linenumber_here())); exit(0);
#%%%%%%%%;


