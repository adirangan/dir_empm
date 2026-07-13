from dir_empm.dir_matlab_macros import * ;
from dir_empm.get_weight_3d_1 import get_weight_3d_1 ;
from dir_empm.get_weight_2d_2 import get_weight_2d_2 ;
from dir_empm.interp_p_to_q import interp_p_to_q ;
from dir_empm.rotate_p_to_p_fftw import rotate_p_to_p_fftw ;
from dir_empm.transf_p_to_p import transf_p_to_p ;

flag_verbose=1; flag_disp=0;nf=0;
print(f' %% testing rotate_to_p_fftw');
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
n_M = 4;
delta_2M__ = torch.reshape(torch.linspace(-0.05,+0.05,n_2*n_M),mtr((n_2,n_M))).to(dtype=torch.float32);
M_k_p_wkM__ = torch.zeros(mtr((n_w_sum,n_M))).to(dtype=torch.complex64);
for nM in range(n_M):
    delta_0 = delta_2M__[nM,0].item(); delta_1 = delta_2M__[nM,1].item();
    M_k_p_wkM__[nM,:] = torch.exp(+i*2*pi*(k_c_0_wk_*delta_0 + k_c_1_wk_*delta_1));
#end;%for nM=0:n_M-1;
#%%%%;
gamma_z_M_ = torch.linspace(0,pi,n_M).to(dtype=torch.float32);
delta_x_M_ = torch.linspace(-0.1,+0.1,n_M).to(dtype=torch.float32);
delta_y_M_ = torch.linspace(-0.2,+0.2,n_M).to(dtype=torch.float32);
T_k_p_wkM__ = torch.zeros(mtr((n_w_sum,n_M))).to(dtype=torch.complex64);
T_k_p_wkM__ = torch.reshape(transf_p_to_p(n_k_p_r,k_p_r_,n_w_,n_w_sum,rotate_p_to_p_fftw(n_k_p_r,n_w_,n_w_sum,M_k_p_wkM__,gamma_z_M_),-delta_x_M_,-delta_y_M_),mtr((n_w_sum,n_M)));
T_k_q_wkM__ = torch.reshape(interp_p_to_q(n_k_p_r,n_w_,n_w_sum,T_k_p_wkM__),mtr((n_w_sum,n_M)));
#%%%%;

dir_base = '/data/rangan' ;
dir_ascii = dir_base + '/dir_cryoem/dir_rangan_python/dir_ascii' ;

fname_ascii = dir_ascii + '/T_k_p_wkM__.ascii'
print(f' %% writing fname_ascii: {fname_ascii}');
np.savetxt(fname_ascii,T_k_p_wkM__.numpy().ravel());
fname_ascii = dir_ascii + '/T_k_q_wkM__.ascii'
print(f' %% writing fname_ascii: {fname_ascii}');
np.savetxt(fname_ascii,T_k_q_wkM__.numpy().ravel());
