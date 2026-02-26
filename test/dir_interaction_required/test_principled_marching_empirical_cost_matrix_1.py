from dir_empm.dir_matlab_macros import * ;
from dir_empm.get_weight_3d_1 import get_weight_3d_1 ;
from dir_empm.get_weight_2d_2 import get_weight_2d_2 ;
from dir_empm.principled_marching_empirical_cost_matrix_1 import principled_marching_empirical_cost_matrix_1 ;

flag_verbose=1; flag_disp=0;nf=0;
print(f' %% testing principled_marching_empirical_cost_matrix_1');
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
);
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
(
    X_kk__,
    X_weight_r_,
) = principled_marching_empirical_cost_matrix_1(
    n_k_p_r,
    k_p_r_,
    weight_2d_k_p_r_,
    n_w_,
    n_M,
    M_k_p_wkM__,
);
#%%%%;
n_UX_rank = n_k_p_r-1; #%<-- just to check dimensions.; 
pm_n_UX_rank = n_UX_rank; #%<-- just to check dimension. ;
UX_kn__ = torch.zeros(mtr((n_k_p_r,n_UX_rank))).to(dtype=torch.float32);
U_t__,Sigma_,V_n__ = torch.linalg.svd(X_kk__.T,full_matrices=True); U_n__ = U_t__.T; #%<-- extra transposes to match matlab. ;
tmp_UX_kn__ = U_n__ ;
tmp_i8_index_ = matlab_index_2d_0(n_k_p_r,':',n_k_p_r,torch.arange(n_UX_rank));
UX_kn__.ravel()[tmp_i8_index_] = tmp_UX_kn__.ravel()[tmp_i8_index_].to(dtype=torch.float32);
#%%%%%%%%;

dir_base = '/data/rangan' ;
dir_ascii = dir_base + '/dir_cryoem/dir_rangan_python/dir_ascii' ;

fname_ascii = dir_ascii + '/UX_kn__.ascii'
print(f' %% writing fname_ascii: {fname_ascii}');
np.savetxt(fname_ascii,UX_kn__.numpy().ravel());
fname_ascii = dir_ascii + '/X_kk__.ascii'
print(f' %% writing fname_ascii: {fname_ascii}');
np.savetxt(fname_ascii,X_kk__.numpy().ravel());
fname_ascii = dir_ascii + '/X_weight_r_.ascii'
print(f' %% writing fname_ascii: {fname_ascii}');
np.savetxt(fname_ascii,X_weight_r_.numpy().ravel());
