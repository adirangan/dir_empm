from dir_empm.dir_matlab_macros import * ;
from dir_empm.interp_p_to_q import interp_p_to_q ;
from dir_empm.get_weight_3d_1 import get_weight_3d_1 ;
from dir_empm.get_weight_2d_2 import get_weight_2d_2 ;
from dir_empm.principled_marching_empirical_cost_matrix_1 import principled_marching_empirical_cost_matrix_1 ;
from dir_empm.principled_marching_empirical_cost_matrix_2 import principled_marching_empirical_cost_matrix_2 ;

str_thisfunction = 'principled_marching_empirical_cost_matrix_2';
flag_verbose=1;
if (flag_verbose>0): disp(sprintf(' %% testing %s',str_thisfunction)); #end;
n_k_p_r = 49;
k_p_r_ = torch.sort(torch.rand(n_k_p_r).to(dtype=torch.float32))[0];
weight_2d_k_p_r_ = torch.rand(n_k_p_r).to(dtype=torch.float32);
n_w_max = 98;
n_w_ = n_w_max*torch.ones(n_k_p_r).to(dtype=torch.int32); n_w_sum = int(torch.sum(n_w_).item());
n_M = 1024;
M_k_p_wkM__ = torch.randn(mtr((n_w_sum,n_M))).to(dtype=torch.complex64) + i*torch.randn(mtr((n_w_sum,n_M))).to(dtype=torch.complex64);
M_k_q_wkM__ = interp_p_to_q(n_k_p_r,n_w_,n_w_sum,M_k_p_wkM__);
#%%%%%%%%;
tmp_t=tic();
(
    X_1_kk__,
    X_1_weight_r_,
) = principled_marching_empirical_cost_matrix_1(
    n_k_p_r,
    k_p_r_,
    weight_2d_k_p_r_,
    n_w_,
    n_M,
    M_k_p_wkM__,
);
tmp_t=toc(tmp_t);
if (flag_verbose>0): disp(sprintf(' %% principled_marching_empirical_cost_matrix_1: %0.6fs',tmp_t)); #end;
#%%%%%%%%;
tmp_t=tic();
(
    X_2_kk__,
    X_2_weight_r_,
) = principled_marching_empirical_cost_matrix_2(
    n_k_p_r,
    k_p_r_,
    weight_2d_k_p_r_,
    n_w_,
    n_M,
    M_k_p_wkM__,
);
tmp_t=toc(tmp_t);
if (flag_verbose>0): disp(sprintf(' %% principled_marching_empirical_cost_matrix_2: %0.6fs',tmp_t)); #end;
#%%%%%%%%;
fnorm_disp(flag_verbose,'X_1_kk__',X_1_kk__,'X_2_kk__',X_2_kk__,' %%<-- should be zero');
fnorm_disp(flag_verbose,'X_1_weight_r_',X_1_weight_r_,'X_2_weight_r_',X_2_weight_r_,' %%<-- should be zero');
#%%%%%%%%;

