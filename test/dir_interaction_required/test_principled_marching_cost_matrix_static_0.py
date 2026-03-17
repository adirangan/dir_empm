from dir_empm.dir_matlab_macros import * ;
from dir_empm.principled_marching_empirical_cost_matrix_1 import principled_marching_empirical_cost_matrix_1 ;
from dir_empm.principled_marching_empirical_cost_matrix_2 import principled_marching_empirical_cost_matrix_2 ;
from dir_empm.principled_marching_cost_matrix_7 import principled_marching_cost_matrix_7 ;

flag_verbose=1;

disp(sprintf(' %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% '));
disp(sprintf(' %% testing principled_marching_empirical_cost_matrix_2 using a static test'));
disp(sprintf(' %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% '));
n_k_p_r = int(3);
k_p_r_ = torch.tensor([3,5,7]).to(dtype=torch.float32);
weight_2d_k_p_r_ = torch.tensor([0.25,0.50,0.75]).to(dtype=torch.float32)**2;
n_w_ = torch.tensor([8,8,8]).to(dtype=torch.int32);
n_w_max = int(torch.max(n_w_).item());
n_w_sum = int(torch.sum(n_w_).item());
n_M = int(4);
M_k_p_wkM__ = torch.reshape(torch.tensor(np.mod(np.arange(n_w_sum*n_M),13)-6).to(dtype=torch.complex64) + i*torch.tensor(np.mod(np.arange(n_w_sum*n_M),17)-8).to(dtype=torch.complex64),mtr((n_w_sum,n_M))) / 19.0;

#%%%%%%%%;
(
    X0_kk__,
    X0_weight_r_,
) = principled_marching_empirical_cost_matrix_2(
    n_k_p_r,
    k_p_r_,
    weight_2d_k_p_r_,
    n_w_,
    n_M,
    M_k_p_wkM__,
)[:2];
#%%%%%%%%;

#%%%%%%%%;
#% disp(num2str(X0_kk__(:),';%+0.16f ...\n'))
#%%%%%%%%;
static_X0_kk__ = torch.tensor([
    +4.0089928109038837,
    -3.7295092288655907,
    +3.9269739377646369,
    -3.7295092288655907,
    +15.5917023474446150,
    -9.8236823757639957,
    +3.9269739377646369,
    -9.8236823757639957,
    +31.8133908051085292,
]).to(dtype=torch.float32);
static_X0_kk__ = torch.reshape(static_X0_kk__,mtr((3,3)));
fnorm_disp(flag_verbose,'static_X0_kk__',static_X0_kk__,'X0_kk__',X0_kk__,'%<-- should be zero');
#%%%%%%%%;
static_X0_weight_r_ = torch.tensor([
    +0.25,
    +0.50,
    +0.75,
]).to(dtype=torch.float32);
fnorm_disp(flag_verbose,'static_X0_weight_r_',static_X0_weight_r_,'X0_weight_r_',X0_weight_r_,'%<-- should be zero');

disp(sprintf(' %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% '));
disp(sprintf(' %% testing principled_marching_empirical_cost_matrix_2 using a static test'));
disp(sprintf(' %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% '));
n_k_p_r = int(3);
k_p_r_ = torch.tensor([3,5,7]).to(dtype=torch.float32);
weight_k_p_r_ = torch.tensor([0.25,0.50,0.75]).to(dtype=torch.float32)**2;
l_max_ = torch.tensor([2,4,6]).to(dtype=torch.int32);
n_y_ = (1+l_max_)**2;
n_y_sum = int(torch.sum(n_y_).item());
n_molecule = int(2);
molecule_density_ = torch.tensor([0.7,1.3]).to(dtype=torch.float32);
a_k_Y_ykv__ = torch.reshape(torch.tensor(np.mod(np.arange(n_y_sum*n_molecule),13)-6).to(dtype=torch.complex64) + i*torch.tensor(np.mod(np.arange(n_y_sum*n_molecule),17)-8).to(dtype=torch.complex64),mtr((n_y_sum,n_molecule))) / 19.0;
CTF_k_p_r_xcor_kk__ = torch.reshape(torch.tensor([1,-2,-3,-2,4,-5,-3,-5,6]).to(dtype=torch.float32),mtr((3,3))) / 7.0;
delta_sigma = 0.15;
pm_delta_integral_tolerance = 1e-9;
#%%%%%%%%;
(
    X_kk__,
    X_weight_r_,
    X_ori_kk__,
    X_tau_kk__,
    weight_so3,
) = principled_marching_cost_matrix_7(
    n_k_p_r,
    k_p_r_,
    weight_k_p_r_,
    l_max_,
    n_molecule,
    molecule_density_,
    a_k_Y_ykv__,
    CTF_k_p_r_xcor_kk__,
    delta_sigma,
    pm_delta_integral_tolerance,
)[:5];
#%%%%%%%%;

#%%%%%%%%;
#% disp(num2str(X_kk__(:),';%+0.16f ...\n'))
#%%%%%%%%;
static_X_kk__ = torch.tensor([
    +0.8779589809787692,
    +0.2870699378074536,
    -0.0048051446643973,
    +0.2870699378074536,
    +33.3109107970662208,
    +4.5345665123465286,
    -0.0048051446643973,
    +4.5345665123465446,
    +215.0515520690952656,
]).to(dtype=torch.float32);
static_X_kk__ = torch.reshape(static_X_kk__,mtr((3,3)));
fnorm_disp(flag_verbose,'static_X_kk__',static_X_kk__,'X_kk__',X_kk__,'%<-- should be zero');
#%%%%%%%%;
static_X_weight_r_ = torch.tensor([
    +0.25,
    +0.50,
    +0.75,
]).to(dtype=torch.float32);
fnorm_disp(flag_verbose,'static_X_weight_r_',static_X_weight_r_,'X_weight_r_',X_weight_r_,'%<-- should be zero');
#%%%%%%%%;
#% disp(num2str(X_ori_kk__(:),';%+0.16f ...\n'))
#%%%%%%%%;
static_X_ori_kk__ = torch.tensor([
    +0.0055598288484369 + +0.0000000000000000*i,
    +0.0107424812030075 + +0.0043485358132173*i,
    -0.0371050776612584 + -0.0114625791452315*i,
    +0.0107424812030075 + -0.0043485358132173*i,
    +0.2109438068856351 + +0.0000000000000000*i,
    +0.1696886129798179 + -0.0110222101305897*i,
    -0.0371050776612584 + +0.0114625791452315*i,
    +0.1696886129798179 + +0.0110222101305897*i,
    +1.3618298624851604 + +0.0000000000000000*i,
]).to(dtype=torch.complex64);
static_X_ori_kk__ = torch.reshape(static_X_ori_kk__,mtr((3,3)));
fnorm_disp(flag_verbose,'static_X_ori_kk__',static_X_ori_kk__,'X_ori_kk__',X_ori_kk__,'%<-- should be zero');
#%%%%%%%%;
#% disp(num2str(X_tau_kk__(:),';%+0.16f ...\n'))
#%%%%%%%%;
static_X_tau_kk__ = torch.tensor([
    +0.0414976441478332 + +0.0000000000000000*i,
    -0.0863150998274930 + +0.0730358537001864*i,
    -0.4169780925489920 + +0.0276813696844958*i,
    -0.0863150998274930 + -0.0730358537001864*i,
    +0.3080785101535136 + +0.0000000000000000*i,
    -0.7633613692888467 + -0.5635868282853953*i,
    -0.4169780925489920 + -0.0276813696844958*i,
    -0.7633613692888467 + +0.5635868282853953*i,
    +2.8055727255467069 + +0.0000000000000000*i,
]).to(dtype=torch.complex64);
static_X_tau_kk__ = torch.reshape(static_X_tau_kk__,mtr((3,3)));
fnorm_disp(flag_verbose,'static_X_tau_kk__',static_X_tau_kk__,'X_tau_kk__',X_tau_kk__,'%<-- should be zero');
static_weight_so3 = 157.9136704174297279 ; #%<-- should be (2*pi)*(2*pi)*4 ;
fnorm_disp(flag_verbose,'static_weight_so3',static_weight_so3,'weight_so3',weight_so3,'%<-- should be zero');



