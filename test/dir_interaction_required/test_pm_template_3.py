from dir_empm.dir_matlab_macros import * ;
from dir_empm.pm_delta_integral_3 import pm_delta_integral_3 ;
from dir_empm.pm_template_2 import pm_template_2 ;
from dir_empm.pm_template_3 import pm_template_3 ;

flag_verbose = 1;
#n_k_p_r = 49;
n_k_p_r = 7;
k_p_r_max = 1;
k_p_r_ = torch.ones(n_k_p_r).to(dtype=torch.float32);
weight_k_p_r_ = torch.ones(n_k_p_r).to(dtype=torch.float32);
#l_max = 80;
l_max = 13;
n_y = (l_max+1)**2;
l_max_ = l_max*torch.ones(n_k_p_r).to(dtype=torch.int32);
n_y_sum = int(torch.sum((1+l_max_)**2).item());
a_k_Y_ = ((torch.fmod(torch.arange(n_y_sum).to(dtype=torch.int32),89)-44)/89 + i*(torch.fmod(torch.arange(n_y_sum).to(dtype=torch.int32),97)-48)/97).to(dtype=torch.complex64);
viewing_euler_k_eq_d = 1/(2*pi);
template_inplane_k_eq_d = -1;
n_w_max = int(np.maximum(6,98));
#%%%%%%%%;
tmp_t = tic();
(
    template3_wkS___,
    n_w_,
    n_S,
    viewing_azimu_b_S_,
    viewing_polar_a_S_,
    viewing_weight_S_,
    n_viewing_polar_a,
    viewing_polar_a_,
    n_viewing_azimu_b_,
) = pm_template_3(
    flag_verbose,
    l_max,
    n_k_p_r,
    torch.reshape(a_k_Y_,mtr((n_y,n_k_p_r))),
    viewing_euler_k_eq_d,
    template_inplane_k_eq_d,
    n_w_max,
)[0:9];
tmp_t = toc(tmp_t); disp(sprintf(' %% pm_template_3: %0.2fs',tmp_t));
disp(sprintf(' %% n_S %d',n_S));
template3_wkS__ = torch.reshape(template3_wkS___,mtr((n_w_max*n_k_p_r,n_S)));
#%%%%%%%%;
tmp_t = tic();
(
    template2_wkS___,
    n_w_,
    n_S,
    viewing_azimu_b_S_,
    viewing_polar_a_S_,
    viewing_weight_S_,
    n_viewing_polar_a,
    viewing_polar_a_,
    n_viewing_azimu_b_,
) = pm_template_2(
    flag_verbose,
    l_max,
    n_k_p_r,
    torch.reshape(a_k_Y_,mtr((n_y,n_k_p_r))),
    viewing_euler_k_eq_d,
    template_inplane_k_eq_d,
    n_w_max,
)[0:9];
tmp_t = toc(tmp_t); disp(sprintf(' %% pm_template_2: %0.2fs',tmp_t));
disp(sprintf(' %% n_S %d',n_S));
template2_wkS__ = torch.reshape(template2_wkS___,mtr((n_w_max*n_k_p_r,n_S)));
#%%%%%%%%;
fnorm_disp(flag_verbose,'template2_wkS__',template2_wkS__,'template3_wkS__',template3_wkS__,' %%<-- should be zero');

