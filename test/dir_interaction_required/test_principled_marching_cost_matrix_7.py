from dir_empm.dir_matlab_macros import * ;
from dir_empm.principled_marching_cost_matrix_7 import principled_marching_cost_matrix_7 ;

str_thisfunction = 'principled_marching_cost_matrix_7';
flag_verbose=1;
if (flag_verbose>0): disp(sprintf(' %% testing %s',str_thisfunction)); #end;
n_k_p_r = 49;
k_p_r_ = torch.linspace(1,7,n_k_p_r).to(dtype=torch.float32);
weight_k_p_r_ = torch.cos(k_p_r_)**2;
l_max_ = torch.ceil(3 + k_p_r_*0.5);
n_molecule = 2;
molecule_density_ = torch.linspace(3,5,n_molecule).to(dtype=torch.float32);
n_y_ = (1+l_max_)**2;
n_y_sum = int(torch.sum(n_y_).item());
a_k_Y_ykv__ = (torch.reshape(torch.linspace(1,10,n_y_sum*n_molecule),mtr((n_y_sum,n_molecule))) + i*torch.reshape(torch.linspace(0,9,n_y_sum*n_molecule),mtr((n_y_sum,n_molecule)))).to(dtype=torch.complex64);
CTF_k_p_r_xcor_k_ = torch.linspace(5,11,n_k_p_r).to(dtype=torch.float32);
CTF_k_p_r_xcor_kk__ = torch.reshape(CTF_k_p_r_xcor_k_,mtr((n_k_p_r,1)))*torch.reshape(CTF_k_p_r_xcor_k_,mtr((1,n_k_p_r)));
delta_sigma = 0.05;
pm_delta_integral_tolerance = 1e-4;
tmp_t=tic();
(
    X_7_kk__,
    X_7_weight_r_,
    X_7_ori_kk__,
    X_7_tau_kk__,
    weight_so3_7,
    n_m_max_7,
    polar_a_7_,
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
);
tmp_t=toc(tmp_t);
if (flag_verbose>0): disp(sprintf(' %% principled_marching_cost_matrix_7: %0.6fs',tmp_t)); #end;
#%%%%%%%%;
dir_base = '/data/rangan' ;
dir_pymat = dir_base + '/dir_cryoem/dir_rangan_python/dir_pymat' ;
fname_pymat = dir_pymat + '/test_principled_marching_cost_matrix_7.mat' ;
disp(sprintf(' %% writing fname_pymat: %s',fname_pymat));
matlab_save(
    fname_mat=fname_pymat,
    dictionary_original= {
        "X_7_kk__":X_7_kk__,
        "X_7_weight_r_":X_7_weight_r_,
        "X_7_ori_kk__":X_7_ori_kk__,
        "X_7_tau_kk__":X_7_tau_kk__,
        "weight_so3_7":weight_so3_7,
        "n_m_max_7":n_m_max_7,
        "polar_a_7_":polar_a_7_,
    },
);
#%%%%%%%%;




