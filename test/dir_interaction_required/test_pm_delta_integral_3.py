from dir_empm.dir_matlab_macros import * ;
from dir_empm.pm_delta_integral_3 import pm_delta_integral_3 ;

flag_verbose=1;
str_thisfunction = 'pm_delta_integral_3';
if (flag_verbose>0): disp(sprintf(' %% testing %s',str_thisfunction)); #end;
n_k_p_r = 9; k_p_r_ = torch.linspace(1,9,n_k_p_r).to(dtype=torch.float32); delta_sigma = 0.05; l_max = 32;
tmp_t = tic();
I_pos_3__,I_neg_3_ = pm_delta_integral_3(n_k_p_r,k_p_r_,delta_sigma,l_max);
tmp_t = toc(tmp_t); disp(sprintf(' %% pm_delta_integral_3: %0.6fs',tmp_t));
dir_base = '/data/rangan' ;
dir_pymat = dir_base + '/dir_cryoem/dir_rangan_python/dir_pymat' ;
fname_pymat = dir_pymat + '/test_pm_delta_integral_3.mat' ;
disp(sprintf(' %% writing fname_pymat: %s',fname_pymat));
matlab_save(
    fname_mat=fname_pymat,
    dictionary_original= {
        "I_pos_3__":I_pos_3__,
        "I_neg_3_":I_neg_3_,
    },
);
