from dir_empm.dir_matlab_macros import * ;
from dir_empm.get_r8_delta_2 import get_r8_delta_2 ;

str_thisfunction = 'get_delta_2';
flag_verbose=1;
if (flag_verbose>0): print(f' %% testing {str_thisfunction}');
k_p_r_max = 48.0/(2*pi); delta_r_max = 0.5/np.maximum(1e-12,k_p_r_max); svd_eps = 1e-6; n_delta_v_requested = 128;
n_delta_v_out,r8_delta_x_,r8_delta_y_ = get_r8_delta_2(delta_r_max,n_delta_v_requested);
if (flag_verbose>0): print(f' %% n_delta_v_requested {n_delta_v_requested}');
if (flag_verbose>0): print(f' %% n_delta_v_out {n_delta_v_out}');
if (flag_verbose>0): print(f' %% r8_delta_x_:'); print(r8_delta_x_);
if (flag_verbose>0): print(f' %% r8_delta_y_:'); print(r8_delta_y_);
