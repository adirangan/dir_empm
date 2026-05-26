from dir_empm.dir_matlab_macros import * ;
from dir_empm.get_weight_3d_1 import get_weight_3d_1 ;
from dir_empm.get_weight_2d_2 import get_weight_2d_2 ;
from dir_empm.CTF_k_p_wkC__from_params_0 import CTF_k_p_wkC__from_params_0 ;

flag_uniform_over_n_k_p_r = 0;
flag_verbose=1; flag_disp=1; nf=0;
str_thisfunction = 'CTF_k_p_wkC__from_params_0';
if (flag_verbose>0): disp(sprintf(' %% testing %s',str_thisfunction)); #end;
k_int = 48*2;
k_eq_d_double = 1.0;
n_w_int = 1;
k_p_r_max = k_int/(2*pi); k_eq_d = k_eq_d_double/(2*pi); str_T_vs_L = 'L';
(
    n_k_p_r,
    k_p_r_,
    weight_3d_k_p_r_,
) = get_weight_3d_1(
    flag_verbose,
    k_p_r_max,
    k_eq_d,
    str_T_vs_L,
);
l_max_upb = matlab_scalar_round(2*pi*k_p_r_max);
l_max_max = int(np.minimum(l_max_upb,1+np.ceil(2*pi*k_p_r_[-1].item())));
n_w_max = int(n_w_int*2*(l_max_max+1)); n_w_0in_ = n_w_max*torch.ones(n_k_p_r).to(dtype=torch.int32);
if flag_uniform_over_n_k_p_r==1:
    template_k_eq_d = -1; 
#%end; %<-- use to test flag_uniform_over_n_k_p_r==1;
if flag_uniform_over_n_k_p_r==0:
    template_k_eq_d = 0.5; n_w_0in_ = None; 
#end;%<-- use to test flag_uniform_over_n_k_p_r==0;
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
    template_k_eq_d,
    n_w_0in_,
    weight_3d_k_p_r_,
);
n_w_max = int(torch.max(n_w_).item()); n_w_sum = int(torch.sum(n_w_).item()); n_w_csum_ = cumsum_0(n_w_);
if (flag_verbose>0): disp(sprintf(' %% n_k_p_r %d n_w_max %d n_w_sum %d numel(unique(n_w_)) %d ',n_k_p_r,n_w_max,n_w_sum,numel_unique(n_w_))); #end;
#%%%%%%%%;
if (flag_verbose>0): disp(sprintf(' %% using params from two CTF functions in image-stack from EMPIAR-10005')); #end;
n_CTF = 2;
Voltage_C_ = torch.zeros(n_CTF).to(dtype=torch.float64);
DefocusU_C_ = torch.zeros(n_CTF).to(dtype=torch.float64);
DefocusV_C_ = torch.zeros(n_CTF).to(dtype=torch.float64);
DefocusAngle_C_ = torch.zeros(n_CTF).to(dtype=torch.float64);
SphericalAberration_C_ = torch.zeros(n_CTF).to(dtype=torch.float64);
AmplitudeContrast_C_ = torch.zeros(n_CTF).to(dtype=torch.float64);
Voltage_C_[0] = 300; Voltage_C_[1] = 300;
DefocusU_C_[0] = 22174.2; DefocusU_C_[1] = 21912.2;
DefocusV_C_[0] = 21393.0; DefocusV_C_[1] = 22462.6;
DefocusAngle_C_[0] = 1.60; DefocusAngle_C_[1] = 73.67;
SphericalAberration_C_[0] = 2.0; SphericalAberration_C_[1] = 2.0;
AmplitudeContrast_C_[0] = 0.1; AmplitudeContrast_C_[1] = 0.1;
n_Pixels_across = 256;
Pixel_Spacing = 1.2156;
#%%%%%%%%;
CTF_k_p_wkC_func__ = CTF_k_p_wkC__from_params_0(
    n_k_p_r,
    k_p_r_,
    n_w_,
    n_CTF,
    Voltage_C_,
    DefocusU_C_,
    DefocusV_C_,
    DefocusAngle_C_,
    SphericalAberration_C_,
    AmplitudeContrast_C_,
    n_Pixels_across,
    Pixel_Spacing,
);
dir_base = '/data/rangan' ;
dir_pymat = dir_base + '/dir_cryoem/dir_rangan_python/dir_pymat' ;
if flag_uniform_over_n_k_p_r==0: fname_pymat = dir_pymat + '/test_CTF_k_p_wkC__from_params_0.mat' ; #end;
if flag_uniform_over_n_k_p_r==1: fname_pymat = dir_pymat + '/test_CTF_k_p_wkC__from_params_0_uniform_over_n_k_p_r.mat' ; #end;
disp(sprintf(' %% writing fname_pymat: %s',fname_pymat));
matlab_save(
    fname_mat=fname_pymat,
    dictionary_original= {
        "CTF_k_p_wkC_func__":CTF_k_p_wkC_func__,
    },
);

