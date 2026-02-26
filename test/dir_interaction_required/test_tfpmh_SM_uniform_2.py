from dir_empm.dir_matlab_macros import * ;
from dir_empm.tfpmh_SM_uniform_2 import tfpmh_SM_uniform_2 ;

flag_verbose = 1;
parameter = {'type':'parameter'};
parameter['flag_deterministic']=1;
parameter['f_rand']=0.5;
n_w_max=8;
n_S=113;
viewing_azimu_b_S_=torch.linspace(0,2*pi,n_S).to(dtype=torch.float32);
viewing_polar_a_S_=torch.linspace(0,1*pi,n_S).to(dtype=torch.float32);
n_M=117;
#%%%%;
n_wSM = n_w_max*n_S*n_M;
X_wSM___ = torch.reshape(torch.fmod(torch.arange(n_wSM),19),mtr((n_w_max,n_S,n_M))).to(dtype=torch.float32);
delta_x_wSM___ = torch.reshape(torch.fmod(torch.arange(n_wSM),23),mtr((n_w_max,n_S,n_M))).to(dtype=torch.float32);
delta_y_wSM___ = torch.reshape(torch.fmod(torch.arange(n_wSM),29),mtr((n_w_max,n_S,n_M))).to(dtype=torch.float32);
gamma_z_wSM___ = torch.reshape(torch.fmod(torch.arange(n_wSM),31),mtr((n_w_max,n_S,n_M))).to(dtype=torch.float32);
I_value_wSM___ = torch.reshape(torch.fmod(torch.arange(n_wSM),37),mtr((n_w_max,n_S,n_M))).to(dtype=torch.float32);
tmp_t=tic();
(
    parameter,
    euler_polar_a_M_,
    euler_azimu_b_M_,
    euler_gamma_z_M_,
    image_delta_x_M_,
    image_delta_y_M_,
    image_I_value_M_,
    image_X_value_M_,
    image_S_index_M_,
) = tfpmh_SM_uniform_2(
    parameter,
    n_w_max,
    n_S,
    viewing_azimu_b_S_,
    viewing_polar_a_S_,
    n_M,
    X_wSM___,
    delta_x_wSM___,
    delta_y_wSM___,
    gamma_z_wSM___,
    I_value_wSM___,
);
tmp_t=toc(tmp_t);
if (flag_verbose>0): disp(sprintf(' %% tfpmh_SM_uniform_2: time %0.6fs',tmp_t)); #end;

dir_base = '/data/rangan' ;
dir_pymat = dir_base + '/dir_cryoem/dir_rangan_python/dir_pymat' ;
fname_pymat = dir_pymat + '/test_tfpmh_SM_uniform_2_wSM.mat' ;
disp(sprintf(' %% writing %s',fname_pymat));
matlab_save(
    fname_mat=fname_pymat,
    dictionary_original= {
        "euler_polar_a_M_":euler_polar_a_M_,
        "euler_azimu_b_M_":euler_azimu_b_M_,
        "euler_gamma_z_M_":euler_gamma_z_M_,
        "image_delta_x_M_":image_delta_x_M_,
        "image_delta_y_M_":image_delta_y_M_,
        "image_I_value_M_":image_I_value_M_,
        "image_X_value_M_":image_X_value_M_,
        "image_S_index_M_":image_S_index_M_,
    },
);

n_SM = n_S*n_M;
X_SM__ = torch.reshape(torch.fmod(torch.arange(n_SM),19),mtr((n_S,n_M))).to(dtype=torch.float32);
delta_x_SM__ = torch.reshape(torch.fmod(torch.arange(n_SM),23),mtr((n_S,n_M))).to(dtype=torch.float32);
delta_y_SM__ = torch.reshape(torch.fmod(torch.arange(n_SM),29),mtr((n_S,n_M))).to(dtype=torch.float32);
gamma_z_SM__ = torch.reshape(torch.fmod(torch.arange(n_SM),31),mtr((n_S,n_M))).to(dtype=torch.float32);
I_value_SM__ = torch.reshape(torch.fmod(torch.arange(n_SM),37),mtr((n_S,n_M))).to(dtype=torch.float32);
tmp_t=tic();
(
    parameter,
    euler_polar_a_M_,
    euler_azimu_b_M_,
    euler_gamma_z_M_,
    image_delta_x_M_,
    image_delta_y_M_,
    image_I_value_M_,
    image_X_value_M_,
    image_S_index_M_,
) = tfpmh_SM_uniform_2(
    parameter,
    n_w_max,
    n_S,
    viewing_azimu_b_S_,
    viewing_polar_a_S_,
    n_M,
    X_SM__,
    delta_x_SM__,
    delta_y_SM__,
    gamma_z_SM__,
    I_value_SM__,
);
tmp_t=toc(tmp_t);
if (flag_verbose>0): disp(sprintf(' %% tfpmh_SM_uniform_2: time %0.6fs',tmp_t)); #end;

dir_base = '/data/rangan' ;
dir_pymat = dir_base + '/dir_cryoem/dir_rangan_python/dir_pymat' ;
fname_pymat = dir_pymat + '/test_tfpmh_SM_uniform_2_SM.mat' ;
disp(sprintf(' %% writing %s',fname_pymat));
matlab_save(
    fname_mat=fname_pymat,
    dictionary_original= {
        "euler_polar_a_M_":euler_polar_a_M_,
        "euler_azimu_b_M_":euler_azimu_b_M_,
        "euler_gamma_z_M_":euler_gamma_z_M_,
        "image_delta_x_M_":image_delta_x_M_,
        "image_delta_y_M_":image_delta_y_M_,
        "image_I_value_M_":image_I_value_M_,
        "image_X_value_M_":image_X_value_M_,
        "image_S_index_M_":image_S_index_M_,
    },
);
