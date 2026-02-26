from dir_empm.dir_matlab_macros import * ;
from dir_empm.tfpmh_SM_uniform_3 import tfpmh_SM_uniform_3 ;

flag_verbose = 1;
parameter = {'type':'parameter'};
parameter['flag_deterministic']=1;
parameter['f_rand']=0.15;
n_S=113;
viewing_azimu_b_S_=torch.linspace(0,2*pi,n_S).to(dtype=torch.float64); #%<-- Note increased precision to match matlab. ;
viewing_polar_a_S_=torch.linspace(0,1*pi,n_S).to(dtype=torch.float64); #%<-- Note increased precision to match matlab. ;
n_M=117;
#%%%%;
n_SM = n_S*n_M;
X_SM__ = torch.reshape(torch.fmod(torch.arange(n_SM).to(dtype=torch.float64),np.sqrt(19)),mtr((n_S,n_M))).to(dtype=torch.float64); #%<-- Note increased precision to match matlab. ;
delta_x_SM__ = torch.reshape(torch.fmod(torch.arange(n_SM).to(dtype=torch.float64),np.sqrt(23)),mtr((n_S,n_M))).to(dtype=torch.float64); #%<-- Note increased precision to match matlab. ;
delta_y_SM__ = torch.reshape(torch.fmod(torch.arange(n_SM).to(dtype=torch.float64),np.sqrt(29)),mtr((n_S,n_M))).to(dtype=torch.float64); #%<-- Note increased precision to match matlab. ;
gamma_z_SM__ = torch.reshape(torch.fmod(torch.arange(n_SM).to(dtype=torch.float64),np.sqrt(31)),mtr((n_S,n_M))).to(dtype=torch.float64); #%<-- Note increased precision to match matlab. ;
I_value_SM__ = torch.reshape(torch.fmod(torch.arange(n_SM).to(dtype=torch.float64),np.sqrt(37)),mtr((n_S,n_M))).to(dtype=torch.float64); #%<-- Note increased precision to match matlab. ;
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
) = tfpmh_SM_uniform_3(
    parameter,
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
if (flag_verbose>0): disp(sprintf(' %% tfpmh_SM_uniform_3: time %0.6fs',tmp_t)); #end;

dir_base = '/data/rangan' ;
dir_pymat = dir_base + '/dir_cryoem/dir_rangan_python/dir_pymat' ;
fname_pymat = dir_pymat + '/test_tfpmh_SM_uniform_3_SM.mat' ;
disp(sprintf(' %% writing %s',fname_pymat));
matlab_save(
    fname_mat=fname_pymat,
    dictionary_original= {
        "viewing_azimu_b_S_":viewing_azimu_b_S_,
        "viewing_polar_a_S_":viewing_polar_a_S_,
        "X_SM__":X_SM__,
        "delta_x_SM__":delta_x_SM__,
        "delta_y_SM__":delta_y_SM__,
        "gamma_z_SM__":gamma_z_SM__,
        "I_value_SM__":I_value_SM__,
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
