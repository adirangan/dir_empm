from dir_empm.dir_matlab_macros import * ;

def tfpmh_SM_uniform_3(
        parameter =None,
        n_S=None,
        viewing_azimu_b_S_=None,
        viewing_polar_a_S_=None,
        n_M=None,
        X_SM__=None,
        delta_x_SM__=None,
        delta_y_SM__=None,
        gamma_z_SM__=None,
        I_value_SM__=None,
):

    #%%%%%%%%;
    str_thisfunction = 'tfpmh_SM_uniform_3';
    #%%%%%%%%;

    if isempty(parameter): parameter = {'type':'parameter'}; #end;%if isempty(parameter);
    if 'flag_verbose' not in parameter: parameter['flag_verbose'] = 0; #end;
    flag_verbose = parameter['flag_verbose'];
    if 'flag_deterministic' not in parameter: parameter['flag_deterministic'] = 0; #end;
    flag_deterministic = parameter['flag_deterministic'];
    if 'f_rand' not in parameter: parameter['f_rand'] = 0.05; #end;
    f_rand = parameter['f_rand'];
    if isempty(I_value_SM__): I_value_SM__ = 1+0*X_SM__; #end;
    #%%%%%%%%;
    if (flag_verbose>0): disp(sprintf(' %% [entering %s],str_thisfunction')); #end;
    #%%%%%%%%;
    euler_polar_a_M_ = torch.zeros(n_M).to(dtype=torch.float32);
    euler_azimu_b_M_ = torch.zeros(n_M).to(dtype=torch.float32);
    euler_gamma_z_M_ = torch.zeros(n_M).to(dtype=torch.float32);
    image_delta_x_M_ = torch.zeros(n_M).to(dtype=torch.float32);
    image_delta_y_M_ = torch.zeros(n_M).to(dtype=torch.float32);
    image_I_value_M_ = torch.zeros(n_M).to(dtype=torch.float32);
    image_X_value_M_ = torch.zeros(n_M).to(dtype=torch.float32);
    image_S_index_M_ = torch.zeros(n_M).to(dtype=torch.int32);
    #%%%%%%%%;
    assert(ndims(X_SM__)==2);
    #%%%%%%%%;
    X_srt_SM__,index_srt_SM__ = torch.sort(X_SM__,dim=1-0,descending=True);
    if (f_rand<=0): index_nS_srt_M_ = torch.zeros(n_M).to(dtype=torch.int32); #end;
    if (f_rand> 0):
        index_nS_srt_M_ = np.floor(f_rand*0.5*int(n_S))*torch.ones(n_M).to(dtype=torch.int32);
        if flag_deterministic==0:
            index_nS_srt_M_ = torch.floor(f_rand*torch.rand(n_M)*int(n_S)).to(dtype=torch.int32)*torch.ones(n_M).to(dtype=torch.int32);
        #end;%if flag_deterministic==0;
    #end;%if (f_rand> 0);
    index_nS_M_ = index_srt_SM__.ravel()[index_nS_srt_M_.to(dtype=torch.int32) + torch.arange(n_M).to(dtype=torch.int32)*int(n_S)];
    index_nSM_ = index_nS_M_.to(dtype=torch.int32) + torch.arange(n_M).to(dtype=torch.int32)*int(n_S);
    euler_polar_a_M_ = viewing_polar_a_S_.ravel()[index_nS_M_];
    euler_azimu_b_M_ = viewing_azimu_b_S_.ravel()[index_nS_M_];
    euler_gamma_z_M_ = gamma_z_SM__.ravel()[index_nSM_];
    image_delta_x_M_ = delta_x_SM__.ravel()[index_nSM_];
    image_delta_y_M_ = delta_y_SM__.ravel()[index_nSM_];
    image_X_value_M_ = X_SM__.ravel()[index_nSM_];
    image_S_index_M_ = index_nS_M_.ravel();
    image_I_value_M_ = I_value_SM__.ravel()[index_nSM_];
    
    return(
        parameter,
        euler_polar_a_M_,
        euler_azimu_b_M_,
        euler_gamma_z_M_,
        image_delta_x_M_,
        image_delta_y_M_,
        image_I_value_M_,
        image_X_value_M_,
        image_S_index_M_,
    );
