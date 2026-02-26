from dir_empm.dir_matlab_macros import * ;

def tfpmh_SM_uniform_2(
        parameter =None,
        n_w_max=None,
        n_S=None,
        viewing_azimu_b_S_=None,
        viewing_polar_a_S_=None,
        n_M=None,
        X_wSM___=None,
        delta_x_wSM___=None,
        delta_y_wSM___=None,
        gamma_z_wSM___=None,
        I_value_wSM___=None,
):

    #%%%%%%%%;
    str_thisfunction = 'tfpmh_SM_uniform_2';
    #%%%%%%%%;

    if isempty(parameter): parameter = {'type':'parameter'}; #end;%if isempty(parameter);
    if 'flag_verbose' not in parameter: parameter['flag_verbose'] = 0; #end;
    flag_verbose = parameter['flag_verbose'];
    if 'flag_deterministic' not in parameter: parameter['flag_deterministic'] = 0; #end;
    flag_deterministic = parameter['flag_deterministic'];
    if 'f_rand' not in parameter: parameter['f_rand'] = 0.05; #end;
    f_rand = parameter['f_rand'];
    if isempty(I_value_wSM___): I_value_wSM___ = 1+0*X_wSM___; #end;
    #%%%%%%%%;
    if (flag_verbose>0): disp(sprintf(' %% [entering %s],str_thisfunction')); #end;

    euler_polar_a_M_ = torch.zeros(n_M).to(dtype=torch.float32);
    euler_azimu_b_M_ = torch.zeros(n_M).to(dtype=torch.float32);
    euler_gamma_z_M_ = torch.zeros(n_M).to(dtype=torch.float32);
    image_delta_x_M_ = torch.zeros(n_M).to(dtype=torch.float32);
    image_delta_y_M_ = torch.zeros(n_M).to(dtype=torch.float32);
    image_I_value_M_ = torch.zeros(n_M).to(dtype=torch.float32);
    image_X_value_M_ = torch.zeros(n_M).to(dtype=torch.float32);
    image_S_index_M_ = torch.zeros(n_M).to(dtype=torch.int32);

    #%%%%%%%%;
    for nM in range(n_M):
        tmp_image_X_value_S_ = torch.zeros(n_S).to(dtype=torch.float32);
        tmp_gamma_z_S_ = torch.zeros(n_S).to(dtype=torch.float32);
        tmp_delta_x_S_ = torch.zeros(n_S).to(dtype=torch.float32);
        tmp_delta_y_S_ = torch.zeros(n_S).to(dtype=torch.float32);
        tmp_I_value_S_ = torch.zeros(n_S).to(dtype=torch.float32);
        for nS in range(n_S):
            if (ndims(X_wSM___)==3):
                tmp_X,nw = torch.max(torch.real(X_wSM___[nM,nS,:].ravel()),dim=0);
                tmp_X = tmp_X.item(); nw = int(nw.item());
                tmp_gamma_z = gamma_z_wSM___[nM,nS,nw].item();
                tmp_delta_x = delta_x_wSM___[nM,nS,nw].item();
                tmp_delta_y = delta_y_wSM___[nM,nS,nw].item();
                tmp_I_value = I_value_wSM___[nM,nS,nw].item();
            #end;%if (ndims(X_wSM___)==3);
            if (ndims(X_wSM___)==2):
                tmp_X = X_wSM___[nM,nS].item();
                tmp_gamma_z = gamma_z_wSM___[nM,nS].item();
                tmp_delta_x = delta_x_wSM___[nM,nS].item();
                tmp_delta_y = delta_y_wSM___[nM,nS].item();
                tmp_I_value = I_value_wSM___[nM,nS].item();
            #end;%if (ndims(X_wSM___)==2);
            tmp_image_X_value_S_[nS] = tmp_X;
            tmp_gamma_z_S_[nS] = tmp_gamma_z;
            tmp_delta_x_S_[nS] = tmp_delta_x;
            tmp_delta_y_S_[nS] = tmp_delta_y;
            tmp_I_value_S_[nS] = tmp_I_value;
        #end;%for nS=0:n_S-1;
        #%%%%%%%%;
        if (f_rand> 0):
            tmp_X_index_ = efind(tmp_image_X_value_S_>=torch.quantile(tmp_image_X_value_S_,(100-100*f_rand)/100.0).item()); 
            tmp_rand = 0.5;
            if flag_deterministic==0: tmp_rand = np.random.rand(); #end;
            tmp_X_index_index = int(np.maximum(0,np.minimum(numel(tmp_X_index_)-1,np.floor(numel(tmp_X_index_)*tmp_rand))));
            nS_best = int(tmp_X_index_[tmp_X_index_index].item());
        #end;%if (f_rand> 0); 
        if (f_rand<=0):
            _,nS_best = torch.max(tmp_image_X_value_S_,dim=0);
            nS_best = nS_best.item();
        #end;%if (f_rand<=0);
        #%%%%%%%%;
        euler_polar_a_M_[nM] = viewing_polar_a_S_[nS_best].item();
        euler_azimu_b_M_[nM] = viewing_azimu_b_S_[nS_best].item();
        euler_gamma_z_M_[nM] = tmp_gamma_z_S_[nS_best].item();
        image_delta_x_M_[nM] = tmp_delta_x_S_[nS_best].item();
        image_delta_y_M_[nM] = tmp_delta_y_S_[nS_best].item();
        image_I_value_M_[nM] = tmp_I_value_S_[nS_best].item();
        image_X_value_M_[nM] = tmp_image_X_value_S_[nS_best].item();
        image_S_index_M_[nM] = nS_best;
    #end;%for nM=0:n_M-1;
    
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
