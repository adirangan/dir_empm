from dir_empm.dir_matlab_macros import * ;

def tfpmh_MS_uniform_2(
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
    str_thisfunction = 'tfpmh_MS_uniform_2';
    #%%%%%%%%;

    if isempty(parameter): parameter = {'type':'parameter'}; #end;%if isempty(parameter);
    if 'flag_verbose' not in parameter: parameter['flag_verbose'] = 0; #end;
    flag_verbose = parameter['flag_verbose'];
    if 'flag_deterministic' not in parameter: parameter['flag_deterministic'] = 0; #end;
    flag_deterministic = parameter['flag_deterministic'];
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
    index_nS_permutation_ = torch.arange(n_S).to(dtype=torch.int32);
    if flag_deterministic==0: index_nS_permutation_ = torch.randperm(n_S); #end;
    #%%%%%%%%;
    flag_M_used_ = torch.zeros(n_M).to(dtype=torch.int32);
    tmp_nS=0; nM_sum=0;
    while (int(torch.sum(flag_M_used_).item())<n_M):
        index_M_unused_ = efind(flag_M_used_==0);
        n_index_M_unused = numel(index_M_unused_);
        index_nS = int(index_nS_permutation_[tmp_nS].item());
        if (ndims(X_wSM___)==3):
            tmp_i8_index_rhs_ = matlab_index_3d_0(n_w_max,':',n_S,index_nS,n_M,index_M_unused_);
            image_X_value,index_wM_best = torch.max(torch.real(X_wSM___.ravel()[tmp_i8_index_rhs_]),dim=0);
            image_X_value = image_X_value.item(); index_wM_best = int(index_wM_best.item());
            nw_best = int(np.mod(index_wM_best,n_w_max));
            index_M_best = int(matlab_scalar_round((index_wM_best - nw_best)/np.maximum(1,n_w_max)));
            nM_best = int(index_M_unused_[index_M_best].item());
            gamma_z_best = 2*pi*nw_best/np.maximum(1,n_w_max); #%<-- gamma_z_wSM__ unnecessary. ;
            delta_x_best = delta_x_wSM___[nM_best,index_nS,nw_best].item();
            delta_y_best = delta_y_wSM___[nM_best,index_nS,nw_best].item();
            I_value_best = I_value_wSM___[nM_best,index_nS,nw_best].item();
        #end;%if (ndims(X_wSM___)==3);
        if (ndims(X_wSM___)==2):
            tmp_i8_index_rhs_ = matlab_index_2d_0(n_S,index_nS,n_M,index_M_unused_);
            image_X_value,index_M_best = torch.max(torch.real(X_wSM___.ravel()[tmp_i8_index_rhs_]),dim=0);
            image_X_value = image_X_value.item(); index_M_best = int(index_M_best.item());
            nM_best = int(index_M_unused_[index_M_best].item());
            gamma_z_best = gamma_z_wSM___[nM_best,index_nS].item();
            delta_x_best = delta_x_wSM___[nM_best,index_nS].item();
            delta_y_best = delta_y_wSM___[nM_best,index_nS].item();
            I_value_best = I_value_wSM___[nM_best,index_nS].item();
        #end;%if (ndims(X_wSM___)==2);
        flag_M_used_[nM_best]=1;
        euler_polar_a_M_[nM_best] = viewing_polar_a_S_[index_nS].item();
        euler_azimu_b_M_[nM_best] = viewing_azimu_b_S_[index_nS].item();
        euler_gamma_z_M_[nM_best] = gamma_z_best;
        image_delta_x_M_[nM_best] = delta_x_best;
        image_delta_y_M_[nM_best] = delta_y_best;
        image_I_value_M_[nM_best] = I_value_best;
        image_X_value_M_[nM_best] = image_X_value;
        image_S_index_M_[nM_best] = index_nS;
        tmp_nS=tmp_nS+1;
        if (tmp_nS>=n_S): tmp_nS=0; #end;
        nM_sum=nM_sum+1;
    #end;%while (sum(flag_M_used_)<n_M);
    assert(nM_sum==n_M);
    
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
