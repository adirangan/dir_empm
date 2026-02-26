from dir_empm.dir_matlab_macros import * ;

def get_weight_Y_0(
        flag_verbose=None,
        n_k_p_r=None,
        k_p_r_=None,
        k_p_r_max=None,
        weight_3d_k_p_r_=None,
):

    str_thisfunction = 'get_weight_Y_0';

    if flag_verbose is None:  flag_verbose = 0; #end;
    
    if (flag_verbose>0): disp(sprintf(' %% [entering %s]',str_thisfunction)); #end;
    
    l_max_upb = matlab_scalar_round(2*pi*k_p_r_max); #%<-- typically sufficient for 2-3 digits of precision. ;
    l_max_ = torch.zeros(n_k_p_r).to(dtype=torch.int32);
    for nk_p_r in range(n_k_p_r):
        l_max_[nk_p_r] = int(np.maximum(0,np.minimum(l_max_upb,1+np.ceil(2*pi*k_p_r_[nk_p_r].item()))));
    #end;%for nk_p_r=0:n_k_p_r-1;
    n_ml_ = (l_max_+1)**2;
    n_ml_max = int(torch.max(n_ml_).item());
    n_ml_sum = int(torch.sum(n_ml_).item());
    n_ml_csum_ = cumsum_0(n_ml_);
    l_max_max = int(torch.max(l_max_).item());
    m_max_ = torch.arange(-l_max_max,+l_max_max+1).to(dtype=torch.int32);
    n_m_max = numel(m_max_);
    Y_l_val_yk_ = torch.zeros(n_ml_sum).to(dtype=torch.int32);
    Y_m_val_yk_ = torch.zeros(n_ml_sum).to(dtype=torch.int32);
    Y_k_val_yk_ = torch.zeros(n_ml_sum).to(dtype=torch.float32);
    for nk_p_r in range(n_k_p_r):
        l_max = int(l_max_[nk_p_r].item());
        n_ml = int(n_ml_[nk_p_r].item());
        tmp_l_val_y_ = torch.zeros(n_ml).to(dtype=torch.int32);
        tmp_m_val_y_ = torch.zeros(n_ml).to(dtype=torch.int32);
        nml=0; 
        for l_val in range(l_max+1):
            for m_val in range(-l_val,+l_val+1):
                tmp_l_val_y_[nml] = l_val;
                tmp_m_val_y_[nml] = m_val;
                nml=nml+1;
            #end;%for m_val=-l_val:+l_val;
        #end;%for l_val=0:l_max;
        assert(nml==n_ml); assert(nml==(l_max+1)**2);
        tmp_index_ = int(n_ml_csum_[nk_p_r].item()) + torch.arange(n_ml).to(dtype=torch.int32);
        Y_l_val_yk_[tmp_index_] = tmp_l_val_y_;
        Y_m_val_yk_[tmp_index_] = tmp_m_val_y_;
        Y_k_val_yk_[tmp_index_] = k_p_r_[nk_p_r].item() * torch.ones(n_ml).to(dtype=torch.float32);
    #end;%for nk_p_r=0:n_k_p_r-1;
    weight_Y_yk_ = torch.zeros(n_ml_sum).to(dtype=torch.float32);
    for nk_p_r in range(n_k_p_r):
        tmp_index_ = int(n_ml_csum_[nk_p_r].item()) + torch.arange(n_ml).to(dtype=torch.int32);
        weight_Y_yk_[tmp_index_] = weight_3d_k_p_r_[nk_p_r].item();
    #end;%for nk_p_r=0:n_k_p_r-1;

    if (flag_verbose>0): disp(sprintf(' %% [finished %s]',str_thisfunction)); #end;

    return (
        l_max_upb,
        l_max_,
        l_max_max,
        n_m_max,
        m_max_,
        n_ml_,
        n_ml_max,
        n_ml_sum,
        n_ml_csum_,
        Y_l_val_yk_,
        Y_m_val_yk_,
        Y_k_val_yk_,
        weight_Y_yk_,
    ) ;
