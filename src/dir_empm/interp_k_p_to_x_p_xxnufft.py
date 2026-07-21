from dir_empm.dir_matlab_macros import * ;
from dir_empm.xxnufft2d3 import xxnufft2d3 ;

def interp_k_p_to_x_p_xxnufft(
        n_x_p_r=None,
        x_p_r_=None,
        x_p_r_max=None,
        n_t_=None,
        weight_2d_x_p_tx_=None,
        n_k_p_r=None,
        k_p_r_=None,
        k_p_r_max=None,
        n_w_=None,
        weight_2d_k_p_wk_=None,
        S_k_p_wk_=None,
        finufft_opts=None,
):
    #%%%%%%%%;
    str_thisfunction = 'interp_k_p_to_x_p_xxnufft';
    flag_verbose=0;
    if (flag_verbose>0): sprintf(' %% [entering %s]',str_thisfunction); #end;

    n_t_sum = sum_0(n_t_);
    #%%%%%%%%;
    if (numel_unique(n_t_)==1):
        n_t_max = int(n_t_[0].item());
        theta_ = 2*pi*torch.arange(n_t_max).to(dtype=torch.float32)/np.maximum(1,n_t_max);
        x_p_t_tx_ = torch.tile(torch.reshape(theta_,mtr((n_t_max,1))),mtr((1,n_x_p_r))).ravel();
        x_p_r_tx_ = torch.tile(torch.reshape(x_p_r_,mtr((1,n_x_p_r))),mtr((n_t_max,1))).ravel();
        x_c_0_tx_ = x_p_r_tx_ * torch.cos(x_p_t_tx_);
        x_c_1_tx_ = x_p_r_tx_ * torch.sin(x_p_t_tx_);
    #end;%if numel(unique(n_t_)==1);
    #%%%%%%%%
    if (numel_unique(n_t_)> 1):
        n_t_sum = sum_0(n_t_[:n_x_p_r]);
        x_c_0_tx_ = torch.zeros(n_t_sum).to(dtype=torch.float32);
        x_c_1_tx_ = torch.zeros(n_t_sum).to(dtype=torch.float32);
        nt_sum=0;
        for nx_p_r in range(n_x_p_r):
            x_p_r = x_p_r_[nx_p_r].item();
            n_t = int(n_t_[nx_p_r].item());
            for nt in range(n_t):
                theta = (2.0*pi*nt)/np.maximum(1,n_t);
                x_c_0_tx_[nt_sum] = x_p_r*np.cos(theta);
                x_c_1_tx_[nt_sum] = x_p_r*np.sin(theta);
                nt_sum = nt_sum+1;
            #end;%for nt=0:n_t-1;
        #end;%for nx_p_r=0:n_x_p_r-1;
        if (nt_sum!=n_t_sum): disp(sprintf(' %% Warning, nt_sum %d vs %d',nt_sum,n_t_sum)); end;
    #end;%if numel(unique(n_t_)> 1);
    #%%%%%%%%;

    n_w_sum = sum_0(n_w_);
    #%%%%%%%%;
    if (numel_unique(n_w_)==1):
        n_w_max = int(n_w_[0].item());
        omega_ = 2*pi*torch.arange(n_w_max).to(dtype=torch.float32)/np.maximum(1,n_w_max);
        k_p_w_wk_ = torch.tile(torch.reshape(omega_,mtr((n_w_max,1))),mtr((1,n_k_p_r))).ravel();
        k_p_r_wk_ = torch.tile(torch.reshape(k_p_r_,mtr((1,n_k_p_r))),mtr((n_w_max,1))).ravel();
        k_c_0_wk_ = k_p_r_wk_ * torch.cos(k_p_w_wk_);
        k_c_1_wk_ = k_p_r_wk_ * torch.sin(k_p_w_wk_);
    #end;%if numel(unique(n_w_)==1);
    #%%%%%%%%
    if (numel_unique(n_w_)> 1):
        n_w_sum = sum_0(n_w_[:n_k_p_r]);
        k_c_0_wk_ = torch.zeros(n_w_sum).to(dtype=torch.float32);
        k_c_1_wk_ = torch.zeros(n_w_sum).to(dtype=torch.float32);
        nw_sum=0;
        for nk_p_r in range(n_k_p_r):
            k_p_r = k_p_r_[nk_p_r].item();
            n_w = int(n_w_[nk_p_r].item());
            for nw in range(n_w):
                omega = (2.0*pi*nw)/np.maximum(1,n_w);
                k_c_0_wk_[nw_sum] = k_p_r*np.cos(omega);
                k_c_1_wk_[nw_sum] = k_p_r*np.sin(omega);
                nw_sum = nw_sum+1;
            #end;%for nw=0:n_w-1;
        #end;%for nk_p_r=0:n_k_p_r-1;
        if (nw_sum!=n_w_sum): disp(sprintf(' %% Warning, nw_sum %d vs %d',nw_sum,n_w_sum)); #end;
    #end;%if numel(unique(n_w_)> 1);
    #%%%%%%%%;

    S_x_p_tx_ = torch.zeros(n_t_sum).to(dtype=torch.complex64);
    eta = 1.0; #%eta = 1/max(1e-12,sqrt(2*pi*k_p_r_max)); %eta = pi/max(1e-12,k_p_r_max);
    S_x_p_tx_ = xxnufft2d3(n_w_sum,2*pi*k_c_0_wk_*eta,2*pi*k_c_1_wk_*eta,S_k_p_wk_*weight_2d_k_p_wk_,+1,1e-12,n_t_sum,x_c_0_tx_/eta,x_c_1_tx_/eta,finufft_opts);

    if (flag_verbose>0): sprintf(' %% [finished %s]',str_thisfunction); #end;

    return S_x_p_tx_;
