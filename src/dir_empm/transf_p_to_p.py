from dir_empm.dir_matlab_macros import * ;

#%%%%%%%%;
#% Note that if delta_x and delta_y are given as arrays, ;
#% then we expect n_S==n_delta_v. ;
#% Thus, this 'per-template' translation is not appropriate ;
#% for the construction of an array of the form M_wkdM___ ;
#% (as might be required for the FTK). ;
#%%%%%%%%;

def transf_p_to_p(
        n_r=None,
        grid_p_=None,
        n_w_=None,
        n_w_sum=None,
        S_p_=None,
        delta_x=None,
        delta_y=None,
):
    flag_verbose = 0;
    str_thisfunction = 'transf_p_to_p' ;

    #%%%%%%%%;
    if (flag_verbose): print(f' %% [entering {str_thisfunction}]');
    #%%%%%%%%;

    n_S = int( numel(S_p_) / int(n_w_sum) );
    if numel(S_p_)!=n_w_sum*n_S: print(f' %% Warning, n_w_sum {n_w_sum} n_S {n_S} in {str_thisfunction}');

    if np.isscalar(delta_x): delta_x = torch.tensor([delta_x]) ;
    if np.isscalar(delta_y): delta_y = torch.tensor([delta_y]) ;
    assert(numel(delta_y)==numel(delta_x));
    n_delta_v = numel(delta_x);
    delta_x_ = delta_x.ravel();
    delta_y_ = delta_y.ravel();
    if (n_delta_v> 1) & (n_S!=n_delta_v):
        print(f' %% Warning, n_S {n_S} n_delta_v {n_delta_v} in {str_thisfunction}');
    #end;%if (n_delta_v> 1) & (n_S~=n_delta_v);

    #%%%%%%%%%%%%%%%%;
    if numel_unique(n_w_) > 1:
        M_p_ = torch.zeros(n_w_sum*n_S).to(dtype=torch.complex64);
        #%%%%%%%%;
        if n_S==1:
            #%%%%;
            #% single transformation on adaptive grid. ;
            #%%%%;
            nw_sum=0;
            for nr in range(n_r):
                R_c = grid_p_[nr].item();
                n_w = int(n_w_[nr].item());
                for nw in range(n_w):
                    W_c = 0.0 + nw*(2*pi)/np.maximum(1,n_w);
                    X_c = R_c*np.cos(W_c);
                    Y_c = R_c*np.sin(W_c);
                    L_c = (X_c * delta_x_[0].item()) + (Y_c * delta_y_[0].item());
                    C_c = +np.cos(2*pi*L_c) - i*np.sin(2*pi*L_c);
                    M_p_[nw_sum] = C_c*S_p_[nw_sum];
                    nw_sum = nw_sum + 1;
                #end;%for nw=0:n_w-1;
            #end;%for nr=0:n_r-1;
            assert(nw_sum==n_w_sum);
        #end;%if n_S==1;
        #%%%%%%%%;
        if n_S> 1:
            for nS in range(n_S):
                tmp_index_ = int(nS*n_w_sum) + torch.arange(n_w_sum).to(dtype=torch.int32);
                if n_delta_v==1:
                    #M_p_.ravel()[tmp_index_] = transf_p_to_p(n_r,grid_p_,n_w_,n_w_sum,S_p_.ravel()[tmp_index_],delta_x_[0].item(),delta_y_[0].item());
                    M_p_ = matlab_assign_lhs_from_rhs_(M_p_,tmp_index_, transf_p_to_p(n_r,grid_p_,n_w_,n_w_sum,S_p_.ravel()[tmp_index_],delta_x_[0].item(),delta_y_[0].item()));
                #end;%if n_delta_v==1;
                if n_delta_v> 1:
                    #M_p_.ravel()[tmp_index_] = transf_p_to_p(n_r,grid_p_,n_w_,n_w_sum,S_p_.ravel()[tmp_index_],delta_x_[nS].item(),delta_y_[nS].item());
                    M_p_ = matlab_assign_lhs_from_rhs_(M_p_,tmp_index_, transf_p_to_p(n_r,grid_p_,n_w_,n_w_sum,S_p_.ravel()[tmp_index_],delta_x_[nS].item(),delta_y_[nS].item()));
                #end;%if n_delta_v> 1;
            #end;%for nS=0:n_S-1;
        #end;%if n_S> 1;
        #%%%%%%%%;
    #end;%if (numel(unique(n_w_))> 1);
    #%%%%%%%%%%%%%%%%;

    #%%%%%%%%%%%%%%%%;
    if numel_unique(n_w_)==1:
        n_w = int(n_w_[0].item()); n_w_max = n_w;
        gamma_z_ = torch.linspace(0,2*pi,n_w_max+1).to(dtype=torch.float32); gamma_z_ = gamma_z_[torch.arange(n_w_max)];
        k_c_0_wk_ = ( torch.reshape(+torch.cos(gamma_z_),mtr((n_w_max,1))) * torch.reshape(grid_p_,mtr((1,n_r))) ).ravel() ;
        assert(numel(k_c_0_wk_)==n_w_sum);
        k_c_1_wk_ = ( torch.reshape(+torch.sin(gamma_z_),mtr((n_w_max,1))) * torch.reshape(grid_p_,mtr((1,n_r))) ).ravel();
        assert(numel(k_c_1_wk_)==n_w_sum);
        L_c_wkv__ = k_c_0_wk_*torch.reshape(delta_x_,mtr((1,n_delta_v))) + k_c_1_wk_*torch.reshape(delta_y_,mtr((1,n_delta_v))) ;
        C_c_wkv__ = torch.exp(-i*2*pi*L_c_wkv__).to(dtype=torch.complex64);
        M_p_ = ( C_c_wkv__ * torch.reshape(S_p_,mtr((n_w_sum,n_S))) ).to(dtype=torch.complex64).ravel();
        assert(numel(M_p_)==numel(S_p_));
    #end;%if (numel(unique(n_w_))==1);
    #%%%%%%%%%%%%%%%%;

    #%%%%%%%%;
    if (flag_verbose): print(f' %% [finished {str_thisfunction}]');
    #%%%%%%%%;
    return(M_p_);


