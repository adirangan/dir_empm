from dir_empm.dir_matlab_macros import * ;

def rotate_p_to_p_fftw(
        n_r=None,
        n_w_=None,
        n_w_sum=None,
        S_p_=None,
        gamma_z=None,
):
    flag_verbose = 0;
    str_thisfunction = 'rotate_p_to_p_fftw' ;

    #%%%%%%%%;
    if (flag_verbose): print(f' %% [entering {str_thisfunction}]');
    #%%%%%%%%;

    n_S = int( numel(S_p_) / int(n_w_sum) );
    if numel(S_p_)!=n_w_sum*n_S: print(f' %% Warning, n_w_sum {n_w_sum} n_S {n_S} in {str_thisfunction}');

    if np.isscalar(gamma_z): gamma_z = torch.tensor([gamma_z]) ;
    n_gamma_z = numel(gamma_z);
    gamma_z_ = gamma_z.ravel();
    if (n_gamma_z> 1) & (n_S!=n_gamma_z):
        print(f' %% Warning, n_S {n_S} n_gamma_z {n_gamma_z} in {str_thisfunction}');
    #end;%if (n_gamma_z> 1) & (n_S~=n_gamma_z);

    #%%%%%%%%%%%%%%%%;
    if numel_unique(n_w_) > 1:
        M_p_ = torch.zeros(n_w_sum*n_S).to(dtype=torch.complex64);
        #%%%%%%%%;
        if n_S==1:
            #%%%%;
            #% single transformation on adaptive grid. ;
            #%%%%;
            n_w_max = int(n_w_[n_r-1].item());
            C_ = torch.zeros(n_w_max).to(dtype=torch.complex64);
            for nw in range(n_w_max):
                q = nw - n_w_max/2;
                C = np.cos(q*gamma) - i*np.sin(q*gamma);
                C_[nw] = C;
            #end;%for nw=0:n_w_max-1;
            nw_sum = 0;
            for nr in range(n_r):
                if n_w_[nr] > 0:
                    fftw_out_ = torch.fft.fft(S_p_[nw_sum:nw_sum + n_w_[nr]]);
                    for nq in range(n_w_[nr]):
                        q = nq;
                        if q> n_w_[nr].item()/2 - 1:  q = q - n_w_[nr];
                        nw = int(np.floor(n_w_max/2 + q));
                        C = C_[nw].item()
                        fftw_out_[nq] = fftw_out_[nq]*C;
                    #end;%for nq=0:n_w_(1+nr)-1;
                    fftw_0in_ = torch.fft.ifft(fftw_out_);
                    M_p_[nw_sum:nw_sum + n_w_[nr]] = torch.tensor(fftw_0in_).to(dtype=torch.complex64);
                    nw_sum = nw_sum+n_w_[nr];
                #end;%if (n_w_(1+nr)>0);
            #end;%for nr=0:n_r-1;
            assert(nw_sum==n_w_sum);
        #end;%if n_S==1;
        #%%%%%%%%;
        if n_S> 1:
            for nS in range(n_S):
                tmp_index_ = int(nS*n_w_sum) + torch.arange(n_w_sum).to(dtype=torch.int32);
                if n_gamma_z==1:
                    #M_p_.ravel()[tmp_index_] = rotate_p_to_p_fftw(n_r,n_w_,n_w_sum,S_p_.ravel()[tmp_index_],gamma_z_[0].item());
                    M_p_ = matlab_assign_lhs_from_rhs_(M_p_,tmp_index_, rotate_p_to_p_fftw(n_r,n_w_,n_w_sum,S_p_.ravel()[tmp_index_],gamma_z_[0].item()));
                #end;%if n_gamma_z==1;
                if n_gamma_z> 1:
                    #M_p_.ravel()[tmp_index_] = rotate_p_to_p_fftw(n_r,n_w_,n_w_sum,S_p_.ravel()[tmp_index_],gamma_z_[nS].item());
                    M_p_ = matlab_assign_lhs_from_rhs_(M_p_,tmp_index_, rotate_p_to_p_fftw(n_r,n_w_,n_w_sum,S_p_.ravel()[tmp_index_],gamma_z_[nS].item()));
                #end;%if n_gamma_z> 1;
            #end;%for nS=0:n_S-1;
        #end;%if n_S> 1;
        #%%%%%%%%;
    #end;%if (numel(unique(n_w_))> 1);
    #%%%%%%%%%%%%%%%%;

    #%%%%%%%%%%%%%%%%;
    if numel_unique(n_w_)==1:
        n_w = int(n_w_[0].item()); n_w_max = n_w;
        tmp_q_ = torch.concatenate( ( torch.arange(0,n_w_max/2) , torch.arange(-n_w_max/2,-1+1) ) , 0 ).to(dtype=torch.int32);
        C_wz__ = torch.exp(-i*tmp_q_*torch.reshape(gamma_z_,mtr((1,n_gamma_z)))).to(dtype=torch.complex64);
        M_p_ = torch.fft.ifft( torch.reshape(C_wz__,mtr((n_w,1,n_gamma_z))) * torch.fft.fft(torch.reshape(S_p_,mtr((n_w,n_r,n_S))),dim=2-0) , dim=2-0 ).to(dtype=torch.complex64).ravel();
        assert(numel(M_p_)==numel(S_p_));
    #end;%if (numel(unique(n_w_))==1);
    #%%%%%%%%%%%%%%%%;

    #%%%%%%%%;
    if (flag_verbose): print(f' %% [finished {str_thisfunction}]');
    #%%%%%%%%;
    return(M_p_);
