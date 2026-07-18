from dir_empm.dir_matlab_macros import * ;

def interp_q_to_p(
        n_r=None,
        n_w_=None,
        n_A=None,
        S_q_=None,
):
    flag_verbose = 0;
    str_thisfunction = 'interp_q_to_p' ;

    #%%%%%%%%;
    if (flag_verbose): print(f' %% [entering {str_thisfunction}]');
    #%%%%%%%%;

    n_S = int( numel(S_q_) / int(n_A) );
    if numel(S_q_)!=n_A*n_S: print(f' %% Warning, n_A {n_A} n_S {n_S} in {str_thisfunction}');

    #%%%%%%%%%%%%%%%%;
    if numel_unique(n_w_) > 1:
        S_p_ = torch.zeros(n_A*n_S).to(dtype=torch.complex64);
        #%%%%%%%%;
        if n_S==1:
            #%%%%;
            #% single transformation on adaptive grid. ;
            #%%%%;
            n_w_max = int(n_w_[n_r-1].item());
            ic=0;
            for nr in range(n_r):
                n_w = int(n_w_[nr]).item();
                if (n_w>0):
                    tmp_index_ = int(ic) + torch.arange(n_w).to(dtype=torch.int32);
                    S_p_[tmp_index_] = torch.tensor(np.fft.ifft(S_q_[tmp_index_].numpy())*np.maximum(0,np.sqrt(n_w))).to(dtype=torch.complex64);
                    ic = ic + n_w;
                #end;%if (n_w>0);
            #end;%for nr=0:n_r-1;
            assert(ic==n_A);
            #%%%%;
        #end;%if n_S==1;
        #%%%%%%%%;
        if n_S> 1:
            for nS in range(n_S):
                tmp_index_ = int(nS*n_A) + torch.arange(n_A).to(dtype=torch.int32);
                S_p_[tmp_index_] = interp_q_to_p(n_r,n_w_,n_A,S_q_[tmp_index_]);
            #end;%for nS=0:n_S-1;
        #end;%if n_S> 1;
        #%%%%%%%%;
    #end;%if (numel(unique(n_w_))> 1);
    #%%%%%%%%%%%%%%%%;

    #%%%%%%%%%%%%%%%%;
    if numel_unique(n_w_)==1:
        n_w = int(n_w_[0].item());
        S_p_ = torch.fft.ifft(torch.reshape(S_q_,mtr((n_w,n_r,n_S))),axis=2-0).to(dtype=torch.complex64).ravel() * np.maximum(0,np.sqrt(n_w));
        assert(numel(S_p_)==numel(S_q_));
    #end;%if (numel(unique(n_w_))==1);
    #%%%%%%%%%%%%%%%%;

    #%%%%%%%%;
    if (flag_verbose): print(f' %% [finished {str_thisfunction}]');
    #%%%%%%%%;
    return(S_p_);


