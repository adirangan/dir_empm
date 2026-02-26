from dir_empm.dir_matlab_macros import * ;
from dir_empm.ylgndr_2 import ylgndr_2 ;

def A1(nl,nmn,nmp):
    numerator   = (nl+nmn)*(nl+nmn-1);
    denominator = (nl+nmp)*(nl+nmp-1);
    output = np.sqrt(numerator/denominator);
    return(output);

def A2(nl,nmn,nmp):
    numerator   = (nl+nmn)*(nl-nmn);
    denominator = (nl+nmp)*(nl+nmp-1);
    output = np.sqrt(numerator/denominator);
    return(output);

def A3(nl,nmn,nmp):
    numerator   = (nl-nmn)*(nl-nmn-1);
    denominator = (nl+nmp)*(nl+nmp-1);
    output = np.sqrt(numerator/denominator);
    return(output);

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def B1(nl,nmn,nmp):
    numerator   = (nl+nmn)*(nl+nmn-1);
    denominator = (nl-nmp)*(nl-nmp-1);
    output = np.sqrt(numerator/denominator);
    return(output);

def B2(nl,nmn,nmp):
    numerator   = (nl+nmn)*(nl-nmn);
    denominator = (nl-nmp)*(nl-nmp-1);
    output = np.sqrt(numerator/denominator);
    return(output);

def B3(nl,nmn,nmp):
    numerator   = (nl-nmn)*(nl-nmn-1);
    denominator = (nl-nmp)*(nl-nmp-1);
    output = np.sqrt(numerator/denominator);
    return(output);

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def C1(nl,nmn,nmp):
    numerator   = (nl+nmn)*(nl+nmn-1);
    denominator = (nl-nmp)*(nl+nmp);
    output = np.sqrt(numerator/denominator);
    return(output);

def C2(nl,nmn,nmp):
    numerator   = (nl+nmn)*(nl-nmn);
    denominator = (nl-nmp)*(nl+nmp);
    output = np.sqrt(numerator/denominator);
    return(output);

def C3(nl,nmn,nmp):
    #%numerator   = (nl-nmn)*(nl-nmn+1);
    numerator   = (nl-nmn)*(nl-nmn-1);
    denominator = (nl-nmp)*(nl+nmp);
    output = np.sqrt(numerator/denominator);
    return(output);

def wignerd_b(
        n_l=None,
        beta=None,
):
    #% Generates wigner-d matrices up to n_l;
    #% Note: uses 3-term recurrences, which are unstable after n_l==88 or so. ;
    #% Fiddles with condon-shortley phase (c.f. wignerd). ;

    if (n_l>=88): print(f' %% Warning, n_l={n_l}>=88 in wignerd_b');
    cb = np.cos(beta/2); sb = np.sin(beta/2); 
    W_ = cell(1+n_l); #<-- cell array. ;
    W_[0] = torch.ones(mtr((1,1))).to(dtype=torch.float32);
    for nl in range(1,n_l+1):
        nlp = nl-1;
        V = W_[nl-1]; Vt = V.T;
        W = torch.zeros(mtr((2*nl+1,2*nl+1))).to(dtype=torch.float32);
        for nmn in range(-nl,+nl+1):
            for nmp in range(-nl,+nl+1):
                #%%%%%%%%%%%%%%%%;
                if (nl!=0-nmp and nl!=1-nmp): #% use recurrence A ;
                    str_tmp = 'A';
                    W1 = 0;
                    if (np.abs(nmp-1)<=nlp and np.abs(nmn-1)<=nlp): W1 = Vt[nlp+(nmp-1),nlp+(nmn-1)];
                    W2 = 0;
                    if (np.abs(nmp-1)<=nlp and np.abs(nmn-0)<=nlp): W2 = Vt[nlp+(nmp-1),nlp+(nmn-0)];
                    W3 = 0;
                    if (np.abs(nmp-1)<=nlp and np.abs(nmn+1)<=nlp): W3 = Vt[nlp+(nmp-1),nlp+(nmn+1)];
                    tmp = cb*cb*A1(nl,nmn,nmp)*W1 - 2*cb*sb*A2(nl,nmn,nmp)*W2 + sb*sb*A3(nl,nmn,nmp)*W3;
                #end;%if (nl~=-nmp and nl~=1-nmp); % use recurrence A ;
                #%%%%%%%%%%%%%%%%;
                if (nl!=0+nmp and nl!=1+nmp): #% use recurrence B ;
                    str_tmp = 'B';
                    W1 = 0;
                    if (np.abs(nmp+1)<=nlp and np.abs(nmn-1)<=nlp): W1 = Vt[nlp+(nmp+1),nlp+(nmn-1)];
                    W2 = 0;
                    if (np.abs(nmp+1)<=nlp and np.abs(nmn-0)<=nlp): W2 = Vt[nlp+(nmp+1),nlp+(nmn-0)];
                    W3 = 0;
                    if (np.abs(nmp+1)<=nlp and np.abs(nmn+1)<=nlp): W3 = Vt[nlp+(nmp+1),nlp+(nmn+1)];
                    tmp = sb*sb*B1(nl,nmn,nmp)*W1 + 2*sb*cb*B2(nl,nmn,nmp)*W2 + cb*cb*B3(nl,nmn,nmp)*W3;
                #end;%if (nl~=+nmp and nl~=1+nmp); % use recurrence B ;
                #%%%%%%%%%%%%%%%%;
                if (nl!=0-nmp and nl!=0+nmp): #% use recurrence C ;
                    str_tmp = 'C';
                    W1 = 0;
                    if (np.abs(nmp-0)<=nlp and np.abs(nmn-1)<=nlp): W1 = Vt[nlp+(nmp-0),nlp+(nmn-1)];
                    W2 = 0;
                    if (np.abs(nmp-0)<=nlp and np.abs(nmn-0)<=nlp): W2 = Vt[nlp+(nmp-0),nlp+(nmn-0)];
                    W3 = 0;
                    if (np.abs(nmp-0)<=nlp and np.abs(nmn+1)<=nlp): W3 = Vt[nlp+(nmp-0),nlp+(nmn+1)];
                    tmp = sb*cb*C1(nl,nmn,nmp)*W1 + (cb*cb-sb*sb)*C2(nl,nmn,nmp)*W2 - sb*cb*C3(nl,nmn,nmp)*W3;
                #end;%if (nl~=-nmp and nl~=+nmp); % use recurrence C ;
                #%%%%%%%%%%%%%%%%;
                tmp_i8_index_rhs_ = matlab_index_2d_0(2*nl+1,nl+nmp,2*nl+1,nl+nmn);
                W.ravel()[tmp_i8_index_rhs_] = tmp;
            #end;for nmp = -nl:+nl;
        #end;%for nmn = -nl:+nl; 
        W_[nl] = W;
    #end;%for nl=1:n_l;

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;
    #% Fix condon-shortley-phase ;
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;
    for nl in range(1,n_l+1):
        m_ = torch.arange(-nl,+nl+1).to(dtype=torch.int32);
        s_=(-1)**((m_<0)*m_); #% needed to preserve condon-shortley phase. ;
        S__ = torch.reshape(s_,mtr((2*nl+1,1))) * torch.reshape(s_,mtr((1,2*nl+1))) ;
        W_[nl] = W_[nl]*S__;
    #end;%for nl=1:n_l;

    return(W_);


