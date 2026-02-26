from dir_empm.dir_matlab_macros import * ;

def r8_Jp_chebcoef_0(
        parameter=None,
        n_k=None,
):
    str_thisfunction = 'r8_Jp_chebcoef_0' ;

    if not isinstance(parameter, dict):
        parameter = {'type': 'parameter'}
    if 'flag_verbose' not in parameter:
        parameter['flag_verbose'] = 0
    flag_verbose = parameter['flag_verbose']

    #%%%%%%%%;
    if (flag_verbose): print(f' %% [entering {str_thisfunction}]');
    #%%%%%%%%;

    n_j = n_k;
    r8_Jp_jk__ = torch.zeros(mtr((n_k,n_k))).to(dtype=torch.float64);
    for nj in range(n_j):
        if nj==0:
            nk=0; r8_Jp_jk__[nk,nj] = 1.0; #%<-- default float64. ;
        if nj==1: 
            nk=0; r8_Jp_jk__[nk,nj] = -0.5; nk=1; r8_Jp_jk__[nk,nj] = +1.5; #%<-- default float64. ;
        if nj>=2:
            nj1 = nj-1; nj2 = nj-2;
            for nk in range(nj+1):
                nkc = nk; nkp = nkc+1; nkn = nkc-1;
                r8_J = 0.0;
                r8_J = r8_J - 1.0/(nj+1)/(2*nj-1) * r8_Jp_jk__[nkc,nj1].item();
                r8_J = r8_J - ((nj-1)/(nj+1))*((2*nj+1)/(2*nj-1)) * r8_Jp_jk__[nkc,nj2].item();
                if nkp<=n_k-1:
                    r8_J = r8_J + ((2*nj+1)/(2*nj+2)) * r8_Jp_jk__[nkp,nj1].item();
                #end;%if nkp<=n_k-1;
                if nkn>=0:
                    r8_J = r8_J + ((2*nj+1)/(2*nj+2)) * r8_Jp_jk__[nkn,nj1].item();
                #end;%if nkn>=0;
                if nk==1:
                    r8_J = r8_J + ((2*nj+1)/(2*nj+2)) * r8_Jp_jk__[nkn,nj1].item(); #%<-- here is the doubling of the J_{j-1,0} term. ;
                #end;%if nk==1;
                r8_Jp_jk__[nk,nj] = r8_J;
            #end;%for nk=0:nj;
        #end;%if nj>=2;
    #end;%for nj=0:n_j-1;

    #%%%%%%%%;
    if (flag_verbose): print(f' %% [finished {str_thisfunction}]');
    #%%%%%%%%;
    return(
        parameter,
        r8_Jp_jk__,
        );

