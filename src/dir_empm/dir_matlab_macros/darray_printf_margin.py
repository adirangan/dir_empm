import numpy as np ;
from . disp_sprintf import disp; from . disp_sprintf import sprintf;
numel = lambda a : int(a.numel()) ;
isempty = lambda t: (t is None) or (isinstance(t, torch.Tensor) and t.numel() == 0) ;

def darray_printf_margin(
        d_=None,
        n_r=None,
        n_c=None,
        str_prefix='',
        n_margin=3,
):
    if ((n_r is None) | (n_c is None)):
        n_r = 1; n_c = numel(d_);
    #end;%if (isempty(n_r) | isempty(n_c)):
    if (str_prefix is None): str_prefix = ''; #end;
    if (n_margin is None): n_margin = 3; #end;

    n_mid = np.minimum((8+1)*n_c,(8+1)*(2*n_margin) + 4);
    str_mid = '.'*(n_mid);

    nr=0; nc=0;
    nr=0;
    while(nr<n_r):
        if ( (nr>n_margin-1) & (nr<n_r-n_margin) ):
            disp(sprintf("%s%s",str_prefix,str_mid));
            nr = np.maximum(0,n_r-n_margin);
        #end;%if ( (nr>n_margin-1) & (nr<n_r-n_margin) );
        if ( (nr<n_margin) | (nr>n_r-n_margin-1) ):
            tmp_str= sprintf("%s",str_prefix);
            nc=0;
            while(nc<n_c):
                if ( (nc>n_margin-1) & (nc<n_c-n_margin) ):
                    tmp_str = sprintf("%s... ",tmp_str);
                    nc = np.maximum(0,n_c-n_margin);
                #end;%if ( (nc>n_margin-1) & (nc<n_c-n_margin) );
                if ( (nc<n_margin) | (nc>n_c-n_margin-1) ):
                    tmp_str = sprintf("%s%+8.2f ",tmp_str,d_.ravel()[nr+nc*n_r]);
                #end;%if ( (nc<n_margin) | (nc>n_c-n_margin-1) );
                nc=nc+1;
            #end;%while(nc<n_c);
            disp(sprintf("%s",tmp_str));
        #end;%if ( (nr<n_margin) | (nr>n_r-n_margin-1) );
        nr=nr+1;
    #end;%while(nr<n_r);
