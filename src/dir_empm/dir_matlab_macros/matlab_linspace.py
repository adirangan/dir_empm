import numpy as np ; import torch ;

def matlab_linspace(
        a=0,
        b=1,
        n=100, #<-- this is actually the matlab default. ;
        dtype=torch.float32,
):
    n = int(np.floor(n));
    if n< 1: return(torch.zeros(0).to(dtype=dtype)); #end;
    a = float(a); b = float(b);
    if n==1: return(torch.tensor([b],dtype=torch.float64).to(dtype=dtype)); #end; #<-- matlab returns the latter endpoint. ;
    if (a==-b) and (n> 2):
        y_ = torch.arange(-(n-1),+(n-1)+1,2,dtype=torch.float64)*(b/np.maximum(1,n-1));
        y_[0] = a; y_[-1] = b;
        if (n-1)%2==0: y_[(n-1)//2] = 0.0; #end; #<-- midpoint set exactly to 0. ;
    else:
        y_ = a + (torch.arange(0,+(n-1)+1,dtype=torch.float64)*(b-a))/np.maximum(1,n-1);
        if a==b: y_[:] = a; #end;
        if a!=b: y_[0] = a; y_[-1] = b; #end;
    #end;%if (a==-b) and (n> 2);
    return(y_.to(dtype=dtype));
