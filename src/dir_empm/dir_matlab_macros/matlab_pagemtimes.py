import numpy as np ; import torch ;

def matlab_pagemtimes(A, *args):
    if len(args)==1: str_transp_A = 'none'; B = args[0]; str_transp_B = 'none'; #end;
    if len(args)==3: str_transp_A, B, str_transp_B = args; #end;
    if len(args)!=1 and len(args)!=3: raise ValueError('matlab_pagemtimes: inputs are (A,B) or (A,str_transp_A,B,str_transp_B)'); #end;
    def fun_transp(X, str_transp):
        if str_transp==      'none': return X;
        if str_transp== 'transpose': return torch.transpose(X,-1,-2);
        if str_transp=='ctranspose': return torch.conj(torch.transpose(X,-1,-2));
        raise ValueError('matlab_pagemtimes: fun_transp: unknown option for str_transp');
    #end;%def fun_transp;
    if A.ndim<2: A = torch.reshape(A,(1,)+tuple(A.shape)) ; #<-- for column-vectors to be n-by-1. ;
    if B.ndim<2: B = torch.reshape(B,(1,)+tuple(B.shape)) ; #<-- for column-vectors to be n-by-1. ;
    tmp_dtype = torch.promote_types(A.dtype,B.dtype);
    return torch.matmul(fun_transp(B.to(dtype=tmp_dtype),str_transp_B), fun_transp(A.to(dtype=tmp_dtype),str_transp_A)) ; #<-- similar to pagemtimes. ;
