import numpy as np ; import torch ;

#%%%%%%%%;
def matlab_index_prebroadcast_(n,index_):
    if index_ is None: return(torch.arange(0,n)); #end;
    if isinstance(index_,str):
        if index_==':': return(torch.arange(0,n)); #end;
        raise ValueError(index_);
    #end;%if isinstance(index_,str);
    if torch.is_tensor(index_):
        if index_.ndim>1: raise ValueError('matlab_index_prebroadcast_: expecting linear indices'); #end;
        return(index_.to(dtype=torch.int64).reshape(-1));
    #end;%if torch.is_tensor(index_);
    if np.isscalar(index_): return(torch.tensor([index_],dtype=torch.int64)); #end;
    return(index_.to(dtype=torch.int64));
#%%%%%%%%;

#%%%%%%%%;
def matlab_index_prebroadcast_gpu_(device_use,n,index_):
    if index_ is None: return(torch.arange(0,n,device=device_use)); #end;
    if isinstance(index_,str):
        if index_==':': return(torch.arange(0,n,device=device_use)); #end;
        raise ValueError(index_);
    #end;%if isinstance(index_,str);
    if torch.is_tensor(index_):
        if index_.ndim>1: raise ValueError('matlab_index_prebroadcast_gpu_: expecting linear indices'); #end;
        return(index_.to(dtype=torch.int64,device=device_use).reshape(-1));
    if np.isscalar(index_): return(torch.tensor([index_],dtype=torch.int64,device=device_use)); #end;
    return(index_.to(dtype=torch.int64,device=device_use));
#%%%%%%%%;

#%%%%%%%%;
def matlab_index_size_(A,n_d):
    # Note that extra dimensions (after true size) are padded with ones. ;
    n_ = list(reversed(tuple(A.shape)));
    if len(n_)< n_d: n_ = n_ + [1]*(n_d-len(n_)); #end;
    if len(n_)> n_d: n_ = n_[:(n_d-1)] + [int(np.prod(n_[(n_d-1):]))]; #end;
    return(n_);
#%%%%%%%%;

#%%%%%%%%;
def matlab_index_args_(n_d,args):
    # Here we allow for two different calling sequences. ;
    # A: args is (n_0,index_n0_,n_1,index_n1_,...) or ;
    # B: args is (A,index_n0_,index_n1_,...). ;
    # A and B are differentiated by the total number of arguments. ;
    if len(args)==n_d+1 and torch.is_tensor(args[0]) and args[0].ndim>=1:
        n_ = matlab_index_size_(args[0],n_d);
        return([x for n,index_ in zip(n_,args[1:]) for x in (n,index_)]);
    if len(args)> 2*n_d: raise ValueError('matlab_index_args_ called with too many arguments'); #end;
    return(list(args)+[None]*(2*n_d-len(args)));
#%%%%%%%%;
