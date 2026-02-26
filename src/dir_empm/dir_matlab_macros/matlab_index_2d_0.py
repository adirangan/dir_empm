import numpy as np ; import torch ;

def matlab_index_2d_0(
        n_0=None,
        index_n0_=None,
        n_1=None,
        index_n1_=None,
):
      str_thisfunction = 'matlab_index_2d_0';
      if index_n0_ is None: index_n0_ = torch.arange(0,n_0);
      if index_n1_ is None: index_n1_ = torch.arange(0,n_1);
      if index_n0_==':': index_n0_ = torch.arange(0,n_0);
      if index_n1_==':': index_n1_ = torch.arange(0,n_1);
      if np.isscalar(index_n0_): index_n0_ = torch.tensor([index_n0_]) ;
      if np.isscalar(index_n1_): index_n1_ = torch.tensor([index_n1_]) ;
      index_n1__,index_n0__ = torch.meshgrid(index_n1_.to(dtype=torch.int64),index_n0_.to(dtype=torch.int64),indexing='ij');
      index_n01__ = index_n0__.to(dtype=torch.int64) + index_n1__.to(dtype=torch.int64)*n_0; #<-- this should encourage index_n01__ to be torch.int64. ;
      index_n01_ = torch.flatten(index_n01__);
      return(index_n01_);
