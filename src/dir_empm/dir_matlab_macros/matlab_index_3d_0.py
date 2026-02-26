import numpy as np ; import torch ;

def matlab_index_3d_0(
        n_0=None,
        index_n0_=None,
        n_1=None,
        index_n1_=None,
        n_2=None,
        index_n2_=None,
):
      str_thisfunction = 'matlab_index_3d_0';
      if index_n0_ is None: index_n0_ = torch.arange(0,n_0);
      if index_n1_ is None: index_n1_ = torch.arange(0,n_1);
      if index_n2_ is None: index_n2_ = torch.arange(0,n_2);
      if index_n0_==':': index_n0_ = torch.arange(0,n_0);
      if index_n1_==':': index_n1_ = torch.arange(0,n_1);
      if index_n2_==':': index_n2_ = torch.arange(0,n_2);
      if np.isscalar(index_n0_): index_n0_ = torch.tensor([index_n0_]) ;
      if np.isscalar(index_n1_): index_n1_ = torch.tensor([index_n1_]) ;
      if np.isscalar(index_n2_): index_n2_ = torch.tensor([index_n2_]) ;
      index_n2___,index_n1___,index_n0___ = torch.meshgrid(index_n2_.to(dtype=torch.int64),index_n1_.to(dtype=torch.int64),index_n0_.to(dtype=torch.int64),indexing='ij');
      index_n012___ = index_n0___.to(dtype=torch.int64) + (index_n1___.to(dtype=torch.int64) + index_n2___.to(dtype=torch.int64)*n_1)*n_0; #<-- this should encourage index_n012___ to be torch.int64. ;
      index_n012_ = torch.flatten(index_n012___);
      return(index_n012_);
