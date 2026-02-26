import numpy as np ; import torch ;

def matlab_index_4d_0(
        n_0=None,
        index_n0_=None,
        n_1=None,
        index_n1_=None,
        n_2=None,
        index_n2_=None,
        n_3=None,
        index_n3_=None,
):
      str_thisfunction = 'matlab_index_4d_0';
      if index_n0_ is None: index_n0_ = torch.arange(0,n_0);
      if index_n1_ is None: index_n1_ = torch.arange(0,n_1);
      if index_n2_ is None: index_n2_ = torch.arange(0,n_2);
      if index_n3_ is None: index_n3_ = torch.arange(0,n_3);
      if index_n0_==':': index_n0_ = torch.arange(0,n_0);
      if index_n1_==':': index_n1_ = torch.arange(0,n_1);
      if index_n2_==':': index_n2_ = torch.arange(0,n_2);
      if index_n3_==':': index_n3_ = torch.arange(0,n_3);
      if np.isscalar(index_n0_): index_n0_ = torch.tensor([index_n0_]) ;
      if np.isscalar(index_n1_): index_n1_ = torch.tensor([index_n1_]) ;
      if np.isscalar(index_n2_): index_n2_ = torch.tensor([index_n2_]) ;
      if np.isscalar(index_n3_): index_n3_ = torch.tensor([index_n3_]) ;
      index_n3____,index_n2____,index_n1____,index_n0____ = torch.meshgrid(index_n3_.to(dtype=torch.int64),index_n2_.to(dtype=torch.int64),index_n1_.to(dtype=torch.int64),index_n0_.to(dtype=torch.int64),indexing='ij');
      index_n0123____ = index_n0____.to(dtype=torch.int64) + (index_n1____.to(dtype=torch.int64) + (index_n2____.to(dtype=torch.int64) + index_n3____.to(dtype=torch.int64)*n_2)*n_1)*n_0; #<-- this should encourage index_n0123____ to be torch.int64. ;
      index_n0123_ = torch.flatten(index_n0123____);
      return(index_n0123_);
