import numpy as np ; import torch ;

'''
def matlab_index_5d_0(
        n_0=None,
        index_n0_=None,
        n_1=None,
        index_n1_=None,
        n_2=None,
        index_n2_=None,
        n_3=None,
        index_n3_=None,
        n_4=None,
        index_n4_=None,
):
      str_thisfunction = 'matlab_index_5d_0';
      if index_n0_ is None: index_n0_ = torch.arange(0,n_0);
      if index_n1_ is None: index_n1_ = torch.arange(0,n_1);
      if index_n2_ is None: index_n2_ = torch.arange(0,n_2);
      if index_n3_ is None: index_n3_ = torch.arange(0,n_3);
      if index_n4_ is None: index_n4_ = torch.arange(0,n_4);
      if index_n0_==':': index_n0_ = torch.arange(0,n_0);
      if index_n1_==':': index_n1_ = torch.arange(0,n_1);
      if index_n2_==':': index_n2_ = torch.arange(0,n_2);
      if index_n3_==':': index_n3_ = torch.arange(0,n_3);
      if index_n4_==':': index_n4_ = torch.arange(0,n_4);
      index_n4_____,index_n3_____,index_n2_____,index_n1_____,index_n0_____ = torch.meshgrid(index_n4_,index_n3_,index_n2_,index_n1_,index_n0_,indexing='ij');
      index_n01234_____ = index_n0_____ + (index_n1_____ + (index_n2_____ + (index_n3_____ + index_n4_____*n_3)*n_2)*n_1)*n_0;
      index_n01234_ = torch.flatten(index_n01234_____);
      return(index_n01234_);
'''

from dir_empm.dir_matlab_macros.matlab_index_prebroadcast import matlab_index_prebroadcast_, matlab_index_args_ ;

def matlab_index_5d_0(*args):
      n_0,index_n0_,n_1,index_n1_,n_2,index_n2_,n_3,index_n3_,n_4,index_n4_ = matlab_index_args_(5,args); #<-- sizes from input array. ;
      i0_ = matlab_index_prebroadcast_(n_0,index_n0_);
      i1_ = matlab_index_prebroadcast_(n_1,index_n1_);
      i2_ = matlab_index_prebroadcast_(n_2,index_n2_);
      i3_ = matlab_index_prebroadcast_(n_3,index_n3_);
      i4_ = matlab_index_prebroadcast_(n_4,index_n4_);
      return((i0_.reshape(1,1,1,1,-1) + (i1_.reshape(1,1,1,-1,1) + (i2_.reshape(1,1,-1,1,1) + (i3_.reshape(1,-1,1,1,1) + i4_.reshape(-1,1,1,1,1)*n_3)*n_2)*n_1)*n_0).ravel());

