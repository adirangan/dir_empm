import numpy as np ; import torch ;

'''
def matlab_index_2d_gpu_0(
            device_use=None,
            n_0=None,
            index_n0_=None,
            n_1=None,
            index_n1_=None,
):
      str_thisfunction = 'matlab_index_2d_gpu_0';
      if index_n0_ is None: index_n0_ = torch.arange(0,n_0).to(device=device_use);
      if index_n1_ is None: index_n1_ = torch.arange(0,n_1).to(device=device_use);
      if index_n0_==':': index_n0_ = torch.arange(0,n_0).to(device=device_use);
      if index_n1_==':': index_n1_ = torch.arange(0,n_1).to(device=device_use);
      if np.isscalar(index_n0_): index_n0_ = torch.tensor([index_n0_]).to(device=device_use) ;
      if np.isscalar(index_n1_): index_n1_ = torch.tensor([index_n1_]).to(device=device_use) ;
      index_n1__,index_n0__ = torch.meshgrid(index_n1_.to(dtype=torch.int64,device=device_use),index_n0_.to(dtype=torch.int64,device=device_use),indexing='ij');
      index_n01__ = index_n0__.to(dtype=torch.int64,device=device_use) + index_n1__.to(dtype=torch.int64,device=device_use)*n_0; #<-- this should encourage index_n01__ to be torch.int64. ;
      index_n01_ = torch.flatten(index_n01__).to(device=device_use);
      return(index_n01_);
'''

from dir_empm.dir_matlab_macros.matlab_index_prebroadcast import matlab_index_prebroadcast_gpu_, matlab_index_args_ ;

def matlab_index_2d_gpu_0(device_use=None,*args):
      n_0,index_n0_,n_1,index_n1_ = matlab_index_args_(2,args); #<-- sizes from input array. ;
      i0_ = matlab_index_prebroadcast_gpu_(device_use,n_0,index_n0_);
      i1_ = matlab_index_prebroadcast_gpu_(device_use,n_1,index_n1_);
      return((i0_.reshape(1,-1) + i1_.reshape(-1,1)*n_0).ravel());

