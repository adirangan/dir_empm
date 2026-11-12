import numpy as np ; import torch ;

'''
def matlab_index_4d_gpu_0(
            device_use=None,
            n_0=None,
            index_n0_=None,
            n_1=None,
            index_n1_=None,
            n_2=None,
            index_n2_=None,
            n_3=None,
            index_n3_=None,
):
      str_thisfunction = 'matlab_index_4d_gpu_0';
      if index_n0_ is None: index_n0_ = torch.arange(0,n_0).to(device=device_use);
      if index_n1_ is None: index_n1_ = torch.arange(0,n_1).to(device=device_use);
      if index_n2_ is None: index_n2_ = torch.arange(0,n_2).to(device=device_use);
      if index_n3_ is None: index_n3_ = torch.arange(0,n_3).to(device=device_use);
      if index_n0_==':': index_n0_ = torch.arange(0,n_0).to(device=device_use);
      if index_n1_==':': index_n1_ = torch.arange(0,n_1).to(device=device_use);
      if index_n2_==':': index_n2_ = torch.arange(0,n_2).to(device=device_use);
      if index_n3_==':': index_n3_ = torch.arange(0,n_3).to(device=device_use);
      if np.isscalar(index_n0_): index_n0_ = torch.tensor([index_n0_]).to(device=device_use) ;
      if np.isscalar(index_n1_): index_n1_ = torch.tensor([index_n1_]).to(device=device_use) ;
      if np.isscalar(index_n2_): index_n2_ = torch.tensor([index_n2_]).to(device=device_use) ;
      if np.isscalar(index_n3_): index_n3_ = torch.tensor([index_n3_]).to(device=device_use) ;
      index_n3____,index_n2____,index_n1____,index_n0____ = torch.meshgrid(index_n3_.to(dtype=torch.int64,device=device_use),index_n2_.to(dtype=torch.int64,device=device_use),index_n1_.to(dtype=torch.int64,device=device_use),index_n0_.to(dtype=torch.int64,device=device_use),indexing='ij');
      index_n0123____ = index_n0____.to(dtype=torch.int64,device=device_use) + (index_n1____.to(dtype=torch.int64,device=device_use) + (index_n2____.to(dtype=torch.int64,device=device_use) + index_n3____.to(dtype=torch.int64,device=device_use)*n_2)*n_1)*n_0; #<-- this should encourage index_n0123____ to be torch.int64. ;
      index_n0123_ = torch.flatten(index_n0123____).to(device=device_use);
      return(index_n0123_);
'''

from dir_empm.dir_matlab_macros.matlab_index_prebroadcast import matlab_index_prebroadcast_gpu_, matlab_index_args_ ;

def matlab_index_4d_gpu_0(device_use=None,*args):
      n_0,index_n0_,n_1,index_n1_,n_2,index_n2_,n_3,index_n3_ = matlab_index_args_(4,args); #<-- sizes from input array. ;
      i0_ = matlab_index_prebroadcast_gpu_(device_use,n_0,index_n0_);
      i1_ = matlab_index_prebroadcast_gpu_(device_use,n_1,index_n1_);
      i2_ = matlab_index_prebroadcast_gpu_(device_use,n_2,index_n2_);
      i3_ = matlab_index_prebroadcast_gpu_(device_use,n_3,index_n3_);
      return((i0_.reshape(1,1,1,-1) + (i1_.reshape(1,1,-1,1) + (i2_.reshape(1,-1,1,1) + i3_.reshape(-1,1,1,1)*n_2)*n_1)*n_0).ravel());

