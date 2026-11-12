import numpy as np ; import torch ;

from dir_empm.dir_matlab_macros.matlab_index_prebroadcast import matlab_index_prebroadcast_gpu_, matlab_index_args_ ;

def matlab_index_5d_gpu_0(device_use=None,*args):
      n_0,index_n0_,n_1,index_n1_,n_2,index_n2_,n_3,index_n3_,n_4,index_n4_ = matlab_index_args_(5,args); #<-- sizes from input array. ;
      i0_ = matlab_index_prebroadcast_gpu_(device_use,n_0,index_n0_);
      i1_ = matlab_index_prebroadcast_gpu_(device_use,n_1,index_n1_);
      i2_ = matlab_index_prebroadcast_gpu_(device_use,n_2,index_n2_);
      i3_ = matlab_index_prebroadcast_gpu_(device_use,n_3,index_n3_);
      i4_ = matlab_index_prebroadcast_gpu_(device_use,n_4,index_n4_);
      return((i0_.reshape(1,1,1,1,-1) + (i1_.reshape(1,1,1,-1,1) + (i2_.reshape(1,1,-1,1,1) + (i3_.reshape(1,-1,1,1,1) + i4_.reshape(-1,1,1,1,1)*n_3)*n_2)*n_1)*n_0).ravel());

