import numpy as np ; import torch ;
from scipy.interpolate import RegularGridInterpolator ;
numel = lambda a : int(a.numel()) ;
mtr = lambda a : tuple(reversed(a)) ; #<-- matlab-arranged size (i.e., tuple(reversed(...))). ;
from . disp_sprintf import disp ; from . disp_sprintf import sprintf ;
from . unique_1 import unique_1 ;
from . matlab_index_2d_0 import matlab_index_2d_0 ;

def matlab_interp2_0(
        torch_x1_source_=None,
        torch_x0_source_=None,
        torch_f_source_01__=None,
        torch_x1_target_=None,
        torch_x0_target_=None,
        str_type='linear',
):
    #%%%%%%%%;
    #% Following to catch sequences which are not strictly monotone ;
    #% (e.g., due to log-compression at lowered precision). ;
    #%%%%%%%%;
    flag_verbose = 0;
    n_x0_source = numel(torch_x0_source_);
    n_x1_source = numel(torch_x1_source_);
    (torch_x0_unique_,index_x0_unique_) = unique_1(torch_x0_source_)[:2];
    torch_x0_sorted_,index_x0_sorted_ = torch.sort(torch_x0_unique_,dim=0-0,descending=False);
    (torch_x1_unique_,index_x1_unique_) = unique_1(torch_x1_source_)[:2];
    torch_x1_sorted_,index_x1_sorted_ = torch.sort(torch_x1_unique_,dim=0-0,descending=False);
    n_x0_sorted = numel(torch_x0_sorted_);
    n_x1_sorted = numel(torch_x1_sorted_);
    if (flag_verbose>0):
        disp(sprintf(' %% torch_x1_source_: ')); print(torch_x1_source_);
        disp(sprintf(' %% index_x1_unique_: ')); print(index_x1_unique_);
        disp(sprintf(' %% torch_x1_unique_: ')); print(torch_x1_unique_);
        disp(sprintf(' %% index_x1_sorted_: ')); print(index_x1_sorted_);
        disp(sprintf(' %% torch_x1_sorted_: ')); print(torch_x1_sorted_);
        disp(sprintf(' %% torch_x0_source_: ')); print(torch_x0_source_);
        disp(sprintf(' %% index_x0_unique_: ')); print(index_x0_unique_);
        disp(sprintf(' %% torch_x0_unique_: ')); print(torch_x0_unique_);
        disp(sprintf(' %% index_x0_sorted_: ')); print(index_x0_sorted_);
        disp(sprintf(' %% torch_x0_sorted_: ')); print(torch_x0_sorted_);
        disp(sprintf(' %% torch_f_source_01__: ')); print(torch_f_source_01__);
    #end;%if (flag_verbose>0);

    tmp_i8_index_rhs_ = matlab_index_2d_0(n_x0_source,index_x0_unique_[index_x0_sorted_],n_x1_source,index_x1_unique_[index_x1_sorted_]);
    torch_f_sorted_01__ = torch.reshape(torch_f_source_01__.ravel()[tmp_i8_index_rhs_],mtr((n_x0_sorted,n_x1_sorted)));
    if (flag_verbose>0):
        disp(sprintf(' %% torch_f_sorted_01__: ')); print(torch_f_sorted_01__);
        disp(sprintf(' %% torch_x1_target_: ')); print(torch_x1_target_);
        disp(sprintf(' %% torch_x0_target_: ')); print(torch_x0_target_);
    #end;%if (flag_verbose>0);

    torch_x1_sorted__,torch_x0_sorted__ = torch.meshgrid(torch_x1_sorted_,torch_x0_sorted_,indexing='ij'); #%<-- reversed to match matlab. ;
    tmp_numpy_f = RegularGridInterpolator((torch_x0_sorted_.numpy(),torch_x1_sorted_.numpy()),torch_f_sorted_01__.T.numpy(),method='linear',bounds_error=False,fill_value=0);
    tmp_x_target_ = torch.row_stack((torch_x0_target_.T,torch_x1_target_.T)).T.to(dtype=torch.float32);
    torch_f_target_ = torch.tensor(tmp_numpy_f(tmp_x_target_.numpy()));
    return torch_f_target_ ;
        
