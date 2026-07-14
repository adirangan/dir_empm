import numpy as np ; import torch ;
from scipy.interpolate import CubicSpline ;
from . unique_1 import unique_1 ;

def matlab_interp1_0(
        torch_x_source_=None,
        torch_y_source_=None,
        torch_x_target_=None,
        str_type='spline',
):
    #%%%%%%%%;
    #% Following to catch sequences which are not strictly monotone ;
    #% (e.g., due to log-compression at lowered precision). ;
    #%%%%%%%%;
    (torch_x_unique_,index_x_unique_) = unique_1(torch_x_source_)[:2];
    torch_y_unique_ = torch_y_source_[index_x_unique_];
    torch_x_sorted_,index_x_sorted_ = torch.sort(torch_x_unique_,dim=0-0,descending=False);
    torch_y_sorted_ = torch_y_unique_[index_x_sorted_];

    #%%%%%%%%;
    #% Now essentially run: ;
    #% tmp_spline = CubicSpline(torch_x_source_.detach().cpu().numpy(),torch_y_source_.detach().cpu().numpy()) ;
    #%%%%%%%%;
    tmp_spline = CubicSpline(torch_x_sorted_.detach().cpu().numpy(),torch_y_sorted_.detach().cpu().numpy()) ;
    numpy_y_target_ = tmp_spline(torch_x_target_.detach().cpu().numpy());
    torch_y_target_ = torch.from_numpy(numpy_y_target_).to(device=torch_y_source_.device);
    return torch_y_target_ ;
        
