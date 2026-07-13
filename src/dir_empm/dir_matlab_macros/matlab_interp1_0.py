import numpy as np ; import torch ;
from scipy.interpolate import CubicSpline ;

def matlab_interp1_0(
        torch_x_source_=None,
        torch_y_source_=None,
        torch_x_target_=None,
        str_type='spline',
):
    
    tmp_spline = CubicSpline(torch_x_source_.detach().cpu().numpy(),torch_y_source_.detach().cpu().numpy()) ;
    numpy_y_target_ = tmp_spline(torch_x_target_.detach().cpu().numpy());
    torch_y_target_ = torch.from_numpy(numpy_y_target_).to(device=torch_y_source_.device);
    return torch_y_target_ ;
        
