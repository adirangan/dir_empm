from dir_empm.dir_matlab_macros import * ;
from scipy.interpolate import RegularGridInterpolator ;

flag_verbose=1;
f = lambda x0_ , x1_ : 1*x0_ + i*x1_ ;
x0_ = torch.round(torch.tensor( [ 2.0, 2.0, 0.0, 0.5, 1.0, 1.5, 2.0] )).to(dtype=torch.float32);
x1_ = torch.round(torch.tensor( [10.0,10.0, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5,10.0] )).to(dtype=torch.float32);
x1__,x0__ = torch.meshgrid(x1_,x0_,indexing='ij'); #<-- reversed to match matlab. ;
f_grid_01__ = f(x0__,x1__);
y0_ = torch.tensor( [1.85,1.25,1.50,1.75,0.75] ).to(dtype=torch.float32);
y1_ = torch.tensor( [7.25,8.75,9.50,9.75,7.75] ).to(dtype=torch.float32);

f_true_ = f(y0_,y1_);
disp(sprintf(' %% %% %% %% %% %% %% %% '));
disp(sprintf(' %% f_true_[0]: [%+0.6f%+0.6fi] ',torch.real(f_true_[0]).item(),torch.imag(f_true_[0]).item()));
disp(sprintf(' %% f_true_[1]: [%+0.6f%+0.6fi] ',torch.real(f_true_[1]).item(),torch.imag(f_true_[1]).item()));
disp(sprintf(' %% f_true_[2]: [%+0.6f%+0.6fi] ',torch.real(f_true_[2]).item(),torch.imag(f_true_[2]).item()));
disp(sprintf(' %% f_true_[3]: [%+0.6f%+0.6fi] ',torch.real(f_true_[3]).item(),torch.imag(f_true_[3]).item()));
disp(sprintf(' %% f_true_[4]: [%+0.6f%+0.6fi] ',torch.real(f_true_[4]).item(),torch.imag(f_true_[4]).item()));
disp(sprintf(' %% %% %% %% %% %% %% %% '));

f_func_ = matlab_interp2_0(x1_,x0_,f_grid_01__,y1_,y0_);
disp(sprintf(' %% %% %% %% %% %% %% %% '));
disp(sprintf(' %% f_func_[0]: [%+0.6f%+0.6fi] ',torch.real(f_func_[0]).item(),torch.imag(f_func_[0]).item()));
disp(sprintf(' %% f_func_[1]: [%+0.6f%+0.6fi] ',torch.real(f_func_[1]).item(),torch.imag(f_func_[1]).item()));
disp(sprintf(' %% f_func_[2]: [%+0.6f%+0.6fi] ',torch.real(f_func_[2]).item(),torch.imag(f_func_[2]).item()));
disp(sprintf(' %% f_func_[3]: [%+0.6f%+0.6fi] ',torch.real(f_func_[3]).item(),torch.imag(f_func_[3]).item()));
disp(sprintf(' %% f_func_[4]: [%+0.6f%+0.6fi] ',torch.real(f_func_[4]).item(),torch.imag(f_func_[4]).item()));
disp(sprintf(' %% %% %% %% %% %% %% %% '));
fnorm_disp(flag_verbose,'f_true_',f_true_,'f_func_',f_func_,'%<-- should be zero');

