from dir_empm.dir_matlab_macros import * ;
from scipy.interpolate import RegularGridInterpolator ;

flag_verbose=1;
f = lambda x0_ , x1_ : 1*x0_ + i*x1_ ;
x0_ = torch.tensor([0,1,2]).to(dtype=torch.float32);
x1_ = torch.tensor([7,8,9,10]).to(dtype=torch.float32);
x1__,x0__ = torch.meshgrid(x1_,x0_,indexing='ij'); #<-- reversed to match matlab. ;
f_grid_01__ = f(x0__,x1__);
y0_ = torch.tensor([1.25,1.50,1.75]).to(dtype=torch.float32);
y1_ = torch.tensor([8.75,9.25,9.75]).to(dtype=torch.float32);
f_true_ = f(y0_,y1_);
tmp_numpy_f = RegularGridInterpolator((x0_.numpy(),x1_.numpy()),f_grid_01__.T.numpy(),method='linear',bounds_error=False,fill_value=0);
tmp_y_ = torch.row_stack((y0_.T,y1_.T)).T.to(dtype=torch.float32);
f_inte_ = torch.tensor(tmp_numpy_f(tmp_y_.numpy()));
disp(sprintf(' %% %% %% %% %% %% %% %% '));
disp(sprintf(' %% f_true_[0]: [%+0.6f%+0.6fi] ',torch.real(f_true_[0]).item(),torch.imag(f_true_[0]).item()));
disp(sprintf(' %% f_true_[1]: [%+0.6f%+0.6fi] ',torch.real(f_true_[1]).item(),torch.imag(f_true_[1]).item()));
disp(sprintf(' %% f_true_[2]: [%+0.6f%+0.6fi] ',torch.real(f_true_[2]).item(),torch.imag(f_true_[2]).item()));
disp(sprintf(' %% %% %% %% %% %% %% %% '));
disp(sprintf(' %% f_inte_[0]: [%+0.6f%+0.6fi] ',torch.real(f_inte_[0]).item(),torch.imag(f_inte_[0]).item()));
disp(sprintf(' %% f_inte_[1]: [%+0.6f%+0.6fi] ',torch.real(f_inte_[1]).item(),torch.imag(f_inte_[1]).item()));
disp(sprintf(' %% f_inte_[2]: [%+0.6f%+0.6fi] ',torch.real(f_inte_[2]).item(),torch.imag(f_inte_[2]).item()));
disp(sprintf(' %% %% %% %% %% %% %% %% '));
fnorm_disp(flag_verbose,'f_true_',f_true_,'f_inte_',f_inte_,'%<-- should be zero');

f_func_ = matlab_interp2_0(x1_,x0_,f_grid_01__,y1_,y0_);
disp(sprintf(' %% %% %% %% %% %% %% %% '));
disp(sprintf(' %% f_func_[0]: [%+0.6f%+0.6fi] ',torch.real(f_func_[0]).item(),torch.imag(f_func_[0]).item()));
disp(sprintf(' %% f_func_[1]: [%+0.6f%+0.6fi] ',torch.real(f_func_[1]).item(),torch.imag(f_func_[1]).item()));
disp(sprintf(' %% f_func_[2]: [%+0.6f%+0.6fi] ',torch.real(f_func_[2]).item(),torch.imag(f_func_[2]).item()));
disp(sprintf(' %% %% %% %% %% %% %% %% '));
fnorm_disp(flag_verbose,'f_true_',f_true_,'f_func_',f_func_,'%<-- should be zero');

