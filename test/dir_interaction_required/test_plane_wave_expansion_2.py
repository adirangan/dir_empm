from dir_empm.dir_matlab_macros import * ;
from dir_empm.plane_wave_expansion_1 import plane_wave_expansion_1 ;
from dir_empm.plane_wave_expansion_2 import plane_wave_expansion_2 ;

str_thisfunction = 'plane_wave_expansion_2';
flag_verbose=1;
if (flag_verbose>0): disp(sprintf(' %% testing %s',str_thisfunction)); #end;
parameter = {'type':'parameter'};
parameter['flag_verbose'] = 1;
n_k_p_r = 49; k_p_r_ = torch.sort(torch.rand(n_k_p_r).to(dtype=torch.float32))[0]; 
n_x = 23; x_3x__ = torch.randn(mtr((n_3,n_x))).to(dtype=torch.float32);
x_3x__[int(np.maximum(0,np.minimum(n_x-1,17))),:] = torch.zeros(3).to(dtype=torch.float32);
l_max_ = 2+torch.arange(n_k_p_r).to(dtype=torch.int32);
l_max_max = int(torch.max(l_max_).item());
n_y_ = (1+l_max_)**2;
n_y_sum = int(torch.sum(n_y_).item());
tmp_t=tic();
_,b_k_Y_yk_ = plane_wave_expansion_2(parameter,n_k_p_r,k_p_r_,n_x,x_3x__,l_max_)[:2];
tmp_t=toc(tmp_t);
if (flag_verbose>0): disp(sprintf(' %% plane_wave_expansion_2: time %0.6fs',tmp_t)); #end;
tmp_t=tic();
a_k_Y_yk_ = torch.zeros(n_y_sum).to(dtype=torch.complex64);
for nx in range(n_x):
    a_k_Y_yk_ = a_k_Y_yk_ + plane_wave_expansion_1(n_k_p_r,k_p_r_,x_3x__[nx,:],l_max_);
#end;%for nx=0:n_x-1;
tmp_t=toc(tmp_t);
if (flag_verbose>0): disp(sprintf(' %% plane_wave_expansion_1: time %0.6fs',tmp_t)); #end;
fnorm_disp(flag_verbose,'a_k_Y_yk_',a_k_Y_yk_,'b_k_Y_yk_',b_k_Y_yk_,' %%<-- should be zero');

