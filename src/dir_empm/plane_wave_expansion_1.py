from dir_empm.dir_matlab_macros import * ;
from dir_empm.get_Ylm__2 import get_Ylm__2 ;
from scipy.special import jv

r'''
function ...
a_k_Y_ = ...
plane_wave_expansion_1( ...
 n_k_p_r ...
,k_p_r_ ...
,x_ ...
,l_max_ ...
);
% plane-wave expansion in terms of spherical-bessel-functions and spherical-harmonics. ;
% we assume plane-wave is of the form exp( +i * < 2*pi*k_ , x_ > ). ;

na=0;
if (nargin<1+na); n_k_p_r = []; end; na=na+1;
if (nargin<1+na); k_p_r_ = []; end; na=na+1;
if (nargin<1+na); x_ = []; end; na=na+1;
if (nargin<1+na); l_max_ = []; end; na=na+1;

flag_verbose=0;

l_max_max = max(l_max_);
n_y_ = (1+l_max_).^2;
n_y_sum = sum(n_y_);
n_y_csum_ = cumsum([0;n_y_]);

x_p_r = fnorm(x_); x_p_r01 = fnorm(x_(1:2)); x_p_azimu_b = atan2(x_(2),x_(1)); x_p_polar_a = atan2(x_p_r01,x_(3));
t_p_r_ = 2*pi*k_p_r_*x_p_r;
Ylm_x__ = get_Ylm__(1+l_max_max,0:l_max_max,1,x_p_azimu_b,x_p_polar_a,0);
a_k_Y_ = zeros(n_y_sum,1);
na=0;
for nk_p_r=0:n_k_p_r-1;
t_p_r = t_p_r_(1+nk_p_r);
n_y = n_y_(1+nk_p_r);
l_max = l_max_(1+nk_p_r);
for l_val=0:l_max;
Ylm_d_x_ = Ylm_x__{1+l_val}(:);
if abs(t_p_r)>=1e-12; jl = besselj(l_val+0.5,t_p_r).*sqrt(pi/(2*t_p_r)); end;
if abs(t_p_r)< 1e-12; jl = 1*(l_val==0) + 0*(l_val> 0); end;
a_k_Y_(1+na+[0:1+2*l_val - 1]) = 4*pi*(i^l_val)*jl*conj(Ylm_d_x_);
na=na+1+2*l_val;
end;%for l_val=0:l_max;
end;%for nk_p_r=0:n_k_p_r-1;
assert(na==n_y_sum);
'''

def plane_wave_expansion_1(
        n_k_p_r,
        k_p_r_,
        x_,
        l_max_,
):
    l_max_max = int(torch.max(l_max_).item()) ;
    n_y_ = (1 + l_max_ ) ** 2 ;
    n_y_sum = int(torch.sum(n_y_).item()) ;
    n_y_csum_ = cumsum_0(n_y_);
    x_p_r = fnorm(x_);
    x_p_r01 = fnorm(x_[:2]);
    x_p_azimu_b = np.arctan2(x_[1].item(), x_[0].item()) ;
    x_p_polar_a = np.arctan2(x_p_r01, x_[2].item()) ;
    t_p_r_ = 2 * pi * k_p_r_ * x_p_r ;
    _,Ylm_x__ = get_Ylm__2(
        [],
        1 + l_max_max,
        torch.arange(0, l_max_max + 1).to(dtype=torch.int32),
        1,
        torch.tensor([x_p_azimu_b]).to(dtype=torch.float32),
        torch.tensor([x_p_polar_a]).to(dtype=torch.float32),
        0
    )[:2] ;
    a_k_Y_ = torch.zeros(n_y_sum).to(dtype=torch.complex64) ;
    na = 0 ;
    for nk_p_r in range(n_k_p_r):
        t_p_r = t_p_r_[nk_p_r].item() ;
        n_y = int(n_y_[nk_p_r].item()) ;
        l_max = int(l_max_[nk_p_r].item()) ;
        for l_val in range(l_max + 1):
            Ylm_d_x_ = Ylm_x__[l_val].flatten() ;
            if abs(t_p_r) >= 1e-12:
                jl = jv(l_val+0.5,t_p_r) * np.sqrt(pi/(2*t_p_r)) ;
            else:
                jl = 1 if l_val == 0 else 0 ;
            idx_start = na ;
            idx_end = na + 2 * l_val + 1 ;
            a_k_Y_[idx_start:idx_end] = 4 * pi * (1j ** l_val) * jl * torch.conj(Ylm_d_x_) ;
            na += 2 * l_val + 1 ;
        #end;%for l_val=0:l_max;
    #end;%for nk_p_r=0:n_k_p_r-1;
    assert na == n_y_sum ;
    return a_k_Y_ ;

