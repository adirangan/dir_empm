from dir_empm.dir_matlab_macros import * ;
from dir_empm.xxnufft2d2 import xxnufft2d2

r'''
function ...
S_k_p_ = ...
interp_x_c_to_k_p_xxnufft( ...
 n_x1 ...
,diameter_x1_c ...
,n_x2 ...
,diameter_x2_c ...
,S_x_c_ ...
,n_r ...
,grid_k_p_ ...
,n_w_ ...
) ;

flag_verbose=0;
iflag = -1;
eps = 1.0d-6;
flag_2_versus_3 = 0; %<-- logical: compare type-2 and type-3 results. ;
dx1 = diameter_x1_c/max(1,n_x1) ;
dx2 = diameter_x2_c/max(1,n_x2) ;
n_A = sum(n_w_) ;
k1_ = zeros(n_A,1);
k2_ = zeros(n_A,1);

S_k_p_ = zeros(n_A,1);
na=0;
for nr=0:n_r-1;
r = 2*pi*grid_k_p_(1+nr);
for nw=0:n_w_(1+nr)-1;
omega = (2*pi*nw)/max(1,n_w_(1+nr));
k1_(1+na) = r*cos(omega)*dx1;
k2_(1+na) = r*sin(omega)*dx2;
na = na+1;
end;%for nw=0:n_w_(1+nr)-1;
end;%for nr=0:n_r-1;
if (na~=n_A);
disp(sprintf(' %% Warning, na~=n_A in interp_x_c_to_k_p_xxnufft.m'));
end;%if (na~=n_A);
[S_k_p_,ier] = xxnufft2d2(n_A,k1_,k2_,iflag,eps,n_x1,n_x2,S_x_c_);
if (ier~=0); disp(sprintf(' %% Warning! precision out of range in xxnufft2d2 in interp_x_c_to_k_p_xxnufft.m')); end;
S_k_p_ = S_k_p_/max(1,sqrt(n_x1*n_x2));
'''

def interp_x_c_to_k_p_xxnufft(
    n_x1, 
    diameter_x1_c, 
    n_x2, 
    diameter_x2_c, 
    S_x_c_, 
    n_r, 
    grid_k_p_, 
    n_w_
    ):
    flag_verbose = 0 ;
    iflag = -1 ;
    eps = 1.0e-6 ;
    dx1 = diameter_x1_c / max(1, n_x1) ;
    dx2 = diameter_x2_c / max(1, n_x2) ;
    n_A = int(torch.sum(n_w_).item()) ;
    k1_ = torch.zeros(n_A).to(dtype=torch.float64) ;
    k2_ = torch.zeros(n_A).to(dtype=torch.float64) ;
    S_k_p_ = torch.zeros(n_A, dtype=torch.complex128) ;
    if numel_unique(n_w_)==1:
        n_w_max = int(torch.max(n_w_).item());
        n_w_sum = int(torch.sum(n_w_).item());
        k1_wk__ = torch.zeros(mtr((n_w_sum,n_r))).to(dtype=torch.float64);
        k2_wk__ = torch.zeros(mtr((n_w_sum,n_r))).to(dtype=torch.float64);
        omega_w_ = 2*pi*torch.arange(n_w_max).to(dtype=torch.float64)/np.maximum(1,n_w_max);
        k1_wk__ = mmmm(torch.reshape(torch.cos(omega_w_),mtr((n_w_max,1))),torch.reshape(2*pi*grid_k_p_,mtr((1,n_r))));
        k1_wk_ = k1_wk__.ravel();
        k2_wk__ = mmmm(torch.reshape(torch.sin(omega_w_),mtr((n_w_max,1))),torch.reshape(2*pi*grid_k_p_,mtr((1,n_r))));
        k2_wk_ = k2_wk__.ravel();
        k1_ = k1_wk_*dx1;
        k2_ = k2_wk_*dx2;
    #end;%if numel_unique(n_w_)==1:
    if numel_unique(n_w_)>1:
        na = 0 ;
        for nr in range(n_r):
            r = 2 * pi * grid_k_p_[nr].item() ;
            for nw in range(int(n_w_[nr].item())):
                omega = (2 * pi * nw) / max(1, int(n_w_[nr].item())) ;
                k1_[na] = r * np.cos(omega) * dx1 ;
                k2_[na] = r * np.sin(omega) * dx2 ;
                na += 1 ;
            #end;%for nw in range(int(n_w_[nr].item()));
        #end;%for nr in range(n_r);
        if na != n_A: print('Warning, na!=n_A in interp_x_c_to_k_p_xxnufft.py') ;
    #end;%if numel_unique(n_w_)>1:
    S_k_p_ = xxnufft2d2(n_A, k1_, k2_, iflag, eps, n_x1, n_x2, S_x_c_) ;
    S_k_p_ = S_k_p_ / max(1, np.sqrt(n_x1 * n_x2)) ;
    return S_k_p_ ;
