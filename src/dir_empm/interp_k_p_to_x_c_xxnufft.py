from dir_empm.dir_matlab_macros import * ;
from dir_empm.xxnufft2d1 import xxnufft2d1

r'''
function ...
S_x_c_ = ...
interp_k_p_to_x_c_xxnufft( ...
 n_x1 ...
,diameter_x1_c ...
,n_x2 ...
,diameter_x2_c ...
,n_r ...
,grid_k_p_ ...
,n_w_ ...
,S_k_p_ ...
);
%%%%%%%%;
verbose=0;
dx1 = diameter_x1_c/max(1,n_x1);
dx2 = diameter_x2_c/max(1,n_x2);
n_w_sum = sum(n_w_(1:n_r));
k0_ = zeros(n_w_sum,1);
k1_ = zeros(n_w_sum,1);
S_x_c_ = zeros(n_x1*n_x2,1);
nw_sum=0;
for nr=0:n_r-1;
r = 2.0d0*pi*grid_k_p_(1+nr);
for nw=0:n_w_(1+nr)-1;
omega = (2.0d0*pi*nw)/max(1,n_w_(1+nr));
k1_(1+nw_sum) = r*cos(omega)*dx1;
k2_(1+nw_sum) = r*sin(omega)*dx2;
nw_sum = nw_sum+1;
end;%for nw=0:n_w_(1+nr)-1;
end;%for nr=0:n_r-1;
if (nw_sum~=n_w_sum); disp(sprintf(' %% Warning, nw_sum %d vs %d',nw_sum,n_w_sum)); end;
S_x_c_ = xxnufft2d1(n_w_sum,k1_,k2_,S_k_p_,+1,1e-12,n_x1,n_x2)/max(1,sqrt(n_x1*n_x2));
'''

def interp_k_p_to_x_c_xxnufft(
        n_x1, 
        diameter_x1_c, 
        n_x2, 
        diameter_x2_c, 
        n_r, 
        grid_k_p_, 
        n_w_, 
        S_k_p_,
        ):
    verbose = 0 ;
    dx1 = diameter_x1_c / max(1, n_x1) ;
    dx2 = diameter_x2_c / max(1, n_x2)   ;
    n_w_sum = int(torch.sum(n_w_[:n_r]).item()) ;
    k1_ = torch.zeros(n_w_sum) ;
    k2_ = torch.zeros(n_w_sum) ;
    nw_sum = 0 ;
    for nr in range(n_r):
        r = 2.0 * pi * grid_k_p_[nr].item() ;
        for nw in range(int(n_w_[nr].item())):
            omega = (2.0 * pi * nw) / max(1, int(n_w_[nr].item())) ;
            k1_[nw_sum] = r * np.cos(omega) * dx1 ;
            k2_[nw_sum] = r * np.sin(omega) * dx2 ;
            nw_sum += 1 ;
        #end;%for nw in range(int(n_w_[nr].item()));
    #end;%for nr in range(n_r);
    if nw_sum != n_w_sum: print(f"Warning, nw_sum {nw_sum} vs {n_w_sum}") ;
    S_x_c_ = torch.zeros(n_x1 * n_x2) ;
    S_x_c_ = xxnufft2d1(n_w_sum,k1_,k2_,S_k_p_,+1,1e-12,n_x1,n_x2) / max(1,np.sqrt(n_x1 * n_x2)) ;
    return S_x_c_ ;
