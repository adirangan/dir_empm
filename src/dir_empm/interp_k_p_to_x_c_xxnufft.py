from dir_empm.dir_matlab_macros import * ;
from dir_empm.xxnufft2d3 import xxnufft2d3 ;

r'''
function ...
S_x_c_ = ...
interp_k_p_to_x_c_xxnufft( ...
 n_x0 ...
,diameter_x0_c ...
,n_x1 ...
,diameter_x1_c ...
,n_k_p_r ...
,k_p_r_ ...
,n_w_ ...
,S_k_p_wk_ ...
,flag_u_vs_c ...
);
%%%%%%%%;

str_thisfunction = 'interp_k_p_to_x_c_xxnufft';

na=0;
if (nargin<1+na); n_x0=[]; end; na=na+1;
if (nargin<1+na); diameter_x0_c=[]; end; na=na+1;
if (nargin<1+na); n_x1=[]; end; na=na+1;
if (nargin<1+na); diameter_x1_c=[]; end; na=na+1;
if (nargin<1+na); n_k_p_r=[]; end; na=na+1;
if (nargin<1+na); k_p_r_=[]; end; na=na+1;
if (nargin<1+na); n_w_=[]; end; na=na+1;
if (nargin<1+na); S_k_p_wk_=[]; end; na=na+1;
if (nargin<1+na); flag_u_vs_c=[]; end; na=na+1;

flag_verbose=0;

if isempty(flag_u_vs_c); flag_u_vs_c=1; end;
if (flag_verbose>0); disp(sprintf(' %% flag_u_vs_c==%d in %s',flag_u_vs_c,str_thisfunction)); end;

if flag_u_vs_c==1;
dx0 = diameter_x0_c/max(1,n_x0);
dx1 = diameter_x1_c/max(1,n_x1);
end;%if flag_u_vs_c==1;

if flag_u_vs_c==0;
dx0 = diameter_x0_c/max(1,n_x0-1);
dx1 = diameter_x1_c/max(1,n_x1-1);
end;%if flag_u_vs_c==0;

n_w_max = max(n_w_); n_w_sum = sum(n_w_(1:n_k_p_r)); n_w_csum_ = cumsum([0;n_w_]);
k0_wk_ = zeros(n_w_sum,1);
k1_wk_ = zeros(n_w_sum,1);
if numel(unique(n_w_))==1;
omega_w_ = 2*pi*transpose(0:n_w_max-1)/max(1,n_w_max);
k0_wk_ = reshape(bsxfun(@times,cos(omega_w_),reshape(2*pi*k_p_r_,[1,n_k_p_r])),[n_w_sum,1]);
k1_wk_ = reshape(bsxfun(@times,sin(omega_w_),reshape(2*pi*k_p_r_,[1,n_k_p_r])),[n_w_sum,1]);
end;%if numel(unique(n_w_))==1;
if numel(unique(n_w_))> 1;
nw_sum=0;
for nk_p_r=0:n_k_p_r-1;
r = 2.0d0*pi*k_p_r_(1+nk_p_r);
for nw=0:n_w_(1+nk_p_r)-1;
omega = (2.0d0*pi*nw)/max(1,n_w_(1+nk_p_r));
k0_wk_(1+nw_sum) = r*cos(omega);
k1_wk_(1+nw_sum) = r*sin(omega);
nw_sum = nw_sum+1;
end;%for nw=0:n_w_(1+nk_p_r)-1;
end;%for nk_p_r=0:n_k_p_r-1;
if (nw_sum~=n_w_sum); disp(sprintf(' %% Warning, nw_sum %d vs %d',nw_sum,n_w_sum)); end;
end;%if numel(unique(n_w_))> 1;

if flag_u_vs_c==1;
%S_x_u_ = xxnufft2d1(n_w_sum,k0_wk_*dx0,k1_wk_*dx1,S_k_p_wk_,+1,1e-12,n_x0,n_x1)/max(1,sqrt(n_x0*n_x1)); %<-- phase-factors not correct if n_x0 or n_x1 is odd. ;
x_u_0_ = transpose(linspace(-diameter_x0_c/2,+diameter_x0_c/2,n_x0+1)); x_u_0_ = x_u_0_(1:n_x0);
x_u_1_ = transpose(linspace(-diameter_x1_c/2,+diameter_x1_c/2,n_x1+1)); x_u_1_ = x_u_1_(1:n_x1);
[x_u_0__,x_u_1__] = ndgrid(x_u_0_,x_u_1_);
S_x_u_ = xxnufft2d3(n_w_sum,k0_wk_,k1_wk_,S_k_p_wk_,+1,1e-12,n_x0*n_x1,x_u_0__(:),x_u_1__(:))/max(1,sqrt(n_x0*n_x1))/max(1,n_w_sum);
S_x_u_ = reshape(S_x_u_,[n_x0,n_x1]);
S_x_g_ = S_x_u_;
end;%if flag_u_vs_c==1;

if flag_u_vs_c==0;
x_c_0_ = transpose(linspace(-diameter_x0_c/2,+diameter_x0_c/2,n_x0));
x_c_1_ = transpose(linspace(-diameter_x1_c/2,+diameter_x1_c/2,n_x1));
[x_c_0__,x_c_1__] = ndgrid(x_c_0_,x_c_1_);
S_x_c_ = xxnufft2d3(n_w_sum,k0_wk_,k1_wk_,S_k_p_wk_,+1,1e-12,n_x0*n_x1,x_c_0__(:),x_c_1__(:))/max(1,sqrt(n_x0*n_x1))/max(1,n_w_sum);
S_x_c_ = reshape(S_x_c_,[n_x0,n_x1]);
S_x_g_ = S_x_c_;
end;%if flag_u_vs_c==0;
'''

def interp_k_p_to_x_c_xxnufft(
    n_x0=None,
    diameter_x0_c=None,
    n_x1=None,
    diameter_x1_c=None,
    n_k_p_r=None,
    k_p_r_=None,
    n_w_=None,
    S_k_p_wk_=None,
    flag_u_vs_c=None,
):

    str_thisfunction = 'interp_k_p_to_x_c_xxnufft';
    flag_verbose=0;
    if isempty(flag_u_vs_c): flag_u_vs_c=1; #end;
    if (flag_verbose>0): disp(sprintf(' %% flag_u_vs_c==%d in %s',flag_u_vs_c,str_thisfunction)); #end;

    if flag_u_vs_c==1:
        dx0 = diameter_x0_c/max(1,n_x0);
        dx1 = diameter_x1_c/max(1,n_x1);
    #end;%if flag_u_vs_c==1;

    if flag_u_vs_c==0:
        dx0 = diameter_x0_c/max(1,n_x0-1);
        dx1 = diameter_x1_c/max(1,n_x1-1);
    #end;%if flag_u_vs_c==0;

    n_w_max = int(torch.max(n_w_).item()); n_w_sum = int(torch.sum(n_w_).item());
    if numel_unique(n_w_)==1:
        k0_wk__ = torch.zeros(mtr((n_w_sum,n_k_p_r))).to(dtype=torch.float32);
        k1_wk__ = torch.zeros(mtr((n_w_sum,n_k_p_r))).to(dtype=torch.float32);
        omega_w_ = 2*pi*torch.arange(n_w_max).to(dtype=torch.float32)/np.maximum(1,n_w_max);
        k0_wk__ = mmmm(torch.reshape(torch.cos(omega_w_),mtr((n_w_max,1))),torch.reshape(2*pi*k_p_r_,mtr((1,n_k_p_r))));
        k0_wk_ = k0_wk__.ravel();
        k1_wk__ = mmmm(torch.reshape(torch.sin(omega_w_),mtr((n_w_max,1))),torch.reshape(2*pi*k_p_r_,mtr((1,n_k_p_r))));
        k1_wk_ = k1_wk__.ravel();
    #end;%if numel_unique(n_w_)==1:
    if numel_unique(n_w_)>1:
        na = 0 ;
        for nk_p_r in range(n_k_p_r):
            r = 2 * pi * k_p_r_[nk_p_r].item() ;
            for nw in range(int(n_w_[nk_p_r].item())):
                omega = (2 * pi * nw) / max(1, int(n_w_[nk_p_r].item())) ;
                k0_wk_[na] = r * np.cos(omega);
                k1_wk_[na] = r * np.sin(omega);
                na += 1 ;
            #end;%for nw in range(int(n_w_[nk_p_r].item()));
        #end;%for nk_p_r in range(n_k_p_r);
        if na != n_w_max: print('Warning, na!=n_w_max in interp_x_c_to_k_p_xxnufft.py') ;
    #end;%if numel_unique(n_w_)>1:

    if flag_u_vs_c==1:
        x_u_0_ = torch.linspace(-diameter_x0_c/2,+diameter_x0_c/2,n_x0+1).to(dtype=torch.float32)[:-1];
        x_u_1_ = torch.linspace(-diameter_x1_c/2,+diameter_x1_c/2,n_x1+1).to(dtype=torch.float32)[:-1];
        x_u_1__,x_u_0__ = torch.meshgrid(x_u_1_,x_u_0_,indexing='ij'); #<-- reversed to match matlab. ;
        S_x_u_ = xxnufft2d3(n_w_sum,k0_wk_,k1_wk_,S_k_p_wk_,+1,1e-12,n_x0*n_x1,x_u_0__.ravel(),x_u_1__.ravel())/max(1,np.sqrt(n_x0*n_x1))/max(1,n_w_sum);
        S_x_u_ = torch.reshape(S_x_u_,mtr((n_x0,n_x1)));
        S_x_g_ = S_x_u_;
    #end;%if flag_u_vs_c==1;

    if flag_u_vs_c==0:
        x_c_0_ = torch.linspace(-diameter_x0_c/2,+diameter_x0_c/2,n_x0).to(dtype=torch.float32);
        x_c_1_ = torch.linspace(-diameter_x1_c/2,+diameter_x1_c/2,n_x1).to(dtype=torch.float32);
        x_c_1__,x_c_0__ = torch.meshgrid(x_c_1_,x_c_0_,indexing='ij'); #<-- reversed to match matlab. ;
        S_x_c_ = xxnufft2d3(n_w_sum,k0_wk_,k1_wk_,S_k_p_wk_,+1,1e-12,n_x0*n_x1,x_c_0__.ravel(),x_c_1__.ravel())/max(1,np.sqrt(n_x0*n_x1))/max(1,n_w_sum);
        S_x_c_ = torch.reshape(S_x_c_,mtr((n_x0,n_x1)));
        S_x_g_ = S_x_c_;
    #end;%if flag_u_vs_c==0;

    return S_x_g_ ;
