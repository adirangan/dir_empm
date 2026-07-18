from dir_empm.dir_matlab_macros import * ;
from scipy.special import roots_jacobi
from scipy.special import eval_jacobi
from dir_empm.sample_shell_6 import sample_shell_6

r'''
function ...
[ ...
 n_qk ...
,n_qk_csum_ ...
,k_p_r_qk_ ...
,k_p_azimu_b_qk_ ...
,k_p_polar_a_qk_ ...
,weight_3d_k_p_qk_ ...
,weight_shell_qk_ ...
,n_k_p_r ...
,k_p_r_ ...
,weight_3d_k_p_r_ ...
,k_c_0_qk_ ...
,k_c_1_qk_ ...
,k_c_2_qk_ ...
,J_node_ ...
,J_weight_ ...
,J_chebfun_ ...
,J_polyval_ ...
,n_polar_a_k_ ...
,polar_a_ka__ ...
,n_azimu_b_ka__ ...
] = ...
sample_sphere_7( ...
 flag_verbose ...
,k_p_r_max ...
,k_eq_d ...
,str_T_vs_L ...
,flag_uniform_over_n_k_p_r ...
,flag_uniform_over_polar_a ...
) ;
% generate k_p_azimu_b_ and k_p_polar_a_ arrays sampled on sphere of radius k_p_r_max at equatorial_distance k_eq_d ;
% Note: k_p_azimu_b_ in [0,2*pi], k_p_polar_a_ in [0,1*pi];
% Note that: ;
% \int (sphere of radius K) exp(+i*k*delta) = I1 = I2 = I3 = I4, where: ;
% I1 = dblquad(@(k,k_p_polar_a) 2*pi*cos(delta.*k.*cos(k_p_polar_a)).*k.^2.*sin(k_p_polar_a),0,Kmax,0,pi);
% I2 = 4*pi*(1/delta^1)*quad(@(k) k.^2.*sin(delta*k)./k,0,Kmax);
% I3 = 4*pi*(1/delta^3)*quad(@(k) k.*sin(k),0,Kmax*delta);
% I4 = 4*pi*(1/delta^3)*(sin(Kd) - Kd*cos(Kd)); 
% ;
% Inputs: ;
% flag_verbose: integer verbosity level. ;
% k_p_r_max: real largest value of k (shell radius). ;
% k_eq_d: real equatorial distance required on each shell (for sampling). ;
% str_T_vs_L: string either 'T' for tschebycheff or 'L' [default] for legendre quadrature in the polar_a-direction (errors are comparable). ;
% flag_uniform_over_n_k_p_r: integer either '1' for uniform or '0' [default] for adaptive spherical grid (in k_p_r) over each k_p_r-shell. ;
% flag_uniform_over_polar_a: integer either '1' for tensor or '0' [default] for adaptive grid (in polar_a) over each azimu_b-ring. ;
% ;
% Outputs: ;
% n_qk: integer number of total points. ;
% n_qk_csum_: integer array of size n_k_p_r. n_qk_csum_(nk_p_r) is the starting index (0-based) for shell nk_p_r. ;
% k_p_r_qk_: real array of size n_qk. k-values for each point. ;
% k_p_azimu_b_qk_: real array of size n_qk. azimu_b values for each point. ;
% k_p_polar_a_qk_: real array of size n_qk. polar_a values for each point. ;
% weight_3d_k_p_qk_: real array of size n_qk. quadrature weights for each point: designed so that sum(weight_3d_k_p_qk_) = volume of sphere = 4/3*pi*k_p_r_max^3. ;
% weight_shell_qk_: real array of size n_qk. quadrature weights for each shell (in sequence). ;
% n_k_p_r: integer number of shells. ;
% k_p_r_: real array of size n_k_p_r. k-values for each shell. ;
% weight_3d_k_p_r_: real array of size n_k_p_r. quadrature weights for each shell. These are designed so that 4*pi*sum(weight_3d_k_p_r_) = volume of sphere. ;
% k_c_0_qk_: real array of size n_qk: k_c_0 values for each point. ;
% k_c_1_qk_: real array of size n_qk: k_c_1 values for each point. ;
% k_c_2_qk_: real array of size n_qk: k_c_2 values for each point. ;
% J_node_: real array of size n_k_p_r: list of quadrature nodes on interval [-1,+1]. ;
% J_weight_: real array of size n_k_p_r: list of quadrature weights on interval [-1,+1]. ;
% J_chebfun_: chebfun cell array of size 1+n_k_p_r: Jv_{1+nk_p_r} is the chebfun associated with the jacobi-polynomial (0,2) of order nk_p_r. ;
% J_polyval_: matrix of size (1+n_k_p_r)*(n_k_p_r): J_polyval_(1+nk_p_r,:) is equal to J_chebfun_{1+nk_p_r} evaluted at the quadrature nodes. ;
% n_polar_a_k_: integer array of size n_k_p_r. number of latitudes for each shell. ;
% polar_a_ka__: cell array of size n_k_p_r. polar_a_ka__{1+nk_p_r} is the array of polar_a values for shell nk_p_r. ;
%         For shell nk_p_r, ;
%         n_polar_a = n_polar_a_k_(1+nk_p_r) is an integer number of latitudes for that shell. ;
%         polar_a_a_ := polar_a_ka__{1+nk_p_r} is an array of size n_polar_a. ; 
%                 polar_a_a_(1+npolar_a) is a double storing the polar_a for latitude-line npolar_a. ;
% n_azimu_b_ka__: cell array of size n_k_p_r. ;
%         For shell nk_p_r, ;
%         n_polar_a = n_polar_a_k_(1+nk_p_r) is an integer number of latitudes for that shell. ;
%         n_azimu_b_a_ := n_azimu_b_ka__{1+nk_p_r} is an array of size n_polar_a. ; 
%                 n_azimu_b_a_(1+npolar_a) is an integer storing the number of azimu_b values for latitude-line npolar_a. ;
% ;

na=0;
if (nargin<1+na); flag_verbose=[]; end; na=na+1;
if (nargin<1+na); k_p_r_max=[]; end; na=na+1;
if (nargin<1+na); k_eq_d=[]; end; na=na+1;
if (nargin<1+na); str_T_vs_L=[]; end; na=na+1;
if (nargin<1+na); flag_uniform_over_n_k_p_r=[]; end; na=na+1;
if (nargin<1+na); flag_uniform_over_polar_a=[]; end; na=na+1;

if (nargin<1);
flag_verbose=2;
k_p_r_max = 48/(2*pi);
k_eq_d_ = 0.5.^[-2:0.25:3]; n_k_eq_d = length(k_eq_d_);
E_ = zeros(n_k_eq_d,1);
for nk_eq_d = 1:n_k_eq_d;
k_eq_d = k_eq_d_(nk_eq_d);
for str_T_vs_L_={'T','C','L'};
for flag_uniform_over_n_k_p_r=[0,1];
for flag_uniform_over_polar_a=[0,1];
sample_sphere_7(flag_verbose,k_p_r_max,k_eq_d,str_T_vs_L_{1},flag_uniform_over_n_k_p_r,flag_uniform_over_polar_a);
end;%for flag_uniform_over_polar_a=[0,1];
end;%for flag_uniform_over_n_k_p_r=[0,1];
end;%for str_T_vs_L_={'T','C','L'};
end;%for nk_eq_d = 1:n_k_eq_d;
disp('returning'); return;
end;%if (nargin<1);

if isempty(str_T_vs_L); str_T_vs_L = 'L'; end;
if isempty(flag_uniform_over_n_k_p_r); flag_uniform_over_n_k_p_r=0; end;
if isempty(flag_uniform_over_polar_a); flag_uniform_over_polar_a=0; end;

if (flag_verbose>2); disp(sprintf(' %% [entering sample_sphere_7]')); end;
%%%%%%%%;
n_k_p_r = 1+ceil(k_p_r_max/max(1e-12,k_eq_d)); [a_jx_,a_jw_] = jacpts(n_k_p_r,0,2);
k_p_r_ = (a_jx_+1.0)*k_p_r_max/2; weight_3d_k_p_r_ = a_jw_*(k_p_r_max/2)^3;
J_node_ = a_jx_ ; J_weight_ = a_jw_ ;
J_chebfun_ = cell(1+n_k_p_r,1+n_k_p_r); J_polyval_ = zeros(1+n_k_p_r,n_k_p_r);
for nk_p_r=0:n_k_p_r;
J_chebfun = jacpoly(nk_p_r,0,2)*sqrt(2*nk_p_r+3)/sqrt(8);
J_polyval_(1+nk_p_r,:) = J_chebfun(a_jx_);
J_chebfun_{1+nk_p_r} = J_chebfun;
end;%for nk_p_r=0:n_k_p_r;
n_qk = 0; n_qk_csum_ = zeros(1+n_k_p_r,1);
for nk_p_r=0:n_k_p_r-1;
k_p_r = k_p_r_(1+nk_p_r);
if flag_uniform_over_n_k_p_r==0; k_eq_d_use = k_eq_d; end; if flag_uniform_over_n_k_p_r==1; k_eq_d_use = k_eq_d*k_p_r/max(1e-12,k_p_r_max); end;
[n_qk_csum] = sample_shell_6(k_p_r,k_eq_d_use,str_T_vs_L,flag_uniform_over_polar_a) ;
n_qk_csum_(1+nk_p_r) = n_qk;
n_qk = n_qk + n_qk_csum;
end;%for nk_p_r=0:n_k_p_r-1;
n_qk_csum_(1+n_k_p_r) = n_qk;
%%%%%%%%;
k_p_r_qk_ = zeros(n_qk,1);
k_p_azimu_b_qk_ = zeros(n_qk,1);
k_p_polar_a_qk_ = zeros(n_qk,1);
weight_3d_k_p_qk_ = zeros(n_qk,1);
weight_shell_qk_ = zeros(n_qk,1);
n_polar_a_k_ = zeros(n_k_p_r,1);
polar_a_k_ = zeros(n_k_p_r,1);
n_azimu_b_ka__ = cell(n_k_p_r,1);
ix=0;
for nk_p_r=0:n_k_p_r-1;
k_p_r = k_p_r_(1+nk_p_r);
if flag_uniform_over_n_k_p_r==0; k_eq_d_use = k_eq_d; end; if flag_uniform_over_n_k_p_r==1; k_eq_d_use = k_eq_d*k_p_r/max(1e-12,k_p_r_max); end;
[ ...
 n_qk_csum ...
,k_p_azimu_b_sub_ ...
,k_p_polar_a_sub_ ...
,weight_k_sub_ ...
,k_c_0_qk_ ...
,k_c_1_qk_ ...
,k_c_2_qk_ ...
,n_polar_a ...
,polar_a_ ...
,n_azimu_b_ ...
] = ...
sample_shell_6( ...
 k_p_r ...
,k_eq_d_use ...
,str_T_vs_L ...
,flag_uniform_over_polar_a ...
);
ij_ = ix + (0:n_qk_csum-1);
k_p_r_qk_(1+ij_) = k_p_r*ones(n_qk_csum,1);
k_p_azimu_b_qk_(1+ij_) = k_p_azimu_b_sub_;
k_p_polar_a_qk_(1+ij_) = k_p_polar_a_sub_;
weight_shell_qk_(1+ij_) = weight_k_sub_;
weight_3d_k_p_qk_(1+ij_) = weight_k_sub_ * weight_3d_k_p_r_(1+nk_p_r) / max(1e-12,k_p_r).^2;
n_polar_a_k_(1+nk_p_r) = n_polar_a;
polar_a_ka__{1+nk_p_r} = polar_a_;
n_azimu_b_ka__{1+nk_p_r} = n_azimu_b_;
ix = ix + n_qk_csum;
end;%for nk_p_r=0:n_k_p_r-1;
%%%%%%%%;

if (flag_verbose>1);
delta_ = [0.51;0.75;1.9]; delta_012 = fnorm(delta_); delta_01 = fnorm(delta_(1:2)); 
delta_k_p_azimu_b = atan2(delta_(1+1),delta_(1+0)); delta_k_p_polar_a = atan2(delta_01,delta_(1+2));
Ix = 4*pi*(1/delta_012^3)*( sin(k_p_r_max*delta_012) - (k_p_r_max*delta_012)*cos(k_p_r_max*delta_012) );
k_c_0_qk_ = k_p_r_qk_.*cos(k_p_azimu_b_qk_).*sin(k_p_polar_a_qk_);
k_c_1_qk_ = k_p_r_qk_.*sin(k_p_azimu_b_qk_).*sin(k_p_polar_a_qk_);
k_c_2_qk_ = k_p_r_qk_.*cos(k_p_polar_a_qk_);
f_qk_ = exp(+i* ( k_c_0_qk_*delta_(1+0) + k_c_1_qk_*delta_(1+1) + k_c_2_qk_*delta_(1+2) ));
Iq = sum(f_qk_.*weight_3d_k_p_qk_);
disp(sprintf(' %% sample_sphere_7: k_p_r_max: %0.2f; k_eq_d %0.2f; n_qk %.8d n_k_shell %.6d flag_uniform_over_n_k_p_r %d flag_uniform_over_polar_a %d str_T_vs_L %s; error %0.16f',k_p_r_max,k_eq_d,n_qk,n_qk_csum_(end)-n_qk_csum_(end-1),flag_uniform_over_n_k_p_r,flag_uniform_over_polar_a,str_T_vs_L,fnorm(Ix-Iq)/max(1e-12,fnorm(Ix)))); 
end;%if (flag_verbose>0); 

if (nargout>10);
k_c_0_qk_ = k_p_r_qk_.*cos(k_p_azimu_b_qk_).*sin(k_p_polar_a_qk_);
k_c_1_qk_ = k_p_r_qk_.*sin(k_p_azimu_b_qk_).*sin(k_p_polar_a_qk_);
k_c_2_qk_ = k_p_r_qk_.*cos(k_p_polar_a_qk_);
end;%if (nargout>10);

if (flag_verbose>2); disp(sprintf(' %% [finished sample_sphere_7]')); end;
'''
def sample_sphere_7(
        flag_verbose, 
        k_p_r_max: np.float32 = 48 / (2 * pi), 
        k_eq_d: np.float32 = 1.0/ (2 * pi), 
        str_T_vs_L: str = 'L', 
        flag_uniform_over_n_k_p_r: int = 0, 
        flag_uniform_over_polar_a: int = 0,
        ):

    if flag_verbose > 2: print("Entering sample_sphere_7") ;

    n_k_p_r = 1 + int(np.ceil(k_p_r_max / np.maximum(1e-12, k_eq_d))) ;
    a_jx_, a_jw_ = roots_jacobi(n_k_p_r, 0, 2) ;
    a_jx_ = torch.tensor(a_jx_).to(dtype=torch.float32).flatten();
    a_jw_ = torch.tensor(a_jw_).to(dtype=torch.float32).flatten();
    k_p_r_ = (a_jx_ + 1.0) * k_p_r_max / 2 ;
    weight_3d_k_p_r_ = a_jw_ * (k_p_r_max / 2) ** 3 ;
    J_node_ = a_jx_ ;
    J_weight_ = a_jw_ ;
    J_chebfun_ = [None] ;
    J_polyval_ = [None] ;
    n_qk = int(0) ;
    n_qk_csum_ = torch.zeros(1 + n_k_p_r).to(dtype=torch.int32) ;

    for nk_p_r in range(n_k_p_r):
        k_p_r = k_p_r_[nk_p_r].item() ;
        k_eq_d_use = k_eq_d if flag_uniform_over_n_k_p_r == 0 else k_eq_d * k_p_r / np.maximum(1e-12, k_p_r_max) ;
        n_qk_csum,*_ = sample_shell_6(k_p_r, k_eq_d_use, str_T_vs_L, flag_uniform_over_polar_a)[:1] ;
        n_qk_csum_[nk_p_r] = n_qk ;
        n_qk += int(n_qk_csum) ;
    #end;%for;

    n_qk_csum_[-1] = n_qk ;

    k_p_r_qk_ = torch.zeros(n_qk).to(dtype=torch.float32) ;
    k_p_azimu_b_qk_ = torch.zeros(n_qk).to(dtype=torch.float32) ;
    k_p_polar_a_qk_ = torch.zeros(n_qk).to(dtype=torch.float32) ;
    weight_3d_k_p_qk_ = torch.zeros(n_qk).to(dtype=torch.float32) ;
    weight_shell_qk_ = torch.zeros(n_qk).to(dtype=torch.float32) ;
    n_polar_a_k_ = torch.zeros(n_k_p_r).to(dtype=torch.int32) ;
    polar_a_ka__ = cell(n_k_p_r) ; #<-- cell array. ;
    n_azimu_b_ka__ = cell(n_k_p_r) ; #<-- cell array. ;

    ix = 0 ;
    for nk_p_r in range(n_k_p_r):
        k_p_r = k_p_r_[nk_p_r].item() ;
        k_eq_d_use = k_eq_d if flag_uniform_over_n_k_p_r == 0 else k_eq_d * k_p_r / np.maximum(1e-12, k_p_r_max) ;
        (
            n_qk_csum,
            k_p_azimu_b_sub_,
            k_p_polar_a_sub_,
            weight_k_sub_,
            k_c_0_qk_,
            k_c_1_qk_,
            k_c_2_qk_,
            n_polar_a,
            polar_a_,
            n_azimu_b_,
        ) = sample_shell_6(
            k_p_r,
            k_eq_d_use,
            str_T_vs_L,
            flag_uniform_over_polar_a,
        ) ;
        ij_ = ix + torch.arange(n_qk_csum).to(dtype=torch.int32) ;
        k_p_r_qk_[ij_] = k_p_r ;
        k_p_azimu_b_qk_[ij_] = k_p_azimu_b_sub_ ;
        k_p_polar_a_qk_[ij_] = k_p_polar_a_sub_ ;
        weight_shell_qk_[ij_] = weight_k_sub_ ;
        weight_3d_k_p_qk_[ij_] = weight_k_sub_ * weight_3d_k_p_r_[nk_p_r].item() / np.maximum(1e-12, k_p_r) ** 2 ;
        n_polar_a_k_[nk_p_r] = n_polar_a ;
        polar_a_ka__[nk_p_r] = polar_a_ ; #<-- cell array. ;
        n_azimu_b_ka__[nk_p_r] = n_azimu_b_ ; #<-- cell array. ;
        ix += n_qk_csum ;
    #end;%for;

    k_c_0_qk_ = k_p_r_qk_ * torch.cos(k_p_azimu_b_qk_) * torch.sin(k_p_polar_a_qk_) ;
    k_c_1_qk_ = k_p_r_qk_ * torch.sin(k_p_azimu_b_qk_) * torch.sin(k_p_polar_a_qk_) ;
    k_c_2_qk_ = k_p_r_qk_ * torch.cos(k_p_polar_a_qk_) ;

    if flag_verbose > 2: print("Finished sample_sphere_7") ;
    
    return (
        n_qk,
        n_qk_csum_,
        k_p_r_qk_,
        k_p_azimu_b_qk_,
        k_p_polar_a_qk_,
        weight_3d_k_p_qk_,
        weight_shell_qk_,
        n_k_p_r,
        k_p_r_,
        weight_3d_k_p_r_,
        k_c_0_qk_,
        k_c_1_qk_,
        k_c_2_qk_,
        J_node_,
        J_weight_,
        J_chebfun_,
        J_polyval_,
        n_polar_a_k_,
        polar_a_ka__,
        n_azimu_b_ka__,
    ) ;
