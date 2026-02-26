from dir_empm.dir_matlab_macros import * ;

r'''
function ...
[ ...
 n_q ...
,azimu_b_q_ ...
,polar_a_q_ ...
,weight_shell_q_ ...
,k_c_0_q_ ...
,k_c_1_q_ ...
,k_c_2_q_ ...
,n_polar_a ...
,polar_a_ ...
,n_azimu_b_ ...
] = ...
sample_shell_6( ...
 k_p_r_max ...
,k_eq_d ...
,str_T_vs_L ...
,flag_uniform_over_polar_a ...
) ;
%%%%%%%%;
% generate azimu_b_ and polar_a_ arrays sampled on shell of radius k_p_r_max at equatorial_distance k_eq_d. ;
% Note: azimu_b_ in [0,2*pi], polar_a_ in [0,1*pi]. ;
%%%%%%%%;
% Note that: ;
% \int (sphere of radius K) exp(+i*k*delta) = I1 = I2 = I3 = I4, where: ;
% I1 = dblquad(@(k,polar_a) 2*pi*cos(delta.*k.*cos(polar_a)).*k.^2.*sin(polar_a),0,Kmax,0,pi);
% I2 = 4*pi*(1/delta^1)*quad(@(k) k.^2.*sin(delta*k)./k,0,Kmax);
% I3 = 4*pi*(1/delta^3)*quad(@(k) k.*sin(k),0,Kmax*delta);
% I4 = 4*pi*(1/delta^3)*(sin(Kd) - Kd*cos(Kd)); 
%%%%%%%%;
% the str_T_vs_L determines: ;
% 'T': tschebycheff (chebyshev) nodes for polar_a of kind 1 (i.e., chebpts(n_polar_a,1)). ;
% 'C': tschebycheff (chebyshev) nodes for polar_a of kind 2 (i.e., chebpts(n_polar_a,2)). ;
% 'C2': similar to 'C', but forces n_azimu_b at each polar_a to be even, ;
%       except at the poles, where n_azimu_b==1. ;
% 'L': legendre nodes for polar_a (i.e., legpts(n_polar_a)). ;
%%%%%%%%;
% the flag_uniform_over_polar_a determines: ;
% 0: adapative, with n_azimu_b proportional to the length of the latitude-line at each polar_a. ;
% 1: uniform, with n_azimu_b constant, set by the length of the equator at polar_a=pi/2. ;
%%%%%%%%;
str_thisfunction = 'sample_shell_6';

if (nargin<1);
disp(sprintf(' %% testing %s',str_thisfunction));
test_sample_shell_6;
disp(sprintf(' %% returning')); return;
end;%if (nargin<1);

na=0;
if (nargin<1+na); k_p_r_max=[]; end; na=na+1;
if (nargin<1+na); d=[]; end; na=na+1;
if (nargin<1+na); str_T_vs_L = []; end; na=na+1;
if (nargin<1+na); flag_uniform_over_polar_a = []; end; na=na+1;

if isempty(k_p_r_max); k_p_r_max = 1.0; end;
if isempty(k_eq_d); k_eq_d = 1/(2*pi); end;
if isempty(str_T_vs_L); str_T_vs_L = 'L'; end;
if isempty(flag_uniform_over_polar_a); flag_uniform_over_polar_a = 0; end;

n_equator = 3+round(2*pi*k_p_r_max/max(1e-12,k_eq_d));
n_polar_a = 3+round(n_equator/2);

if str_T_vs_L == 'T';
%[tsch_node_,tsch_weight_] = tsch_node_weight_0(n_polar_a,-1,1);
[tsch_node_,tsch_weight_] = chebpts(n_polar_a,1);
polar_a_ = acos(tsch_node_); dpolar_a = mean(diff(sort(polar_a_)));
weight_ = tsch_weight_;
end;%if str_T_vs_L == 'T';

if str_T_vs_L == 'C' | str_T_vs_L == 'C2';
[tsch_node_,tsch_weight_] = chebpts(n_polar_a,2);
polar_a_ = acos(tsch_node_); dpolar_a = mean(diff(sort(polar_a_)));
weight_ = tsch_weight_;
end;%if str_T_vs_L == 'C' | str_T_vs_L == 'C2';

if str_T_vs_L == 'L';
%[lgnd_node_,lgnd_weight_] = lgnd_node_weight_0(n_polar_a,-1,1);
[lgnd_node_,lgnd_weight_] = legpts(n_polar_a);
polar_a_ = acos(lgnd_node_); dpolar_a = mean(diff(sort(polar_a_)));
weight_ = lgnd_weight_;
end;%if str_T_vs_L == 'L';

n_azimu_b_max = 3+round(2*pi*k_p_r_max/max(1e-12,k_eq_d));
n_azimu_b_ = zeros(n_polar_a,1);
n_q = 0 ;
for npolar_a = 1:n_polar_a;
polar_a = polar_a_(npolar_a);
spolar_a = sin(polar_a);
if flag_uniform_over_polar_a==0; n_azimu_b = 3+round(2*pi*spolar_a*k_p_r_max/max(1e-12,k_eq_d)); end;
if flag_uniform_over_polar_a==1; n_azimu_b = n_azimu_b_max; end;
if (str_T_vs_L == 'C2'); n_azimu_b = n_azimu_b + mod(n_azimu_b,2); end;
if (str_T_vs_L == 'C2') & (1-abs(cos(polar_a))<1e-16); n_azimu_b = 1; end;
n_azimu_b_(npolar_a) = n_azimu_b;
n_q = n_q + n_azimu_b;
end;%for npolar_a = 1:n_polar_a;
azimu_b_q_ = zeros(n_q,1); polar_a_q_ = zeros(n_q,1); weight_shell_q_ = zeros(n_q,1); ix=0;
for npolar_a = 1:n_polar_a;
polar_a = polar_a_(npolar_a);
spolar_a = sin(polar_a);
n_azimu_b = n_azimu_b_(npolar_a);
azimu_b_ = linspace(0,2*pi,n_azimu_b+1); azimu_b_ = azimu_b_(1:end-1); dazimu_b = mean(diff(azimu_b_));
if ~isfinite(dazimu_b); dazimu_b = 2*pi; end;
if n_azimu_b==1; dazimu_b = 2*pi; end;
azimu_b_q_(1+ix+(0:n_azimu_b-1)) = transpose(azimu_b_);
polar_a_q_(1+ix+(0:n_azimu_b-1)) = polar_a*ones(n_azimu_b,1);
weight_shell_q_(1+ix+(0:n_azimu_b-1)) = k_p_r_max.^2 .* weight_(npolar_a) .* dazimu_b .* ones(n_azimu_b,1);
ix = ix+n_azimu_b;
end;%for npolar_a = 1:n_polar_a;

if (nargout>4);
k_c_0_q_ = k_p_r_max.*cos(azimu_b_q_).*sin(polar_a_q_);
k_c_1_q_ = k_p_r_max.*sin(azimu_b_q_).*sin(polar_a_q_);
k_c_2_q_ = k_p_r_max.*cos(polar_a_q_);
end;%if (nargout>4);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ;
'''
def sample_shell_6(
        k_p_r_max=1.0,
        k_eq_d=1/(2*pi),
        str_T_vs_L='L',
        flag_uniform_over_polar_a=0,
):
    n_equator = 3 + matlab_scalar_round(2 * pi * k_p_r_max / max(1e-12, k_eq_d)) ;
    n_polar_a = 3 + matlab_scalar_round(n_equator / 2) ;

    if str_T_vs_L == 'T':
        tsch_node_, tsch_weight_ = np.polynomial.chebyshev.chebgauss(n_polar_a) #<-- note these range from high to low, which is the opposite of matlab chebpts. ;
        tsch_node_   = torch.tensor(  tsch_node_).to(dtype=torch.float32) ;
        tsch_weight_ = torch.tensor(tsch_weight_).to(dtype=torch.float32) ;
        polar_a_ = torch.arccos(torch.flip(tsch_node_,(0,))) ; #<-- and so we flip the nodes so that polar_a_ ranges from pi to 0. ;
        weight_ = tsch_weight_ ;
        print(f' %% Warning, np.polynomial.chebyshev.chebgauss(n_polar_a) now corresponds to inverse-sqrt-weighted quadrature'); exit();
    elif str_T_vs_L in ['C', 'C2']:
        print('Warning, type-2 nodes and weights not implemented');
        tsch_node_, tsch_weight_ = np.polynomial.chebyshev.chebgauss(n_polar_a, 'type_2') #<-- not implemented. ;
        tsch_node_   = torch.tensor(  tsch_node_).to(dtype=torch.float32) ;
        tsch_weight_ = torch.tensor(tsch_weight_).to(dtype=torch.float32) ;
        polar_a_ = torch.arccos(torch.flip(tsch_node_,(0,))) ;
        weight_ = tsch_weight_ ;
        print(f' %% Warning, np.polynomial.chebyshev.chebgauss(n_polar_a) now corresponds to inverse-sqrt-weighted quadrature'); exit();
    elif str_T_vs_L == 'L':
        lgnd_node_, lgnd_weight_ = np.polynomial.legendre.leggauss(n_polar_a) ;
        lgnd_node_   = torch.tensor(  lgnd_node_).to(dtype=torch.float32) ;
        lgnd_weight_ = torch.tensor(lgnd_weight_).to(dtype=torch.float32) ;
        polar_a_ = torch.arccos(lgnd_node_) ;
        weight_ = lgnd_weight_ ;
    else:
        raise ValueError("Invalid str_T_vs_L value") ;
    #end;%ifelse;
    
    n_azimu_b_max = 3 + matlab_scalar_round(2 * pi * k_p_r_max / np.maximum(1e-12, k_eq_d)) ;
    n_azimu_b_ = torch.zeros(n_polar_a).to(dtype=torch.int32) ;
    n_q = int(0) ;

    for npolar_a in range(n_polar_a):
        polar_a = polar_a_[npolar_a].item() ;
        spolar_a = np.sin(polar_a) ;
        if flag_uniform_over_polar_a == 0:
            n_azimu_b = 3 + matlab_scalar_round(2 * pi * spolar_a * k_p_r_max / np.maximum(1e-12, k_eq_d)) ;
        elif flag_uniform_over_polar_a == 1:
            n_azimu_b = n_azimu_b_max ;
        #end;%ifelse;
        if str_T_vs_L == 'C2':
            n_azimu_b += n_azimu_b % 2 ;
            if 1 - abs(np.cos(polar_a)) < 1e-16:
                n_azimu_b = 1 ;
            #end;%if;
        #end;%if;
        n_azimu_b_[npolar_a] = n_azimu_b ;
        n_q += n_azimu_b ;
    #end;%for;

    azimu_b_q_ = torch.zeros(n_q).to(dtype=torch.float32) ;
    polar_a_q_ = torch.zeros(n_q).to(dtype=torch.float32) ;
    weight_shell_q_ = torch.zeros(n_q).to(dtype=torch.float32) ;
    ix = 0 ;
    for npolar_a in range(n_polar_a):
        polar_a = polar_a_[npolar_a].item() ;
        spolar_a = np.sin(polar_a) ;
        n_azimu_b = int(n_azimu_b_[npolar_a].item()) ;
        azimu_b_ = torch.linspace(0, 2 * pi, n_azimu_b + 1)[:-1].to(dtype=torch.float32) ;
        dazimu_b = torch.mean(torch.diff(azimu_b_)).item() if torch.isfinite(torch.mean(torch.diff(azimu_b_))) else 2 * pi ;
        if n_azimu_b == 1: dazimu_b = 2 * pi ;
        azimu_b_q_[ix:ix + n_azimu_b] = azimu_b_ ;
        polar_a_q_[ix:ix + n_azimu_b] = polar_a ;
        weight_shell_q_[ix:ix + n_azimu_b] = k_p_r_max**2 * weight_[npolar_a].item() * dazimu_b ;
        ix += n_azimu_b ;
    #end;%for;

    k_c_0_q_ = k_p_r_max * torch.cos(azimu_b_q_) * torch.sin(polar_a_q_) ;
    k_c_1_q_ = k_p_r_max * torch.sin(azimu_b_q_) * torch.sin(polar_a_q_) ;
    k_c_2_q_ = k_p_r_max * torch.cos(polar_a_q_) ;

    return (
        n_q,
        azimu_b_q_,
        polar_a_q_,
        weight_shell_q_,
        k_c_0_q_,
        k_c_1_q_,
        k_c_2_q_,
        n_polar_a,
        polar_a_,
        n_azimu_b_,
    ) ;
