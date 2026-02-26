from dir_empm.dir_matlab_macros import * ;

'''
flag_verbose=0; flag_disp=1; nf=0;
k_p_r_max = 48/(2*pi); 
n_k_p_r = [];
k_p_r_ = [];
template_k_eq_d = [];
n_w_0in_ = [];
weight_3d_k_p_r_ = [];
flag_precalc = 1;
if flag_precalc;
k_eq_d = 1.0/(2*pi); str_T_vs_L = 'L';
[ ...
 n_k_p_r ...
,k_p_r_ ...
,weight_3d_k_p_r_ ...
] = ...
get_weight_3d_1( ...
 0*flag_verbose ...
,k_p_r_max ...
,k_eq_d ...
,str_T_vs_L ...
);
template_k_eq_d = -1;
n_w_max = 98; n_w_0in_ = n_w_max*ones(n_k_p_r,1);
end;%if flag_precalc;
%%%%%%%%;
[ ...
 n_w_ ...
,weight_2d_k_p_r_ ...
,weight_2d_k_p_wk_ ...
,k_p_r_wk_ ...
,k_p_w_wk_ ...
,k_c_0_wk_ ...
,k_c_1_wk_ ...
] = ...
get_weight_2d_2( ...
 0*flag_verbose ...
,n_k_p_r ...
,k_p_r_ ...
,k_p_r_max ...
,template_k_eq_d ...
,n_w_0in_ ...
,weight_3d_k_p_r_ ...
);
n_w_max = max(n_w_); n_w_sum = sum(n_w_); n_w_csum_ = cumsum([0;n_w_]);
%%%%%%%%;
if flag_disp>0;
figure(1+nf);nf=nf+1;clf;figmed;
p_row = 1; p_col = 3; np=0;
end;%if flag_disp>0;
sigma = k_p_r_max/2;
%%%%;
f = @(k,w) ones(size(k));
F_tru = pi*k_p_r_max.^2;
F_est = dot(f(k_p_r_wk_,k_p_w_wk_),weight_2d_k_p_wk_)*(2*pi)^2;
errrel_0 = abs(F_tru-F_est)/abs(F_tru);
disp(sprintf(' %% F_tru vs F_est: %0.16f',errrel_0));
if flag_disp>0;
subplot(p_row,p_col,1+np);np=np+1;
imagesc_p(n_k_p_r,k_p_r_,n_w_,n_w_sum,real(f(k_p_r_wk_,k_p_w_wk_)),[],colormap_beach());
axis image; axisnotick;
title(sprintf('log10(errrel_0) %+0.2f',log10(errrel_0)),'Interpreter','none');
end;%if flag_disp>0;
%%%%;
f = @(k,w) -1.*(1/sigma.^2).*exp(-k.^2/(2*sigma^2));
F_tru = 2*pi*(exp(-k_p_r_max.^2/(2*sigma^2)) - 1);
F_est = dot(f(k_p_r_wk_,k_p_w_wk_),weight_2d_k_p_wk_)*(2*pi)^2;
errrel_1 = abs(F_tru-F_est)/abs(F_tru);
disp(sprintf(' %% F_tru vs F_est: %0.16f',errrel_1));
if flag_disp>0;
subplot(p_row,p_col,1+np);np=np+1;
imagesc_p(n_k_p_r,k_p_r_,n_w_,n_w_sum,real(f(k_p_r_wk_,k_p_w_wk_)),[],colormap_beach());
axis image; axisnotick;
title(sprintf('log10(errrel_1) %+0.2f',log10(errrel_1)),'Interpreter','none');
end;%if flag_disp>0;
%%%%;
f = @(k,w) -(sin(w).*cos(w)+1).*(1/sigma.^2).*exp(-k.^2/(2*sigma^2));
F_tru = 2*pi*(exp(-k_p_r_max.^2/(2*sigma^2)) - 1);
F_est = dot(f(k_p_r_wk_,k_p_w_wk_),weight_2d_k_p_wk_)*(2*pi)^2;
errrel_2 = abs(F_tru-F_est)/abs(F_tru);
disp(sprintf(' %% F_tru vs F_est: %0.16f',errrel_2));
if flag_disp>0;
subplot(p_row,p_col,1+np);np=np+1;
imagesc_p(n_k_p_r,k_p_r_,n_w_,n_w_sum,real(f(k_p_r_wk_,k_p_w_wk_)),[],colormap_beach());
axis image; axisnotick;
title(sprintf('log10(errrel_2) %+0.2f',log10(errrel_2)),'Interpreter','none');
end;%if flag_disp>0;
%%%%%%%%;
disp('returning'); return;
'''
from dir_empm.get_weight_3d_1 import get_weight_3d_1
from dir_empm.get_weight_2d_2 import get_weight_2d_2

flag_verbose = 0 ;
k_p_r_max = 48 / (2 * pi) ;

k_eq_d = 1.0 / (2 * pi) ;
str_T_vs_L = 'L' ;
(
    n_k_p_r,
    k_p_r_,
    weight_3d_k_p_r_,
) = get_weight_3d_1(
    flag_verbose,
    k_p_r_max,
    k_eq_d,
    str_T_vs_L,
) ;
template_k_eq_d = -1 ;
n_w_max = 98 ;
n_w_0in_ = n_w_max * torch.ones(n_k_p_r).to(dtype=torch.int32) ;
(
    n_w_,
    weight_2d_k_p_r_,
    weight_2d_k_p_wk_,
    k_p_r_wk_,
    k_p_w_wk_,
    k_c_0_wk_,
    k_c_1_wk_,
) = get_weight_2d_2(
    flag_verbose,
    n_k_p_r,
    k_p_r_,
    k_p_r_max,
    template_k_eq_d,
    n_w_0in_,
    weight_3d_k_p_r_,
) ;

n_w_max = torch.max(n_w_).item() ;
n_w_sum = torch.sum(n_w_).item() ;
n_w_csum_ = cumsum_0(n_w_);

sigma = k_p_r_max / 2 ;

# Function 1
f = lambda k, w: torch.ones_like(k,dtype=torch.float64) ;
F_tru = torch.tensor( pi * k_p_r_max**2 ).to(dtype=torch.float64);
F_est = torch.dot(f(k_p_r_wk_, k_p_w_wk_), weight_2d_k_p_wk_.to(dtype=torch.float64)) * (2 * pi)**2 ;
errrel_0 = torch.abs(F_tru - F_est) / torch.abs(F_tru) ;
print(f" %% F_tru vs F_est: {errrel_0:.16f}") ;

# Function 2
f = lambda k, w: ( -1 * (1 / sigma**2) * torch.exp(-k**2 / (2 * sigma**2))).to(dtype=torch.float64) ;
F_tru = torch.tensor( 2 * pi * (np.exp(-k_p_r_max**2 / (2 * sigma**2)) - 1) ).to(dtype=torch.float64);
F_est = torch.dot(f(k_p_r_wk_, k_p_w_wk_), weight_2d_k_p_wk_.to(dtype=torch.float64)) * (2 * pi)**2 ;
errrel_1 = torch.abs(F_tru - F_est) / torch.abs(F_tru) ;
print(f" %% F_tru vs F_est: {errrel_1:.16f}") ;

# Function 3
f = lambda k, w: ( -(torch.sin(w) * torch.cos(w) + 1) * (1 / sigma**2) * torch.exp(-k**2 / (2 * sigma**2))).to(dtype=torch.float64) ;
F_tru = torch.tensor( 2 * pi * (np.exp(-k_p_r_max**2 / (2 * sigma**2)) - 1) ).to(dtype=torch.float64);
F_est = torch.dot(f(k_p_r_wk_, k_p_w_wk_), weight_2d_k_p_wk_.to(dtype=torch.float64)) * (2 * pi)**2 ;
errrel_2 = torch.abs(F_tru - F_est) / torch.abs(F_tru) ;
print(f" %% F_tru vs F_est: {errrel_2:.16f}") ;

