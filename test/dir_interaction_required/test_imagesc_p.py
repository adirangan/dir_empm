from dir_empm.dir_matlab_macros import * ;

'''
n_k_p_r = 16;
k_p_r_ = transpose([0:n_k_p_r-1]/max(1,n_k_p_r));
n_w_ = ceil(1+n_k_p_r*k_p_r_); n_w_ = n_w_ + mod(n_w_,2);
n_w_sum = sum(n_w_);
S_k_p_wk_ = zeros(n_w_sum,1);
nw_sum=0; sigma_w = pi/4;
for nk_p_r=0:n_k_p_r-1;
p = k_p_r_(1+nk_p_r);
n_w = n_w_(1+nk_p_r);
for nw=0:n_w-1;
w = periodize((2*pi*nw)/max(1,n_w),-pi,+pi);
S_k_p_wk_(1+nw_sum) = p*exp(-w.^2/(2*sigma_w.^2));
nw_sum=nw_sum+1;
end;%for nw=0:n_w-1;
end;%for nk_p_r=0:n_k_p_r-1;
imagesc_p(n_k_p_r,k_p_r_,n_w_,n_w_sum,S_k_p_wk_,[0,+1],colormap_beach());
'''
import matplotlib.pyplot as plt;
from dir_empm.imagesc_p import imagesc_p;
from dir_empm.colormap_beach import colormap_beach;
from dir_empm.colormap_pm import colormap_pm;
from dir_empm.colormap_80s import colormap_80s;
from dir_empm.colormap_81s import colormap_81s;

n_k_p_r = int(16);
k_p_r_ = torch.arange(n_k_p_r).to(dtype=torch.float32)/np.maximum(1,n_k_p_r);
n_w_ = torch.ceil(1 + n_k_p_r * k_p_r_).to(dtype=torch.int32);
n_w_ = n_w_ + torch.remainder(n_w_,2);
n_w_sum = int(torch.sum(n_w_).item()); sigma_w = pi/4;
S_k_p_wk_ = torch.zeros(n_w_sum).to(dtype=torch.complex64);

nw_sum = 0;
for nk_p_r in range(n_k_p_r):
    p = k_p_r_[nk_p_r].item();
    n_w = int(n_w_[nk_p_r].item());
    for nw in range(n_w):
        w = periodize((2 * pi * nw) /np.maximum(1, n_w),-pi,+pi);
        S_k_p_wk_[nw_sum] = p * np.exp(-w**2/(2*sigma_w**2));
        nw_sum += 1;
    #end;%for nw in range(n_w):
#end;%for nk_p_r in range(n_k_p_r):
assert(nw_sum==n_w_sum);

fig = plt.figure(figsize=(8, 6));
p_row = 2; p_col = 2; np=0;
for np in range(p_row*p_col):
    ax = fig.add_subplot(p_row,p_col,1+np);
    if np==0: imagesc_p(ax,n_k_p_r, k_p_r_, n_w_, n_w_sum, torch.real(S_k_p_wk_), torch.tensor([ 0, +1]), colormap_beach()); ax.set_title('colormap_beach');
    if np==1: imagesc_p(ax,n_k_p_r, k_p_r_, n_w_, n_w_sum, torch.real(S_k_p_wk_), torch.tensor([ 0, +1]), colormap_pm()); ax.set_title('colormap_pm');
    if np==2: imagesc_p(ax,n_k_p_r, k_p_r_, n_w_, n_w_sum, torch.real(S_k_p_wk_), torch.tensor([ 0, +1]), colormap_80s()); ax.set_title('colormap_80s');
    if np==3: imagesc_p(ax,n_k_p_r, k_p_r_, n_w_, n_w_sum, torch.real(S_k_p_wk_), torch.tensor([ 0, +1]), colormap_81s()); ax.set_title('colormap_81s');
    ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]);
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xticklabels([]); ax.set_yticklabels([]);
#end;%for np in range(p_row*p_col):

plt.show();
