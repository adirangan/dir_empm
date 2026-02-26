from dir_empm.dir_matlab_macros import * ;
from dir_empm.i4_torch_arange import i4_torch_arange ;
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

'''
function clim = imagesc_p(n_r,grid_p_,n_w_,n_A,S_p_,clim,cra_);
% polar imagesc;

if (nargin<1);
n_r = 8;
grid_p_ = transpose([0:n_r-1]/max(1,n_r));
n_w_ = ceil(1+8*grid_p_);
n_A = sum(n_w_);
S_p_ = zeros(n_A,1);
ic=0;
for nr=0:n_r-1;
p = grid_p_(1+nr);
n_w = n_w_(1+nr);
for nw=0:n_w-1;
w = (2*pi*nw)/max(1,n_w);
S_p_(1+ic) = p*cos(w);
ic=ic+1;
end;%for nw=0:n_w-1;
end;%for nr=0:n_r-1;
imagesc_p(n_r,grid_p_,n_w_,n_A,S_p_,[-1,+1],colormap_beach());
return;
end;%if (nargin<1);

if (nargin<6); clim = []; end;
if (nargin<7); cra_ = []; end;
if isempty(clim); 
clim = mean(S_p_) + std(S_p_)*2.5*[-1,1]; 
end;%if isempty(clim);
if isempty(cra_); 
cra_ = colormap_beach();
end;%if isempty(clim);

r0_ = zeros(1,n_A); r1_ = zeros(1,n_A);
w0_ = zeros(1,n_A); w1_ = zeros(1,n_A);
ncra = size(cra_,1);
C_ = zeros(1,n_A,3);
ic=0;
for nr=0:n_r-1;
r = grid_p_(1+nr);
if nr==0; r_pre = grid_p_(1); else r_pre = 0.5*(grid_p_(1+nr-1) + grid_p_(1+nr)); end;
if nr==n_r-1; r_pos = grid_p_(end); else r_pos = 0.5*(grid_p_(1+nr+1) + grid_p_(1+nr)); end;
if n_r==1; r_pre = grid_p_(end)/4.0; r_pos = grid_p_(end); end;
n_w = n_w_(1+nr);
dw = 2*pi/n_w;
for nw=0:n_w_(1+nr)-1;
w_pre = nw*dw - 0.5*dw;
w_pos = nw*dw + 0.5*dw;
r0_(1+ic) = r_pre; r1_(1+ic) = r_pos;
w0_(1+ic) = w_pre; w1_(1+ic) = w_pos;
nc = max(1,min(ncra,floor(ncra*(S_p_(1+ic) - min(clim))/diff(clim))));
C_(1,1+ic,1:3) = cra_(nc,:);
ic=ic+1;
end;%for nw=0:n_w_(1+nr)-1;
end;%for nr=0:n_r-1;
X_ = [r0_.*cos(w0_) ; r1_.*cos(w0_) ; r1_.*cos(w1_) ; r0_.*cos(w1_) ; r0_.*cos(w0_) ];
Y_ = [r0_.*sin(w0_) ; r1_.*sin(w0_) ; r1_.*sin(w1_) ; r0_.*sin(w1_) ; r0_.*sin(w0_) ];
p=patch(X_,Y_,C_); set(p,'EdgeColor','none');
'''
import matplotlib.pyplot as plt

def imagesc_p(
        ax,
        n_r,
        grid_p_,
        n_w_,
        n_w_sum,
        S_p_,
        clim_=None,
        c_3c__=None,
):
    if clim_ is None: clim_ = torch.mean(S_p_).item() + 2.5 * torch.std(S_p_).item() * torch.tensor([-1,+1]).to(dtype=torch.float32);
    if c_3c__ is None: c_3c__ = torch.tensor(plt.cm.viridis(np.linspace(0, 1, 256))).to(dtype=torch.float32);  # Default colormap

    r0_ = torch.zeros(n_w_sum).to(dtype=torch.float32)
    r1_ = torch.zeros(n_w_sum).to(dtype=torch.float32)
    w0_ = torch.zeros(n_w_sum).to(dtype=torch.float32)
    w1_ = torch.zeros(n_w_sum).to(dtype=torch.float32)
    n_c = int(c_3c__.size()[0]);
    C_3a__ = torch.zeros(mtr((n_3,n_w_sum))).to(dtype=torch.float32);
    nw_sum = 0;
    patches = [];
    for nr in range(n_r):
        r = grid_p_[nr].item();
        r_pre = grid_p_[ 0].item() if nr ==       0 else 0.5 * (grid_p_[nr - 1].item() + grid_p_[nr].item());
        r_pos = grid_p_[-1].item() if nr == n_r - 1 else 0.5 * (grid_p_[nr + 1].item() + grid_p_[nr].item());
        if n_r == 1:
            r_pre = grid_p_[-1] / 4.0 ;
            r_pos = grid_p_[-1] / 1.0 ;
        n_w = int(n_w_[nr].item()) ;
        dw = 2 * pi / np.maximum(1,n_w) ;
        for nw in range(n_w):
            w_pre = nw * dw - 0.5 * dw ;
            w_pos = nw * dw + 0.5 * dw ;
            r0_[nw_sum] = r_pre ;
            r1_[nw_sum] = r_pos ;
            w0_[nw_sum] = w_pre ;
            w1_[nw_sum] = w_pos ;
            nc = int(np.maximum(0, np.minimum(n_c - 1, int(n_c * (S_p_[nw_sum].item() - clim_[0].item()) / np.maximum(1e-12,clim_[1].item() - clim_[0].item())))));
            C_3a__[nw_sum, :] = c_3c__[nc, :] ;
            x = [ r0_[nw_sum].item() * np.cos(w0_[nw_sum].item()), r1_[nw_sum].item() * np.cos(w0_[nw_sum].item()), r1_[nw_sum].item() * np.cos(w1_[nw_sum].item()), r0_[nw_sum].item() * np.cos(w1_[nw_sum].item()), r0_[nw_sum].item() * np.cos(w0_[nw_sum].item()) ] ;
            y = [ r0_[nw_sum].item() * np.sin(w0_[nw_sum].item()), r1_[nw_sum].item() * np.sin(w0_[nw_sum].item()), r1_[nw_sum].item() * np.sin(w1_[nw_sum].item()), r0_[nw_sum].item() * np.sin(w1_[nw_sum].item()), r0_[nw_sum].item() * np.sin(w0_[nw_sum].item()) ] ;
            polygon = Polygon(np.column_stack((x, y)), closed=True) ;
            patches.append(polygon) ;
            nw_sum += 1 ;
        #end;%for nw in range(n_w):
    #end;%for nr in range(n_r):
    assert(nw_sum==n_w_sum);

    p = PatchCollection(patches, edgecolor='none', linewidth=0) ;
    p.set_facecolor(C_3a__.numpy()) ;
    ax.add_collection(p) ;
    ax.set_aspect('equal') ;
