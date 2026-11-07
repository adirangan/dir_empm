from dir_empm.dir_matlab_macros import * ;
from dir_empm.i4_torch_arange import i4_torch_arange ;
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt

def imagesc_p(
        ax,
        n_k_p_r,
        k_p_r_,
        n_w_,
        n_w_sum,
        S_k_p_wk_,
        clim_=None,
        c_3c__=None,
):
    if clim_ is None: clim_ = torch.mean(S_k_p_wk_).item() + 2.5 * torch.std(S_k_p_wk_).item() * torch.tensor([-1,+1]).to(dtype=torch.float32);
    if c_3c__ is None: c_3c__ = colormap_beach();  # Default colormap

    r0_ = torch.zeros(n_w_sum).to(dtype=torch.float32)
    r1_ = torch.zeros(n_w_sum).to(dtype=torch.float32)
    w0_ = torch.zeros(n_w_sum).to(dtype=torch.float32)
    w1_ = torch.zeros(n_w_sum).to(dtype=torch.float32)
    n_c = int(c_3c__.size()[0]);
    c_3wk__ = torch.zeros(mtr((n_3,n_w_sum))).to(dtype=torch.float32);
    nw_sum = 0;
    patches = [];
    for nk_p_r in range(n_k_p_r):
        k_p_r = k_p_r_[nk_p_r].item();
        k_p_r_pre = k_p_r_[ 0].item() if nk_p_r ==       0 else 0.5 * (k_p_r_[nk_p_r - 1].item() + k_p_r_[nk_p_r].item());
        k_p_r_pos = k_p_r_[-1].item() if nk_p_r == n_k_p_r - 1 else 0.5 * (k_p_r_[nk_p_r + 1].item() + k_p_r_[nk_p_r].item());
        if n_k_p_r == 1:
            k_p_r_pre = k_p_r_[-1] / 4.0 ;
            k_p_r_pos = k_p_r_[-1] / 1.0 ;
        n_w = int(n_w_[nk_p_r].item()) ;
        dw = 2 * pi / np.maximum(1,n_w) ;
        for nw in range(n_w):
            w_pre = nw * dw - 0.5 * dw ;
            w_pos = nw * dw + 0.5 * dw ;
            r0_[nw_sum] = k_p_r_pre ;
            r1_[nw_sum] = k_p_r_pos ;
            w0_[nw_sum] = w_pre ;
            w1_[nw_sum] = w_pos ;
            nc = int(np.maximum(0, np.minimum(n_c - 1, int(n_c * (S_k_p_wk_[nw_sum].item() - clim_[0].item()) / np.maximum(1e-12,clim_[1].item() - clim_[0].item())))));
            c_3wk__[nw_sum, :] = c_3c__[nc, :] ;
            x = [ r0_[nw_sum].item() * np.cos(w0_[nw_sum].item()), r1_[nw_sum].item() * np.cos(w0_[nw_sum].item()), r1_[nw_sum].item() * np.cos(w1_[nw_sum].item()), r0_[nw_sum].item() * np.cos(w1_[nw_sum].item()), r0_[nw_sum].item() * np.cos(w0_[nw_sum].item()) ] ;
            y = [ r0_[nw_sum].item() * np.sin(w0_[nw_sum].item()), r1_[nw_sum].item() * np.sin(w0_[nw_sum].item()), r1_[nw_sum].item() * np.sin(w1_[nw_sum].item()), r0_[nw_sum].item() * np.sin(w1_[nw_sum].item()), r0_[nw_sum].item() * np.sin(w0_[nw_sum].item()) ] ;
            polygon = Polygon(np.column_stack((x, y)), closed=True) ;
            patches.append(polygon) ;
            nw_sum += 1 ;
        #end;%for nw in range(n_w):
    #end;%for nk_p_r in range(n_k_p_r):
    assert(nw_sum==n_w_sum);

    p = PatchCollection(patches, edgecolor='none', linewidth=0) ;
    p.set_facecolor(c_3wk__.numpy()) ;
    ax.add_collection(p) ;
    ax.set_aspect('equal') ;
