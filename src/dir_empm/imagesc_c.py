from dir_empm.dir_matlab_macros import * ;
from dir_empm.i4_torch_arange import i4_torch_arange ;
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

'''
function imagesc_c(n_x_0,x_0_,n_x_1,x_1_,S_c__,clim,cra_);
% cartesian imagesc;
% assumes S_c__(nx_0,nx_1) = S_c__(nx_0 + nx_1*n_x_0);

if (nargin<1);
n_x_0 = 24+1; x_0_ = transpose(linspace(-1,+1,n_x_0));
n_x_1 = 16+1; x_1_ = transpose(linspace(-1,+1,n_x_1));
[x_0__,x_1__] = ndgrid(x_0_,x_1_);
S_c__ = (x_0__ + x_1__)./2;
imagesc_c(n_x_0,x_0_,n_x_1,x_1_,S_c__,[-1,+1],colormap_beach());
axis image; axisnotick;
return;
end;%if (nargin<1);

na=0;
if (nargin<1+na); n_x_0=[]; end; na=na+1;
if (nargin<1+na); x_0_=[]; end; na=na+1;
if (nargin<1+na); n_x_1=[]; end; na=na+1;
if (nargin<1+na); x_1_=[]; end; na=na+1;
if (nargin<1+na); S_c__=[]; end; na=na+1;
if (nargin<1+na); clim=[]; end; na=na+1;
if (nargin<1+na); cra_=[]; end; na=na+1;

if isempty(S_c__); S_c__ = zeros(2,2); end;
if isempty(n_x_0); if size(S_c__,2)>1; n_x_0 = size(S_c__,1); else; n_x_0 = fix(sqrt(numel(S_c__))); end; end;
if isempty(n_x_1); if size(S_c__,2)>1; n_x_1 = size(S_c__,2); else; n_x_0 = fix(numel(S_c__)/n_x_0); end; end;
if isempty(x_0_); x_0_ = linspace(-1,+1,n_x_0); end;
if isempty(x_1_); x_1_ = linspace(-1,+1,n_x_1); end;

if isempty(clim); 
clim = mean(S_c__(:)) + std(S_c__(:))*2.5*[-1,1]; 
end;%if isempty(clim);
if isempty(cra_); 
cra_ = colormap_beach();
end;%if isempty(clim);
dx_0_ = diff(x_0_); dx_1_ = diff(x_1_);
x_00_ = zeros(1,n_x_0*n_x_1);
x_01_ = zeros(1,n_x_0*n_x_1);
x_10_ = zeros(1,n_x_0*n_x_1);
x_11_ = zeros(1,n_x_0*n_x_1);
ncra = size(cra_,1);
C_ = zeros(1,n_x_0*n_x_1,3);
ic=0;
for nx_1=0:n_x_1-1;
if (nx_1==0); x_1_pre = x_1_(1)-0.5*dx_1_(1); else; x_1_pre = 0.5*(x_1_(1+nx_1-1)+x_1_(1+nx_1)); end;
if (nx_1==n_x_1-1); x_1_pos = x_1_(end)+0.5*dx_1_(end); else; x_1_pos = 0.5*(x_1_(1+nx_1+1)+x_1_(1+nx_1)); end;
for nx_0=0:n_x_0-1;
if (nx_0==0); x_0_pre = x_0_(1)-0.5*dx_0_(1); else; x_0_pre = 0.5*(x_0_(1+nx_0-1)+x_0_(1+nx_0)); end;
if (nx_0==n_x_0-1); x_0_pos = x_0_(end)+0.5*dx_0_(end); else; x_0_pos = 0.5*(x_0_(1+nx_0+1)+x_0_(1+nx_0)); end;
x_00_(1+ic) = x_0_pre; x_01_(1+ic) = x_0_pos;
x_10_(1+ic) = x_1_pre; x_11_(1+ic) = x_1_pos;
nc = max(1,min(ncra,floor(ncra*(S_c__(1+ic) - min(clim))/diff(clim))));
C_(1,1+ic,1:3) = cra_(nc,:);
ic = ic+1;
end;%for nx_0=0:n_x_0-1;
end;%for nx_1=0:n_x_1-1;
X_0_ = [x_00_ ; x_00_ ; x_01_ ; x_01_ ; x_00_];
X_1_ = [x_10_ ; x_11_ ; x_11_ ; x_10_ ; x_10_];
p=patch(X_0_,X_1_,C_); set(p,'EdgeColor','none');
'''
import matplotlib.pyplot as plt

def imagesc_c(
        ax,
        n_x_0=None,
        x_0_=None,
        n_x_1=None,
        x_1_=None,
        S_c__=None,
        clim_=None,
        c_3c__=None,
):
    if S_c__ is None: S_c__ = torch.zeros(mtr((2, 2))).to(dtype=torch.float32);
    if n_x_0 is None: n_x_0 = int(S_c__.size()[0]) ;
    if n_x_1 is None: n_x_1 = int(S_c__.size()[1]) ;
    if x_0_ is None: x_0_ = torch.linspace(-1, +1, n_x_0).to(dtype=torch.float32) ;
    if x_1_ is None: x_1_ = torch.linspace(-1, +1, n_x_1).to(dtype=torch.float32) ;
    if clim_ is None: clim_ = torch.mean(S_c__).item() + 2.5 * torch.std(S_c__).item() * torch.tensor([-1,+1]).to(dtype=torch.float32);
    if c_3c__ is None: c_3c__ = colormap_beach() ; # Default colormap

    dx_0_ = torch.diff(x_0_.ravel()).to(dtype=torch.float32) ;
    x_0_pre_ = x_0_ -  0.5 * torch.concatenate( ( torch.tensor([dx_0_[0]]) , dx_0_ ) , 0 );
    x_0_pos_ = x_0_ +  0.5 * torch.concatenate( ( dx_0_ , torch.tensor([dx_0_[n_x_0-1-1]]) ) , 0 );
    dx_1_ = torch.diff(x_1_.ravel()).to(dtype=torch.float32) ;
    x_1_pre_ = x_1_ -  0.5 * torch.concatenate( ( torch.tensor([dx_1_[0]]) , dx_1_ ) , 0 );
    x_1_pos_ = x_1_ +  0.5 * torch.concatenate( ( dx_1_ , torch.tensor([dx_1_[n_x_1-1-1]]) ) , 0 );
    x_1_prepre__,x_0_prepre__ = torch.meshgrid(x_1_pre_,x_0_pre_,indexing='ij');
    x_1_prepos__,x_0_prepos__ = torch.meshgrid(x_1_pos_,x_0_pre_,indexing='ij');
    x_1_pospre__,x_0_pospre__ = torch.meshgrid(x_1_pre_,x_0_pos_,indexing='ij');
    x_1_pospos__,x_0_pospos__ = torch.meshgrid(x_1_pos_,x_0_pos_,indexing='ij');
    n_c = int(c_3c__.size()[0]);
    nc_ = torch.maximum(torch.tensor(0.0),torch.minimum(torch.tensor(n_c-1),torch.floor(n_c*(S_c__.ravel()-clim_[0].item())/np.maximum(1e-12,clim_[1].item()-clim_[0].item())))).to(dtype=torch.int32);
    C_3a__ = torch.zeros(mtr((n_3,n_x_0*n_x_1))).to(dtype=torch.float32);
    patches = [];
    for na in range(n_x_0*n_x_1):
        nc = int(nc_[na].item());
        C_3a__[na,:] = c_3c__[nc,:];
        polygon = Polygon( [ [x_0_prepre__.ravel()[na].item(),x_1_prepre__.ravel()[na].item()] , [x_0_prepos__.ravel()[na].item(),x_1_prepos__.ravel()[na].item()] , [x_0_pospos__.ravel()[na].item(),x_1_pospos__.ravel()[na].item()] , [x_0_pospre__.ravel()[na].item(),x_1_pospre__.ravel()[na].item()] ], closed=True ) ;
        patches.append(polygon) ;
    #end;%for na in range(n_x_0*n_x_1):

    p = PatchCollection(patches, edgecolor='none', linewidth=0) ;
    p.set_facecolor(C_3a__.numpy()) ;
    ax.add_collection(p) ;
    ax.set_aspect('equal') ;
