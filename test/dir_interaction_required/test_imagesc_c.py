from dir_empm.dir_matlab_macros import * ;

'''
n_x = 24+1; x_ = transpose(linspace(-1,+1,n_x));
n_y = 16+1; y_ = transpose(linspace(-1,+1,n_y));
[x__,y__] = ndgrid(x_,y_);
S_c__ = (x__ + y__)./2;
imagesc_c(n_x,x_,n_y,y_,S_c__,[-1,+1],colormap_beach());
axis image; axisnotick;
return;
'''
import matplotlib.pyplot as plt;
from dir_empm.imagesc_c import imagesc_c;
from dir_empm.colormap_beach import colormap_beach;
from dir_empm.colormap_pm import colormap_pm;
from dir_empm.colormap_80s import colormap_80s;
from dir_empm.colormap_81s import colormap_81s;

n_x = int(24 + 1);
x_ = torch.linspace(-1,+1,n_x).to(dtype=torch.float32);
n_y = int(16 + 1);
y_ = torch.linspace(-1,+1,n_y).to(dtype=torch.float32);
y__,x__ = torch.meshgrid(y_, x_, indexing='ij');
S_c__ = (x__ + y__).to(dtype=torch.complex64) / 2 ;

fig = plt.figure(figsize=(8, 6)) ;
p_row = 2; p_col = 2; np=0;
for np in range(p_row*p_col):
    ax = fig.add_subplot(p_row,p_col,1+np);
    if np==0: imagesc_c(ax,n_x,x_,n_y,y_,torch.real(S_c__),torch.tensor([-1,+1]),colormap_beach()); ax.set_title('colormap_beach')
    if np==1: imagesc_c(ax,n_x,x_,n_y,y_,torch.real(S_c__),torch.tensor([-1,+1]),colormap_pm()); ax.set_title('colormap_pm')
    if np==2: imagesc_c(ax,n_x,x_,n_y,y_,torch.real(S_c__),torch.tensor([-1,+1]),colormap_80s()); ax.set_title('colormap_80s')
    if np==3: imagesc_c(ax,n_x,x_,n_y,y_,torch.real(S_c__),torch.tensor([-1,+1]),colormap_81s()); ax.set_title('colormap_81s')
    ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); 
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xticklabels([]); ax.set_yticklabels([]);
#end;%for np in range(p_row*p_col):
plt.show();
