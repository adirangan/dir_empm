from dir_empm.dir_matlab_macros import * ;

'''
function output = colormap_beach(n_cra);
if (nargin<1); n_cra = 64; end;
if n_cra<0; imagesc(1:64); colormap(colormap_beach()); return; end;
gamma1 = 0.5;
gamma2 = 0.25;
cra = zeros(n_cra,3);
m1 = floor(n_cra/2);
cra(m1,:) = [1,1,1];
tmp_ = linspace(0,1,m1-1).^gamma1;
cra(1:m1-1,1) = transpose(tmp_).^6;
cra(1:m1-1,2) = 0.5 + 0.5*transpose(tmp_);
cra(1:m1-1,3) = ones(m1-1,1);
m2 = ceil(n_cra/2);
cra(m2,:) = [1,1,1];
tmp_ = linspace(1,0,n_cra-m2).^gamma2;
cra(m2+1:end,1) = ones(n_cra-m2,1);
cra(m2+1:end,2) = transpose(tmp_).^6;
cra(m2+1:end,3) = 0.5 + 0.5*transpose(tmp_);
cra(:,3) = cra(:,3).*transpose(sqrt(sqrt(linspace(-1,+1,n_cra).^2)));
output = cra;
'''
def colormap_beach(
        n_c=None,
):
    if isempty(n_c): n_c = 64; #end;
    if n_c< 0:
        import matplotlib.pyplot as plt ; from matplotlib.colors import ListedColormap ;
        plt.imshow(torch.reshape(torch.arange(1,64+1),mtr((1,64))).T.numpy(),aspect='auto',cmap=ListedColormap(colormap_beach().numpy())); plt.show(); #<-- similar to matlab: imagesc(1:64); colormap(colormap_beach()). ;
        return;
    #end;%if n_c< 0;
    gamma1 = 0.5;
    gamma2 = 0.25;
    c_3c__ = torch.zeros(mtr((3,n_c))).to(dtype=torch.float64);
    m1 = int(np.floor(n_c/2));
    c_3c__[m1-1,:] = torch.tensor([1,1,1]).to(dtype=torch.float64);
    tmp_ = matlab_linspace(0,1,m1-1,dtype=torch.float64)**gamma1;
    c_3c__[:m1-1,0] = tmp_**6;
    c_3c__[:m1-1,1] = 0.5 + 0.5*tmp_;
    c_3c__[:m1-1,2] = torch.ones(m1-1).to(dtype=torch.float64);
    m2 = int(np.ceil((n_c+1)/2));
    c_3c__[m2-1,:] = torch.tensor([1,1,1]).to(dtype=torch.float64);
    tmp_ = matlab_linspace(1,0,n_c-m2,dtype=torch.float64)**gamma2;
    c_3c__[m2:,0] = torch.ones(n_c-m2).to(dtype=torch.float64);
    c_3c__[m2:,1] = tmp_**6;
    c_3c__[m2:,2] = 0.5 + 0.5*tmp_;
    c_3c__[:,2] = c_3c__[:,2]*torch.sqrt(torch.sqrt(matlab_linspace(-1,+1,n_c,dtype=torch.float64)**2));
    c_3c__ = c_3c__.to(dtype=torch.float32);
    return(c_3c__);

