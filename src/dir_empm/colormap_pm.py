from dir_empm.dir_matlab_macros import * ;

'''
function output = colormap_pm(n_c);
na=0;
if (nargin<1+na); n_c=[]; end; na=na+1;
if isempty(n_c); n_c = 64; end;

gamma1 = 0.5;
gamma2 = 0.25;
c__ = zeros(n_c,3);
m1 = floor(n_c/2);
c__(m1,:) = [1,1,1];
tmp_ = linspace(0,1,m1-1).^gamma1;
c__(1:m1-1,1) = transpose(tmp_).^6;
c__(1:m1-1,2) = 0.5 + 0.5*transpose(tmp_);
c__(1:m1-1,3) = ones(m1-1,1);
m2 = ceil(n_c/2);
c__(m2,:) = [1,1,1];
tmp_ = linspace(1,0,n_c-m2).^gamma2;
c__(m2+1:end,1) = ones(n_c-m2,1);
c__(m2+1:end,2) = transpose(tmp_).^6;
c__(m2+1:end,3) = 0.5 + 0.5*transpose(tmp_);
output = c__;
'''
def colormap_pm(n_c=64):
    gamma1 = 0.5;
    gamma2 = 0.25;
    c_3c__ = np.zeros((n_c, 3));
    m1 = n_c // 2;
    c_3c__[m1, :] = [1, 1, 1];
    tmp_ = np.linspace(0, 1, m1 - 1) ** gamma1;
    c_3c__[0:m1 - 1, 0] = tmp_ ** 6;
    c_3c__[0:m1 - 1, 1] = 0.5 + 0.5 * tmp_;
    c_3c__[0:m1 - 1, 2] = 1;
    m2 = (n_c + 1) // 2;
    c_3c__[m2 - 1, :] = [1, 1, 1];
    tmp_ = np.linspace(1, 0, n_c - m2) ** gamma2;
    c_3c__[m2:, 0] = 1;
    c_3c__[m2:, 1] = tmp_ ** 6;
    c_3c__[m2:, 2] = 0.5 + 0.5 * tmp_;
    c_3c__ = torch.tensor(c_3c__).to(dtype=torch.float32);
    return c_3c__;
