from dir_empm.dir_matlab_macros import * ;

'''
function output = colormap_81s(n_c,gamma1,gamma2,gamma3);
if (nargin<1); n_c = 64; end;
%if (nargin<2); gamma1 = 0.25; end;
%if (nargin<3); gamma2 = 0.75; end;
%if (nargin<4); gamma3 = 0.50; end;
if (nargin<2); gamma1 = 0.50; end;
if (nargin<3); gamma2 = 1.50; end;
if (nargin<4); gamma3 = 1.50; end;

%{
  n_c = 64;
  imagesc(1:n_c);
  figure(2);
  c_ = colormap_81s;
  plot(1:n_c,c_(:,1),'r-',1:n_c,c_(:,2),'g-',1:n_c,c_(:,3),'b-');
  figure(2);set(gcf,'Position',512+1+[0,0,512,512]);
  figure(1);
  fig81s;
  figure(1);set(gcf,'Position',1+[0,0,512,512]);
  %}

if n_c<0;
gamma1_ = linspace(0,2,9);
gamma2_ = linspace(0,2,9);
gamma3_ = [0.25:0.25:1.75]; n_g3 = numel(gamma3_);
for ng3=0:n_g3-1;
figure(1+ng3);clf;
gamma3 = gamma3_(1+ng3);
n_c = 64;
p_ = 0:n_c-1;
x_ = [0+p_;1+p_;1+p_;0+p_;0+p_];
y_ = repmat([0;0;1;1;0],[1,n_c]);
for ng1=0:8; for ng2=0:8;
gamma1 = gamma1_(1+ng1);
gamma2 = gamma2_(1+ng2);
subplot(9,9,1+ng1+ng2*9);
c_ = reshape(colormap_81s(n_c,gamma1,gamma2,gamma3),[1,n_c,3]);
p_ = patch(x_,y_,c_); set(p_,'EdgeColor','none');
title(sprintf('%0.2f %0.2f',gamma1,gamma2));
axisnotick;
end;end;%for ng1=0:8; for ng2=0:8;
figbig;
sgtitle(sprintf('gamma3 = %0.2f',gamma3));
end;%for ng3=0:n_g3-1;
disp('returning'); return;
end;%if n_c<0;

c_ = zeros(n_c,3);
%%%%;
m00 = max(1,ceil(0*n_c/3));
m33 = floor(1*n_c/3);
tmp1_ = linspace(0,1,m33-m00+1).^gamma1;
tmp2_ = linspace(0,1,m33-m00+1).^gamma2;
tmp3_ = linspace(0,1,m33-m00+1).^gamma3;
c_(m00:m33,1) = tmp2_;
c_(m00:m33,2) = 0;
c_(m00:m33,3) = tmp1_;
%%%%;
m33 = ceil(1*n_c/3);
m67 = floor(2*n_c/3);
tmp1_ = linspace(0,1,m67-m33+1).^gamma1;
tmp2_ = linspace(0,1,m67-m33+1).^gamma2;
tmp3_ = linspace(1,0,m67-m33+1).^gamma3;
c_(m33:m67,1) = 1;
c_(m33:m67,2) = tmp2_;
c_(m33:m67,3) = tmp3_;
%%%%;
m67 = ceil(2*n_c/3);
m99 = min(n_c,floor(3*n_c/3));
tmp1_ = linspace(0,1,m99-m67+1).^gamma1;
tmp2_ = linspace(0,1,m99-m67+1).^gamma2;
tmp3_ = linspace(0,1,m99-m67+1).^gamma3;
c_(m67:m99,1) = 1;
c_(m67:m99,2) = 1;
c_(m67:m99,3) = tmp2_;
%%%%;
%alpha0=-1.0; beta0=2.0; tmp0_ = exp(beta0*(linspace(0,1,n_c)-alpha0)); tmp0_ = tmp0_./(1+tmp0_);
%alpha0=-1.0; beta0=1.5; tmp0_ = exp(beta0*(linspace(0,1,n_c)-alpha0)); tmp0_ = tmp0_./(1+tmp0_);
%c_(:,1+0) = c_(:,1+0).*transpose(tmp0_);
alpha0=+0.0; beta0=8.5; tmp0_ = exp(beta0*(linspace(0,1,n_c)-alpha0)); tmp0_ = tmp0_./(1+tmp0_);
c_(:,1+0) = 2.0*(tmp0_-0.5);
alpha1=0.45; beta1=10.0; tmp1_ = exp(beta1*(linspace(0,1,n_c)-alpha1)); tmp1_ = tmp1_./(1+tmp1_);
%c_(:,1+1) = 0.5*c_(:,1+1) + 0.5*transpose(tmp1_);
c_(:,1+1) = transpose(tmp1_);
%%%%%%%%;
dt = 0.2;
x_ = c_(:,1+2);
k_ = periodize(transpose(0:n_c-1),-n_c/2,n_c/2);
dd = @(x_) ifft(fft(x_).*exp(-dt*k_.^2));
y_ = dd(x_);
tmp_ij_ = 1:min(find(y_<x_));
y_(tmp_ij_) = min(x_(tmp_ij_),y_(tmp_ij_));
tmp_ij_ = max(find(y_>x_)):n_c;
y_(tmp_ij_) = max(x_(tmp_ij_),y_(tmp_ij_));
%plot(1:n_c,c_(:,1+2),'b-',1:n_c,y_,'ko');
c_(:,1+2) = y_;
%%%%%%%%;
output = c_;
'''

def colormap_81s(n_c=64, gamma1=0.5, gamma2=1.5, gamma3=1.5):
    if n_c < 0:
        gamma1_ = np.linspace(0, 2, 9);
        gamma2_ = np.linspace(0, 2, 9);
        gamma3_ = np.arange(0.25, 1.75 + 0.25, 0.25);
        n_g3 = numel(gamma3_);
        for ng3 in range(n_g3):
            gamma3 = gamma3_[ng3];
            n_c = 64;
            for ng1 in range(9):
                for ng2 in range(9):
                    gamma1 = gamma1_[ng1];
                    gamma2 = gamma2_[ng2];
                    c_3c__ = colormap_81s(n_c, gamma1, gamma2, gamma3);
        return;

    c_3c__ = np.zeros((n_c, 3));
    m00 = max(1, int(np.ceil(0 * n_c / 3)));
    m33 = int(np.floor(1 * n_c / 3));
    tmp1_ = np.linspace(0, 1, m33 - m00 + 0) ** gamma1;
    tmp2_ = np.linspace(0, 1, m33 - m00 + 0) ** gamma2;
    tmp3_ = np.linspace(0, 1, m33 - m00 + 0) ** gamma3;
    c_3c__[m00:m33, 0] = tmp2_;
    c_3c__[m00:m33, 1] = 0;
    c_3c__[m00:m33, 2] = tmp1_;

    m33 = int(np.ceil(1 * n_c / 3));
    m67 = int(np.floor(2 * n_c / 3));
    tmp1_ = np.linspace(0, 1, m67 - m33 + 0) ** gamma1;
    tmp2_ = np.linspace(0, 1, m67 - m33 + 0) ** gamma2;
    tmp3_ = np.linspace(1, 0, m67 - m33 + 0) ** gamma3;
    c_3c__[m33:m67, 0] = 1;
    c_3c__[m33:m67, 1] = tmp2_;
    c_3c__[m33:m67, 2] = tmp3_;

    m67 = int(np.ceil(2 * n_c / 3));
    m99 = min(n_c, int(np.floor(3 * n_c / 3)));
    tmp1_ = np.linspace(0, 1, m99 - m67 + 0) ** gamma1;
    tmp2_ = np.linspace(0, 1, m99 - m67 + 0) ** gamma2;
    tmp3_ = np.linspace(0, 1, m99 - m67 + 0) ** gamma3;
    c_3c__[m67:m99, 0] = 1;
    c_3c__[m67:m99, 1] = 1;
    c_3c__[m67:m99, 2] = tmp2_;

    alpha0 = 0.0;
    beta0 = 8.5;
    tmp0_ = np.exp(beta0 * (np.linspace(0, 1, n_c) - alpha0));
    tmp0_ = tmp0_ / (1 + tmp0_);
    c_3c__[:, 0] = 2.0 * (tmp0_ - 0.5);

    alpha1 = 0.45;
    beta1 = 10.0;
    tmp1_ = np.exp(beta1 * (np.linspace(0, 1, n_c) - alpha1));
    tmp1_ = tmp1_ / (1 + tmp1_);
    c_3c__[:, 1] = tmp1_;

    dt = 0.2;
    x_ = c_3c__[:, 2];
    k_ = np.fft.fftfreq(n_c) * n_c;
    dd = lambda x_: np.fft.ifft(np.fft.fft(x_) * np.exp(-dt * k_**2)).real;
    y_ = dd(x_);
    tmp_ij_ = np.where(y_ < x_)[0];
    if len(tmp_ij_) > 0:
        y_[tmp_ij_] = np.minimum(x_[tmp_ij_], y_[tmp_ij_]);
    tmp_ij_ = np.where(y_ > x_)[0];
    if len(tmp_ij_) > 0:
        y_[tmp_ij_] = np.maximum(x_[tmp_ij_], y_[tmp_ij_]);
    c_3c__[:, 2] = y_;

    c_3c__ = torch.tensor(c_3c__).to(dtype=torch.float32);
    return c_3c__;
