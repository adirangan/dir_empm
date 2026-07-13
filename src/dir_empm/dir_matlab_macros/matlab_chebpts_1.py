import numpy as np ; import torch ;
isempty = lambda t: (t is None) or (isinstance(t, torch.Tensor) and t.numel() == 0) ;

def matlab_chebpts_1(
        N=None,
        I_=None,
        flag_type=None,
):

    if isempty(I_):
        I_=torch.tensor([-1,+1]).to(dtype=torch.float32);
    #end;%if isempty(I_):
    I_rad = (I_[1] - I_[0])/2.0;
    I_mid = 0.5*torch.sum(I_).item();
    
    if isempty(flag_type):
        flag_type = 1;
    #end;%if isempty(flag_type);
    if (flag_type != 1):
        print(' % Warning: flag type not implemented in matlab_chebpts_1');
    #end;%if (flag_type != 1);
    
    x_ = torch.tensor(0).to(dtype=torch.float32);
    if N==1: x_ = torch_zeros(1).to(dtype=torch.float32); #end;
    if N> 1:
        x_ = torch.sin(np.pi*(torch.arange(-N+1,+N+1,2)/np.maximum(1,2*N))).to(dtype=torch.float32);
    #end;%if N> 1;
    w_ = torch.tensor(0).to(dtype=torch.float32);
    if N==1: w_ = 2.0*torch_ones(1).to(dtype=torch.float32); #end;
    if N> 1:
        m_ = 2.0 / torch.cat((torch.tensor([1]),1-torch.arange(2,N,2)**2),0).to(dtype=torch.float32);
        if np.mod(N,2)==0: c_ = torch.cat((m_,torch.tensor([0]),-m_[torch.arange(N/2,1,-1).to(dtype=torch.int64)-1]),0).to(dtype=torch.float32); #end;
        if np.mod(N,2)==1: c_ = torch.cat((m_,-m_[torch.arange((N+1)/2,1,-1).to(dtype=torch.int64)-1]),0).to(dtype=torch.float32); #end;
        w_ = torch.real(torch.fft.ifft(c_ * torch.exp(1j * torch.arange(N)*np.pi/np.maximum(1,N)))).to(dtype=torch.float32);
    #end;%if N> 1;
    x_ = x_*I_rad + I_mid ;
    w_ = w_*I_rad ;
    x_ = x_.ravel(); #%<-- unravel both, rather than matching chebpts. ;
    w_ = w_.ravel(); #%<-- unravel both, rather than matching chebpts. ;

    return(
        x_,
        w_,
    );
