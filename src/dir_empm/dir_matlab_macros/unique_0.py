import torch;
numel = lambda a : int(a.numel()) ;
efind = lambda a : torch.where(a)[0] ;

#%%%%%%%%;
#% Warning, This is a brute-force O(n^2) unique function. ;
#% The only goal is for this to mimic the matlab unique. ;
#%%%%%%%%;
def unique_0(a_):
    u_a_,index_nu_a_from_na_,n_u_a_ = torch.unique(a_.ravel(),return_inverse=True,return_counts=True,dim=0);
    n_u_a = numel(u_a_);
    index_nu_a_from_na_ = index_nu_a_from_na_.to(dtype=torch.int32);
    index_na_from_nu_a_ = torch.zeros(n_u_a).to(dtype=torch.int32);
    for nu_a in range(n_u_a):
        u_a = u_a_[nu_a].item();
        na = int(torch.min(efind(a_==u_a)).item());
        index_na_from_nu_a_[nu_a] = na;
    #end;%for nu_a=0:n_u_a-1;
    return(u_a_,index_na_from_nu_a_,index_nu_a_from_na_);
