import torch;
numel = lambda a : int(a.numel()) ;
efind = lambda a : torch.where(a)[0] ;

#%%%%%%%%;
#% Warning, This is a brute-force O(n^2) intersection function. ;
#% The only goal is for this to mimic the matlab intersection. ;
#%%%%%%%%;
def intersect_0(a_, b_):
    u_a_,index_nu_a_from_na_,n_u_a_ = torch.unique(a_.ravel(),return_inverse=True,return_counts=True,dim=0);
    u_b_,index_nu_b_from_nb_,n_u_b_ = torch.unique(b_.ravel(),return_inverse=True,return_counts=True,dim=0);
    u_a_u_b_ = torch.concatenate([u_a_.ravel(), u_b_.ravel()],0);
    u_u_a_u_b_ , index_nu_u_a_u_b_from_nu_a_u_b_ , n_u_u_a_u_b_ = torch.unique(u_a_u_b_.ravel(),return_inverse=True,return_counts=True,dim=0);
    index_cap_ = efind(n_u_u_a_u_b_> 1);
    cap_ = u_u_a_u_b_.ravel()[index_cap_];
    n_cap = numel(cap_);
    index_na_from_ncap_ = torch.zeros(n_cap).to(dtype=torch.int32);
    index_nb_from_ncap_ = torch.zeros(n_cap).to(dtype=torch.int32);
    for ncap in range(n_cap):
        cap = cap_[ncap].item();
        na = int(torch.min(efind(a_==cap)).item());
        index_na_from_ncap_[ncap] = na;
        nb = int(torch.min(efind(b_==cap)).item());
        index_nb_from_ncap_[ncap] = nb;
    #end;%for ncap in range(n_cap):
    return(cap_,index_na_from_ncap_,index_nb_from_ncap_);
