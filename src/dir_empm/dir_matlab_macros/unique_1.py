import torch;
numel = lambda a : int(a.numel()) ;
efind = lambda a : torch.where(a)[0] ;

#%%%%%%%%;
#% Warning, This is a simple O(n^1) unique function. ;
#% The only goal is for this to mimic the matlab unique. ;
#%%%%%%%%;
def unique_1(a_):
    u_a_,index_nu_a_from_na_,n_u_a_ = torch.unique(a_.ravel(),return_inverse=True,return_counts=True,dim=0);
    n_a = numel(a_);
    n_u_a = numel(u_a_);
    index_nu_a_from_na_ = index_nu_a_from_na_.to(dtype=torch.int32);
    index_na_from_nu_a_ = torch.zeros(n_u_a).to(dtype=torch.int32);
    flag_filled_ = torch.zeros(n_u_a).to(dtype=torch.int32);
    tmp_n_u_a = n_u_a; na=0;
    while ((tmp_n_u_a>=0) & (na<=n_a-1)):
        a = a_[na];
        nu_a = index_nu_a_from_na_[na];
        flag_filled = flag_filled_[nu_a];
        if (flag_filled!=1):
            flag_filled_[nu_a] = 1; flag_filled = 1;
            index_na_from_nu_a_[nu_a] = na;
            tmp_n_u_a = tmp_n_u_a - 1;
        #end;%if (flag_filled!=1);
        na=na+1;
    #end;%while ((tmp_n_u_a>=0) & (na<=n_a-1));
    return(u_a_,index_na_from_nu_a_,index_nu_a_from_na_);
