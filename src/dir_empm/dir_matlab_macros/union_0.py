import torch;
numel = lambda a : int(a.numel()) ;
efind = lambda a : torch.where(a)[0] ;

#%%%%%%%%;
#% Warning, This is a brute-force O(n^2) union function. ;
#% The only goal is for this to mimic the matlab union. ;
#%%%%%%%%;
def union_0(a_, b_):
    u_a_,index_nu_a_from_na_,n_u_a_ = torch.unique(a_.ravel(),return_inverse=True,return_counts=True,dim=0);
    u_b_,index_nu_b_from_nb_,n_u_b_ = torch.unique(b_.ravel(),return_inverse=True,return_counts=True,dim=0);
    u_a_u_b_ = torch.concatenate([u_a_.ravel(), u_b_.ravel()],0);
    u_u_a_u_b_ , index_nu_u_a_u_b_from_nu_a_u_b_ , n_u_u_a_u_b_ = torch.unique(u_a_u_b_.ravel(),return_inverse=True,return_counts=True,dim=0);
    cup_ = u_u_a_u_b_.ravel();
    n_cup = numel(cup_);
    index_na_from_ncup_ = -torch.ones(n_cup).to(dtype=torch.int32);
    index_nb_from_ncup_ = -torch.ones(n_cup).to(dtype=torch.int32);
    for ncup in range(n_cup):
        cup = cup_[ncup].item();
        tmp_index_na_ = efind(a_==cup);
        if numel(tmp_index_na_)> 0:
            na = int(torch.min(tmp_index_na_).item());
            index_na_from_ncup_[ncup] = na;
        #end;%if (numel(tmp_index_na_)> 0;
        if numel(tmp_index_na_)==0:
            tmp_index_nb_ = efind(b_==cup);
            if numel(tmp_index_nb_)> 0:
                nb = int(torch.min(tmp_index_nb_).item());
                index_nb_from_ncup_[ncup] = nb;
            #end;%if (numel(tmp_index_nb_)> 0;
        #end;%if (numel(tmp_index_na_)==0;
    #end;%for ncup in range(n_cup):
    index_na_from_ncup_ = index_na_from_ncup_[efind(index_na_from_ncup_>=0)];
    index_nb_from_ncup_ = index_nb_from_ncup_[efind(index_nb_from_ncup_>=0)];
    return(cup_,index_na_from_ncup_,index_nb_from_ncup_);
