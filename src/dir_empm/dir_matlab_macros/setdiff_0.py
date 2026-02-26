import torch;
numel = lambda a : int(a.numel()) ;
efind = lambda a : torch.where(a)[0] ;
from . union_0 import union_0 ;

#%%%%%%%%;
#% Warning, This is a brute-force O(n^2) setdiff function. ;
#% The only goal is for this to mimic the matlab setdiff. ;
#%%%%%%%%;
def setdiff_0(a_, b_):
    cup_,index_nb_from_ncup_,index_na_from_ncup_ = union_0(b_,a_);
    setdiff_ = a_.ravel()[index_na_from_ncup_];
    return(setdiff_);

