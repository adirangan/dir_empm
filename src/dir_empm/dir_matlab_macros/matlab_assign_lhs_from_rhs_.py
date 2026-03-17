import numpy as np ; pi = np.pi ; import torch ;

#%%%%%%%%;
#% This ensures that ravel() does not inadvertently make a copy of the lhs. ;
#%%%%%%%%;

def matlab_assign_lhs_from_rhs_(
        array_lhs_=None,
        index_lhs_=None,
        array_rhs_=None,
):
    tmp_shape_ = array_lhs_.shape;
    array_lhs_ = array_lhs_.ravel(); #%<-- force ravel of original. ;
    array_lhs_[index_lhs_] = array_rhs_;
    array_lhs_ = torch.reshape(array_lhs_,tmp_shape_);
    return(array_lhs_);
