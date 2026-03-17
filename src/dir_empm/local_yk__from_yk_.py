from dir_empm.dir_matlab_macros import * ;

def local_yk__from_yk_(
        n_k_p_r,
        l_max_,
        tmp_yk_,
):
    n_y_ = (l_max_+1)**2;
    n_y_max = int(torch.max(n_y_).item());
    n_y_sum = int(torch.sum(n_y_).item());
    n_y_csum_ = cumsum_0(n_y_);
    l_max_max = int(torch.max(l_max_).item());
    tmp_yk__ = torch.zeros(mtr((n_y_max,n_k_p_r))).to(dtype=torch.complex64);
    for nk_p_r in range(n_k_p_r):
        n_y = int(n_y_[nk_p_r].item());
        tmp_i8_index_rhs_ = int(n_y_csum_[nk_p_r].item()) + torch.arange(n_y);
        tmp_i8_index_lhs_ = matlab_index_2d_0(n_y_max,torch.arange(n_y),n_k_p_r,nk_p_r);
        #tmp_yk__.ravel()[tmp_i8_index_lhs_] = tmp_yk_[tmp_i8_index_rhs_].to(dtype=torch.complex64);
        tmp_yk__ = matlab_assign_lhs_from_rhs_(tmp_yk__,tmp_i8_index_lhs_, tmp_yk_[tmp_i8_index_rhs_].to(dtype=torch.complex64));
    #end;%for nk_p_r=0:n_k_p_r-1;
    return(tmp_yk__) ;
