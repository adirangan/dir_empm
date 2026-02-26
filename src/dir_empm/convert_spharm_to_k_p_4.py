from dir_empm.dir_matlab_macros import * ;
from dir_empm.get_Ylm__2 import get_Ylm__2 ;
from dir_empm.get_Ylm_condense_wrap_0 import get_Ylm_condense_wrap_0 ;

def convert_spharm_to_k_p_4(
        flag_verbose=None,
        n_qk=None,
        n_qk_csum_=None,
        k_p_r_qk_=None,
        k_p_azimu_b_qk_=None,
        k_p_polar_a_qk_=None,
        weight_3d_k_p_qk_=None,
        weight_shell_qk_=None,
        n_k_p_r=None,
        k_p_r_=None,
        weight_3d_k_p_r_=None,
        l_max_=None,
        a_k_Y_=None,
        Ylm_uklma___=None,
        k_p_azimu_b_sub_uka__=None,
        k_p_polar_a_sub_uka__=None,
        l_max_uk_=None,
        index_nu_n_k_per_shell_from_nk_p_r_=None,
        index_k_per_shell_uka__=None,
):
    str_thisfunction = 'convert_spharm_to_k_p_4' ;
    flag_Ylm_create = 0; 
    if Ylm_uklma___ is None:
        flag_Ylm_create=1;
        (
            Ylm_uklma___,
            k_p_azimu_b_sub_uka__,
            k_p_polar_a_sub_uka__,
            l_max_uk_,
            index_nu_n_k_per_shell_from_nk_p_r_,
            index_k_per_shell_uka__,
        ) = get_Ylm_condense_wrap_0(
            flag_verbose,
            n_qk,
            n_qk_csum_,
            k_p_azimu_b_qk_,
            k_p_polar_a_qk_,
            n_k_p_r,
            l_max_,
        );
    #end;%if isempty(Ylm_uklma___);
    n_y_ = (l_max_+1)**2;
    if (flag_verbose): print(f' %% [entering {str_thisfunction}] n_qk {n_qk}, n_y_sum {torch.sum(n_y_).item()}');
    a_k_qk_ = torch.zeros(n_qk).to(dtype=torch.complex64);
    ix=0;
    for nk_p_r in range(n_k_p_r):
        k_p_r = k_p_r_[nk_p_r].item(); 
        n_qk_csum = int(n_qk_csum_[nk_p_r].item());
        if (flag_verbose>1): print(f' %% nk_p_r {nk_p_r}/{n_k_p_r} k_p_r {k_p_r:.2f}, n_qk_csum {n_qk_csum} --> {n_qk_csum/n_qk:.2f}%');
        if (nk_p_r<n_k_p_r-1):
            n_sub = int(n_qk_csum_[nk_p_r+1].item()) - int(n_qk_csum_[nk_p_r].item());
        else:
            n_sub = n_qk - n_qk_csum ;
        #end;%ifelse;
        index_sub_ = n_qk_csum + torch.arange(n_sub);
        k_p_r_sub_ = k_p_r_qk_[index_sub_]; 
        assert int(torch.sum(k_p_r_sub_==k_p_r).item())==n_sub;
        k_p_azimu_b_sub_ = k_p_azimu_b_qk_[index_sub_];
        k_p_polar_a_sub_ = k_p_polar_a_qk_[index_sub_];
        weight_shell_qk_sub_ = torch.reshape(weight_shell_qk_[index_sub_]/np.maximum(1e-12,k_p_r**2), mtr((n_sub,1)) );
        l_max = int(l_max_[nk_p_r].item());
        n_l_max = l_max+1; flag_flip=0;
        n_y = int(n_y_[nk_p_r].item());
        #%%%%;
        nu_n_k_per_shell = int(index_nu_n_k_per_shell_from_nk_p_r_[nk_p_r].item());
        tmp_k_p_azimu_b_sub_ = k_p_azimu_b_sub_uka__[nu_n_k_per_shell];
        tmp_k_p_polar_a_sub_ = k_p_polar_a_sub_uka__[nu_n_k_per_shell];
        if (fnorm(k_p_azimu_b_sub_-tmp_k_p_azimu_b_sub_)>1e-6):
            print(f' %% Warning, fnorm(k_p_azimu_b_sub_-tmp_k_p_azimu_b_sub_) {fnorm(k_p_azimu_b_sub_-tmp_k_p_azimu_b_sub_):.16f} in {str_thisfunction}') ;
        #end;%if (fnorm(k_p_azimu_b_sub_-tmp_k_p_azimu_b_sub_)>1e-6);
        if (fnorm(k_p_polar_a_sub_-tmp_k_p_polar_a_sub_)>1e-6):
            print(f' %% Warning, fnorm(k_p_polar_a_sub_-tmp_k_p_polar_a_sub_) {fnorm(k_p_polar_a_sub_-tmp_k_p_polar_a_sub_):.16f} in {str_thisfunction}') ;
        #end;%if (fnorm(k_p_polar_a_sub_-tmp_k_p_polar_a_sub_)>1e-6);
        tmp_matlab_size_rhs_ = mtr(Ylm_uklma___[nu_n_k_per_shell].size());
        tmp_i8_index_rhs_ = matlab_index_2d_0(tmp_matlab_size_rhs_[0],torch.arange(n_y),tmp_matlab_size_rhs_[1],':');
        Ylm_lma__ = torch.reshape(Ylm_uklma___[nu_n_k_per_shell].ravel()[tmp_i8_index_rhs_],mtr((n_y,tmp_matlab_size_rhs_[1])));
        #%%%%;
        tmp_t = timeit.default_timer();
        a_k_qk_sub_ = torch.zeros(mtr((1,n_sub))).to(dtype=torch.complex64);
        tmp_i8_index_rhs_ = torch.arange(ix,ix+n_y);
        str_einsum = msr('oy') + ',' + msr('ya') + '->' + msr('a') ;
        a_k_qk_sub_ = torch.einsum( str_einsum , torch.reshape(a_k_Y_[tmp_i8_index_rhs_],mtr((1,n_y))).to(dtype=torch.complex64) , Ylm_lma__.to(dtype=torch.complex64) ) ;
        ix = ix + n_y;
        a_k_qk_[index_sub_] = a_k_qk_sub_;
        del Ylm_lma__;
        tmp_t = timeit.default_timer()-tmp_t;
        if (flag_verbose): print(f' %% nk_p_r {nk_p_r}/{n_k_p_r} a_k_qk_sub_: {tmp_t:.6f}s');
    #end;%for nk_p_r=0:n_k_p_r-1;
    if (flag_verbose): print(f' %% [finished {str_thisfunction}] n_qk {n_qk}, n_y_sum {torch.sum(n_y_).item()}');
    return(
        a_k_qk_,
        Ylm_uklma___,
        k_p_azimu_b_sub_uka__,
        k_p_polar_a_sub_uka__,
        l_max_uk_,
        index_nu_n_k_per_shell_from_nk_p_r_,
        index_k_per_shell_uka__,
    );
