from dir_empm.dir_matlab_macros import * ;
import sys ;
from dir_empm.i4_torch_arange import i4_torch_arange ;

def tpmh_VUXM_lwnM____3(
        FTK=None,
        n_k_p_r=None,
        n_w_=None,
        n_M=None,
        M_k_q__=None,
        n_UX_rank=None,
        UX__=None,
        X_weight_r_=None,
):
    flag_verbose=0;
    str_thisfunction = 'tpmh_VUXM_lwnM____3';

    if flag_verbose > 0: print(f' %% [entering {str_thisfunction}]');

    n_w_max = int(torch.max(n_w_).item()); n_w_2 = matlab_scalar_round(n_w_max/2);
    l_max = int(torch.max(torch.abs(FTK['i4_svd_l_'])));
    #%%%%%%%%;

    tmp_t = tic();
    V_r__ = torch.reshape(FTK['r8_svd_chebval_V_r_'],mtr((FTK['n_svd_l'],n_k_p_r))).to(dtype=torch.float32);
    tmp_t = toc(tmp_t); 
    if (flag_verbose>0): print(f' %% V_r__ %0.6fs',tmp_t);
    V_UX_lrn___ = torch.zeros(mtr((FTK['n_svd_l'],n_k_p_r,n_UX_rank))).to(dtype=torch.float32);
    for nUX_rank in range(n_UX_rank):
        V_UX_lrn___[nUX_rank,:,:] = mmmm( V_r__ , diag( UX__[nUX_rank,:].ravel() * X_weight_r_.ravel() ).to(dtype=torch.float32) ) ;
    #end;%for nUX_rank=0:n_UX_rank-1;
    V_UX_nrl___ = torch.permute(V_UX_lrn___,mtr(mts((2,1,0))));
    #%%%%%%%%;
    index_nw_out__ = cell(1+2*l_max); #<-- cell array. ;
    index_nw_0in__ = cell(1+2*l_max); #<-- cell array. ;
    n_nw_ = torch.zeros(1+2*l_max).to(dtype=torch.int32);
    for l_shift in range(-l_max,+l_max+1):
        index_nw_out_head_ = i4_torch_arange(0,1+n_w_2-1-int(np.maximum(0,+l_shift))) ;
        index_nw_0in_head_ = torch.concatenate( ( i4_torch_arange(n_w_max+l_shift,1+n_w_max-1) , i4_torch_arange(int(np.maximum(0,+l_shift)),1+n_w_2-1-int(np.maximum(0,-l_shift))) ) , 0 ) ;
        index_nw_out_tail_ = i4_torch_arange(n_w_2+1+int(np.maximum(0,-l_shift)),1+n_w_max-1) ;
        index_nw_0in_tail_ = torch.concatenate( ( i4_torch_arange(n_w_2+1+int(np.maximum(0,+l_shift)),1+n_w_max-1-int(np.maximum(0,-l_shift))) , i4_torch_arange(0,1+l_shift-1) ) , 0 ) ;
        index_nw_out_ = torch.concatenate( ( index_nw_out_head_ , index_nw_out_tail_ ) , 0 ) ;
        index_nw_0in_ = torch.concatenate( ( index_nw_0in_head_ , index_nw_0in_tail_ ) , 0 ) ;
        n_nw = numel(index_nw_out_);
        n_nw_[l_max+l_shift] = n_nw;
        index_nw_out__[l_max+l_shift] = index_nw_out_;
        index_nw_0in__[l_max+l_shift] = index_nw_0in_;
    #end;%for l_shift=-l_max:+l_max;
    #%%%%%%%%;
    assert(np.mod(n_w_max,2)==0); #%<-- assert that n_w_max is even. ;
    index_nw_zerobased_from_centered_ = torch.concatenate( ( i4_torch_arange(n_w_2,1+n_w_max-1) , i4_torch_arange(0,1+n_w_2-1) ) , 0 ) ; #%<-- note that we place n_w_2 mode first, ;
    index_nw_centered_from_zerobased_ = torch.concatenate( ( i4_torch_arange(n_w_2,1+n_w_max-1) , i4_torch_arange(0,1+n_w_2-1) ) , 0 ) ; #%<-- note that we place first mode at n_w_2. ;
    index_nw_centered_out_start_ = torch.zeros(1+2*l_max).to(dtype=torch.int32);
    index_nw_centered_out_final_ = torch.zeros(1+2*l_max).to(dtype=torch.int32);
    index_nw_centered_0in_start_ = torch.zeros(1+2*l_max).to(dtype=torch.int32);
    index_nw_centered_0in_final_ = torch.zeros(1+2*l_max).to(dtype=torch.int32);
    for l_shift in range(-l_max,+l_max+1):
        index_nw_centered_out_start_[l_max+l_shift] = 1+int(np.maximum(0,-l_shift));
        index_nw_centered_out_final_[l_max+l_shift] = n_w_max-1+int(np.minimum(0,-l_shift));
        index_nw_centered_0in_start_[l_max+l_shift] = 1+int(np.maximum(0,+l_shift));
        index_nw_centered_0in_final_[l_max+l_shift] = n_w_max-1+int(np.minimum(0,+l_shift));
    #end;%for l_shift=-l_max:+l_max;
    #%%%%%%%%;
    M_k_q_centered_rwM___ = torch.permute(torch.reshape(M_k_q__,mtr((n_w_max,n_k_p_r,n_M))),mtr(mts((1,0,2))));
    tmp_i8_index_rhs_ = matlab_index_3d_0(n_k_p_r,':',n_w_max,index_nw_centered_from_zerobased_,n_M,':');
    M_k_q_centered_rwM___ = torch.reshape(M_k_q_centered_rwM___.ravel()[tmp_i8_index_rhs_],mtr((n_k_p_r,numel(index_nw_centered_from_zerobased_),n_M)));
    #%%%%%%%%;
    VUXM_centered_nwMl____ = torch.zeros(mtr((n_UX_rank,n_w_max,n_M,FTK['n_svd_l']))).to(dtype=torch.complex64);
    for nl in range(FTK['n_svd_l']):
        l_shift = int(FTK['i4_svd_l_'][nl]);
        index_nw_centered_out_start = int(index_nw_centered_out_start_[l_max+l_shift].item());
        index_nw_centered_out_final = int(index_nw_centered_out_final_[l_max+l_shift].item());
        index_nw_centered_0in_start = int(index_nw_centered_0in_start_[l_max+l_shift].item());
        index_nw_centered_0in_final = int(index_nw_centered_0in_final_[l_max+l_shift].item());
        index_nw_centered_out_ = i4_torch_arange(index_nw_centered_out_start,1+index_nw_centered_out_final);
        index_nw_centered_0in_ = i4_torch_arange(index_nw_centered_0in_start,1+index_nw_centered_0in_final);
        n_nw = numel(index_nw_centered_out_);
        tmp_i8_index_lhs_ = matlab_index_4d_0(n_UX_rank,':',n_w_max,index_nw_centered_out_,n_M,':',FTK['n_svd_l'],nl);
        tmp_i8_index_rhs_rwM_ = matlab_index_3d_0(n_k_p_r,':',n_w_max,index_nw_centered_0in_,n_M,':');
        #VUXM_centered_nwMl____.ravel()[tmp_i8_index_lhs_] = torch.reshape( mmmm( torch.reshape(V_UX_nrl___[nl,:,:],mtr((n_UX_rank,n_k_p_r))).to(dtype=torch.complex64) , torch.reshape(M_k_q_centered_rwM___.ravel()[tmp_i8_index_rhs_rwM_],mtr((n_k_p_r,n_nw*n_M))) ) / np.maximum(1,n_w_max) , mtr((n_UX_rank,n_nw,n_M)) ).to(dtype=torch.complex64).ravel();
        VUXM_centered_nwMl____ = matlab_assign_lhs_from_rhs_(VUXM_centered_nwMl____,tmp_i8_index_lhs_, torch.reshape( mmmm( torch.reshape(V_UX_nrl___[nl,:,:],mtr((n_UX_rank,n_k_p_r))).to(dtype=torch.complex64) , torch.reshape(M_k_q_centered_rwM___.ravel()[tmp_i8_index_rhs_rwM_],mtr((n_k_p_r,n_nw*n_M))) ) / np.maximum(1,n_w_max) , mtr((n_UX_rank,n_nw,n_M)) ).to(dtype=torch.complex64).ravel());
    #end;%for nl=0:FTK['n_svd_l']-1;
    tmp_i8_index_rhs_nwMl_ = matlab_index_4d_0(n_UX_rank,':',n_w_max,index_nw_zerobased_from_centered_,n_M,':',FTK['n_svd_l'],':');
    VUXM_lwnM____ = torch.permute(torch.reshape(VUXM_centered_nwMl____.ravel()[tmp_i8_index_rhs_nwMl_],mtr((n_UX_rank,numel(index_nw_zerobased_from_centered_),n_M,FTK['n_svd_l']))),mtr(mts((3,1,0,2)))).to(dtype=torch.complex64);
    #%%%%;

    if flag_verbose > 0: print(f' %% [finished {str_thisfunction}]');
    return(VUXM_lwnM____);
