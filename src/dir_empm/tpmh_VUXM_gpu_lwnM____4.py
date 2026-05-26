from dir_empm.dir_matlab_macros import * ;
import sys ;
from dir_empm.i4_torch_arange import i4_torch_arange ;

def tpmh_VUXM_gpu_lwnM____4(
        device_use=None,
        FTK=None,
        n_k_p_r=None,
        n_w_=None,
        n_M=None,
        M_k_q_gpu_wrM__=None,
        n_UX_rank=None,
        UX_gpu_rn__=None,
        X_weight_gpu_r_=None,
):
    flag_verbose=0;
    str_thisfunction = 'tpmh_VUXM_gpu_lwnM____4';

    if (flag_verbose>0): disp(sprintf(' %% [entering %s]',str_thisfunction)); #;end;
    M_k_q_gpu_wrM__ = M_k_q_gpu_wrM__.to(dtype=torch.complex64,device=device_use);
    UX_gpu_rn__ = UX_gpu_rn__.to(dtype=torch.float32,device=device_use);
    X_weight_gpu_r_ = X_weight_gpu_r_.to(dtype=torch.float32,device=device_use);

    n_w_max = int(torch.max(n_w_).item()); n_w_2 = matlab_scalar_round(n_w_max/2);
    l_max = int(torch.max(torch.abs(FTK['i4_svd_l_'])));
    n_svd_l = int(FTK['n_svd_l']);
    #%%%%%%%%;

    tmp_t = tic();
    V_gpu_lr__ = torch.reshape(FTK['r8_svd_chebval_V_r_'],mtr((n_svd_l,n_k_p_r))).to(dtype=torch.float32);
    tmp_t = toc(tmp_t); 
    if (flag_verbose>0): disp(sprintf(' %% V_gpu_lr__: time %0.6fs',tmp_t)); #end;
    tmp_t = tic();
    V_UX_gpu_lrn___ = torch.zeros(mtr((n_svd_l,n_k_p_r,n_UX_rank))).to(dtype=torch.float32,device=device_use);
    for nUX_rank in range(n_UX_rank):
        V_UX_gpu_lrn___[nUX_rank,:,:] = mmmm( V_gpu_lr__.to(dtype=torch.float32,device=device_use) , diag( UX_gpu_rn__[nUX_rank,:].ravel() * X_weight_gpu_r_.ravel() ).to(dtype=torch.float32,device=device_use) ) ;
    #end;%for nUX_rank=0:n_UX_rank-1;
    tmp_t = toc(tmp_t);
    if (flag_verbose>0): disp(sprintf(' %% V_UX_gpu_lrn___: time %0.6fs',tmp_t)); #end;
    #%%%%%%%%;

    #%%%%%%%%;
    tmp_t = tic();
    assert(np.mod(n_w_max,2)==0); #%<-- assert that n_w_max is even. ;
    index_nw_zerobased_from_centered_ = torch.concatenate( ( i4_torch_arange(n_w_2,1+n_w_max-1) , i4_torch_arange(0,1+n_w_2-1) ) , 0 ) ; #%<-- note that we place n_w_2 mode first, ;
    index_nw_centered_from_zerobased_ = torch.concatenate( ( i4_torch_arange(n_w_2,1+n_w_max-1) , i4_torch_arange(0,1+n_w_2-1) ) , 0 ) ; #%<-- note that we place first mode at n_w_2. ;
    #%%%%;
    index_nw_centered_padded_out_start_ = torch.zeros(1+2*l_max).to(dtype=torch.int32);
    index_nw_centered_padded_out_final_ = torch.zeros(1+2*l_max).to(dtype=torch.int32);
    index_nw_centered_padded_0in_start_ = torch.zeros(1+2*l_max).to(dtype=torch.int32);
    index_nw_centered_padded_0in_final_ = torch.zeros(1+2*l_max).to(dtype=torch.int32);
    for l_shift in range(-l_max,+l_max+1):
        index_nw_centered_padded_out_start_[l_max+l_shift] = 1+l_max;
        index_nw_centered_padded_out_final_[l_max+l_shift] = n_w_max-1+l_max;
        index_nw_centered_padded_0in_start_[l_max+l_shift] = 1+l_max+l_shift;
        index_nw_centered_padded_0in_final_[l_max+l_shift] = n_w_max-1+l_max+l_shift;
    #end;%for l_shift=-l_max:+l_max;
    tmp_t = toc(tmp_t);
    if (flag_verbose>0): disp(sprintf(' %% index_nw_centered_padded_xxx_xxxxx_: time %0.6fs',tmp_t)); #end;
    #%%%%%%%%;

    tmp_t = tic();
    V_UX_gpu_nrl___ = torch.permute(V_UX_gpu_lrn___,mtr(mts((2,1,0))));
    tmp_t = toc(tmp_t);
    if (flag_verbose): disp(sprintf(' %% V_UX_gpu_rnl___: time %0.6fs',tmp_t)); #end;
    tmp_t = tic();
    M_k_q_centered_padded_gpu_rwM___ = torch.concatenate( ( torch.zeros(mtr((l_max,n_k_p_r,n_M))).to(dtype=torch.complex64,device=device_use) , torch.roll(torch.reshape(M_k_q_gpu_wrM__,mtr((n_w_max,n_k_p_r,n_M))),-n_w_2,dims=2-0) , torch.zeros(mtr((l_max,n_k_p_r,n_M))).to(dtype=torch.complex64,device=device_use) ) , 2-0 ).to(dtype=torch.complex64,device=device_use);
    tmp_i8_index_lhs_ = matlab_index_3d_gpu_0(device_use,l_max+n_w_max+l_max,l_max,n_k_p_r,':',n_M,':');
    #M_k_q_centered_padded_gpu_rwM___.ravel()[tmp_i8_index_lhs_] = torch.zeros(1*n_k_p_r*n_M).to(dtype=torch.complex64,device=device_use);
    M_k_q_centered_padded_gpu_rwM___ = matlab_assign_lhs_from_rhs_(M_k_q_centered_padded_gpu_rwM___,tmp_i8_index_lhs_, torch.zeros(1*n_k_p_r*n_M).to(dtype=torch.complex64,device=device_use));
    M_k_q_centered_padded_gpu_rwM___ = torch.permute(M_k_q_centered_padded_gpu_rwM___,mtr(mts((1,0,2))));
    tmp_t = toc(tmp_t);
    if (flag_verbose): disp(sprintf(' %% M_k_q_centered_padded_gpu_rwM___: time %0.6fs',tmp_t)); #end;
    tmp_t = tic();
    V_UX_M_centered_gpu_nwMl____ = torch.zeros(mtr((n_UX_rank,n_w_max,n_M,n_svd_l))).to(dtype=torch.complex64,device=device_use);
    for nl in range(n_svd_l):
        l_shift = int(FTK['i4_svd_l_'][nl].item());
        index_nw_centered_padded_out_start = int(index_nw_centered_padded_out_start_[l_max+l_shift].item());
        index_nw_centered_padded_out_final = int(index_nw_centered_padded_out_final_[l_max+l_shift].item());
        index_nw_centered_padded_0in_start = int(index_nw_centered_padded_0in_start_[l_max+l_shift].item());
        index_nw_centered_padded_0in_final = int(index_nw_centered_padded_0in_final_[l_max+l_shift].item());
        index_nw_centered_padded_out_ = torch.arange(index_nw_centered_padded_out_start,index_nw_centered_padded_out_final+1).to(dtype=torch.int32,device=device_use);
        index_nw_centered_padded_0in_ = torch.arange(index_nw_centered_padded_0in_start,index_nw_centered_padded_0in_final+1).to(dtype=torch.int32,device=device_use);
        tmp_i8_index_lhs_nwMl_ = matlab_index_4d_gpu_0(device_use,n_UX_rank,':',n_w_max,torch.arange(1,n_w_max),n_M,':',n_svd_l,nl);
        tmp_i8_index_rhs_nrl_ = matlab_index_3d_gpu_0(device_use,n_UX_rank,':',n_k_p_r,':',n_svd_l,nl);
        tmp_i8_index_rhs_rwM_ = matlab_index_3d_gpu_0(device_use,n_k_p_r,':',l_max+n_w_max+l_max,index_nw_centered_padded_0in_,n_M,':');
        str_einsum = msr('nr') + ',' + msr('rwM') + '->' + msr('nwM') ;
        #V_UX_M_centered_gpu_nwMl____.ravel()[tmp_i8_index_lhs_nwMl_] = torch.einsum( str_einsum , torch.reshape(V_UX_gpu_nrl___.to(dtype=torch.complex64,device=device_use).ravel()[tmp_i8_index_rhs_nrl_],mtr((n_UX_rank,n_k_p_r)))/np.maximum(1,n_w_max) , torch.reshape(M_k_q_centered_padded_gpu_rwM___.to(dtype=torch.complex64,device=device_use).ravel()[tmp_i8_index_rhs_rwM_],mtr((n_k_p_r,n_w_max-1,n_M))) ).ravel() ;
        V_UX_M_centered_gpu_nwMl____ = matlab_assign_lhs_from_rhs_(V_UX_M_centered_gpu_nwMl____,tmp_i8_index_lhs_nwMl_, torch.einsum( str_einsum , torch.reshape(V_UX_gpu_nrl___.to(dtype=torch.complex64,device=device_use).ravel()[tmp_i8_index_rhs_nrl_],mtr((n_UX_rank,n_k_p_r)))/np.maximum(1,n_w_max) , torch.reshape(M_k_q_centered_padded_gpu_rwM___.to(dtype=torch.complex64,device=device_use).ravel()[tmp_i8_index_rhs_rwM_],mtr((n_k_p_r,n_w_max-1,n_M))) ).ravel() );
    #end;%for nl=0:n_svd_l-1;
    tmp_t = toc(tmp_t);
    if (flag_verbose): disp(sprintf(' %% V_UX_M_centered_padded_lwMn____: time %0.6fs',tmp_t)); #end;
    tmp_t = tic();
    tmp_i8_index_rhs_ = matlab_index_4d_gpu_0(device_use,n_UX_rank,':',n_w_max,index_nw_zerobased_from_centered_,n_M,':',n_svd_l,':');
    V_UX_M_gpu_lwnM____ = torch.permute(torch.reshape(V_UX_M_centered_gpu_nwMl____.ravel()[tmp_i8_index_rhs_],mtr((n_UX_rank,n_w_max,n_M,n_svd_l))),mtr(mts((3,1,0,2))));
    tmp_t = toc(tmp_t);
    if (flag_verbose): disp(sprintf(' %% V_UX_M_gpu_lwnM____: time %0.6fs',tmp_t)); #end;

    if (flag_verbose>0): disp(sprintf(' %% [finished %s]',str_thisfunction)); #;end;
    return(V_UX_M_gpu_lwnM____);
