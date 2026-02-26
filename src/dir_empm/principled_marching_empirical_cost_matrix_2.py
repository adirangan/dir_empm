from dir_empm.dir_matlab_macros import * ;
from dir_empm.interp_p_to_q import interp_p_to_q ;

def principled_marching_empirical_cost_matrix_2(
        n_k_p_r=None,
        k_p_r_=None,
        weight_2d_k_p_r_=None,
        n_w_=None,
        n_M=None,
        M_k_p_wkM__=None,
):
    flag_verbose = 0;
    str_thisfunction = 'principled_marching_empirical_cost_matrix_2';

    if (flag_verbose>0): disp(sprintf(' %% [entering %s]',str_thisfunction)); #end;

    n_w_max = int(torch.max(n_w_).item());
    n_w_sum = int(torch.sum(n_w_).item());
    n_w_csum_ = cumsum_0(n_w_);

    M_k_q_wkM__ = interp_p_to_q(n_k_p_r,n_w_,n_w_sum,M_k_p_wkM__);
    if numel_unique(n_w_)> 1:
        print(f' %% Warning: numel_unique(n_w_)> 1 not implemented in {str_thisfunction}');
    #end;%if numel(unique(n_w_))> 1;
    if numel_unique(n_w_)==1:
        M_k_q_rwM___ = torch.permute(torch.reshape(M_k_q_wkM__,mtr((n_w_max,n_k_p_r,n_M))),mtr(mts((1,0,2)))); 
        tmp_i8_index_lhs_ = matlab_index_3d_0(n_k_p_r,':',n_w_max,int(n_w_max/2),n_M,':');
        M_k_q_rwM___.ravel()[tmp_i8_index_lhs_] = 0.0;
    #end;%if numel(unique(n_w_))==1;

    str_einsum = msr('awM') + ',' + msr('bwM') + '->' + msr('abM') ;
    X_00_kk__ = torch.reshape(torch.sum(torch.einsum( str_einsum , torch.conj(M_k_q_rwM___) , M_k_q_rwM___ ),2-2),mtr((n_k_p_r,n_k_p_r))) * (2*pi)**2/np.maximum(1,n_M);

    X_01_kk__ = torch.zeros(mtr((n_k_p_r,n_k_p_r)));
    tmp_i8_index_rhs_ = matlab_index_3d_0(n_k_p_r,':',n_w_max,0,n_M,':');
    M_k_q_r0M_ = torch.sum(torch.reshape(M_k_q_rwM___.ravel()[tmp_i8_index_rhs_],mtr((n_k_p_r,n_M))),1-1).ravel(); assert(numel(M_k_q_r0M_)==n_k_p_r);
    X_01_kk__ = (2*pi)**2 * torch.reshape(torch.conj(M_k_q_r0M_),mtr((n_k_p_r,1))) * torch.reshape(M_k_q_r0M_,mtr((1,n_k_p_r))) / np.maximum(1,n_M**2) ;

    X_weight_r_ = torch.sqrt(weight_2d_k_p_r_).ravel();

    X_kk__ = mmmm( torch.diagflat(X_weight_r_).to(dtype=torch.float32) , mmmm( (2*torch.real(X_00_kk__) - 2*torch.real(X_01_kk__)) , torch.diagflat(X_weight_r_).to(dtype=torch.float32) ) ) ;

    if (flag_verbose>0): disp(sprintf(' %% [finished %s]',str_thisfunction)); #end;
    return(
        X_kk__,
        X_weight_r_,
    );
