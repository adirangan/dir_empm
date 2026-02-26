from dir_empm.dir_matlab_macros import * ;
from dir_empm.interp_p_to_q import interp_p_to_q ;
from dir_empm.tpmh_VUXM_lwnM____3 import tpmh_VUXM_lwnM____3 ;
from dir_empm.tfpmh_UX_T_M_l2_dM__1 import tfpmh_UX_T_M_l2_dM__1 ;

def tfpmh_Z_wSM___12(
        parameter=None,
        n_k_p_r=None,
        k_p_r_=None,
        k_p_r_max=None,
        n_w_=None,
        weight_2d_k_p_r_=None,
        weight_2d_k_p_wk_=None,
        n_M=None,
        M_k_p_wkM__=None,
        CTF_k_p_wk_=None,
        n_S=None,
        S_k_p_wkS__=None,
        pm_n_UX_rank=None,
        pm_UX_kn__=None,
        pm_X_weight_r_=None,
        FTK=None,
):
    str_thisfunction = 'tfpmh_Z_wSM___12' ;

    if not isinstance(parameter, dict): parameter = {'type': 'parameter'};
    if 'flag_verbose' not in parameter: parameter['flag_verbose'] = 0 ;
    flag_verbose = parameter['flag_verbose'];
    if 'tolerance_master' not in parameter: parameter['tolerance_master'] = 1e-2 ;
    tolerance_master = parameter['tolerance_master'];
    if 'n_M_per_Mbatch' not in parameter: parameter['n_M_per_Mbatch'] = 24 ;
    n_M_per_Mbatch = parameter['n_M_per_Mbatch'];
    if 'n_S_per_Sbatch' not in parameter: parameter['n_S_per_Sbatch'] = 24 ;
    n_S_per_Sbatch = parameter['n_S_per_Sbatch'];
    if 'flag_optimize_over_gamma_z' not in parameter: parameter['flag_optimize_over_gamma_z'] = 0 ;
    flag_optimize_over_gamma_z = parameter['flag_optimize_over_gamma_z'];
    if 'flag_dwSM' not in parameter: parameter['flag_dwSM'] = 0 ;
    flag_dwSM = parameter['flag_dwSM'];
    if 'memory_limit_GB' not in parameter: parameter['memory_limit_GB'] = 4.0 ;
    memory_limit_GB = parameter['memory_limit_GB'];
    if 'device_use' not in parameter: parameter['device_use'] = 'cpu' ; #<-- need to control batch sizes below in order to push onto cuda. ;
    device_use = parameter['device_use'];

    #%%%%%%%%;
    if (flag_verbose): print(f' %% [entering {str_thisfunction}]');
    #%%%%%%%%;

    n_w_ = n_w_.ravel();
    n_w_max = int(torch.max(n_w_).item());
    n_w_sum = int(torch.sum(n_w_).item());
    n_w_csum_ = cumsum_0(n_w_);
    if (n_w_sum!=n_w_max*n_k_p_r): print(f' %% Warning, n_w_sum {n_w_sum} ~= n_w_max*n_k_p_r {n_w_max}*{n_k_p_r} in {str_thisfunction}');
    n_delta_v = FTK['n_delta_v'];
    n_svd_l = FTK['n_svd_l'];

    #%%%%%%%%;
    #% allocate memory for output. ;
    #%%%%%%%%;
    if flag_optimize_over_gamma_z==0:
        Z_wSM___ = torch.zeros(mtr((n_w_max,n_S,n_M))).to(dtype=torch.float32);
        UX_CTF_R_S_l2_wS__ = torch.zeros(mtr((n_w_max,n_S))).to(dtype=torch.float32);
        CTF_R_S_l2_wS__ = torch.zeros(mtr((n_w_max,n_S))).to(dtype=torch.float32);
        UX_T_M_l2_dM__ = torch.zeros(mtr((n_delta_v,n_M))).to(dtype=torch.float32);
        UX_M_l2_M_ = torch.zeros(n_M).to(dtype=torch.float32);
        X_wSM___ = torch.zeros(mtr((n_w_max,n_S,n_M))).to(dtype=torch.float32);
        delta_x_wSM___ = torch.zeros(mtr((n_w_max,n_S,n_M))).to(dtype=torch.float32);
        delta_y_wSM___ = torch.zeros(mtr((n_w_max,n_S,n_M))).to(dtype=torch.float32);
        gamma_z_wSM___ = torch.zeros(mtr((n_w_max,n_S,n_M))).to(dtype=torch.float32);
        index_sub_wSM___ = torch.zeros(mtr((n_w_max,n_S,n_M))).to(dtype=torch.int32);
    #end;%if flag_optimize_over_gamma_z==0;
    if flag_optimize_over_gamma_z==1:
        Z_SM__ = torch.zeros(mtr((n_S,n_M))).to(dtype=torch.float32);
        UX_CTF_R_S_l2_wS__ = torch.zeros(n_S).to(dtype=torch.float32);
        CTF_R_S_l2_wS__ = torch.zeros(n_S).to(dtype=torch.float32);
        UX_T_M_l2_dM__ = torch.zeros(mtr((n_delta_v,n_M))).to(dtype=torch.float32);
        UX_M_l2_M_ = torch.zeros(n_M).to(dtype=torch.float32);
        X_SM__ = torch.zeros(mtr((n_S,n_M))).to(dtype=torch.float32);
        delta_x_SM__ = torch.zeros(mtr((n_S,n_M))).to(dtype=torch.float32);
        delta_y_SM__ = torch.zeros(mtr((n_S,n_M))).to(dtype=torch.float32);
        gamma_z_SM__ = torch.zeros(mtr((n_S,n_M))).to(dtype=torch.float32);
        index_sub_SM__ = torch.zeros(mtr((n_S,n_M))).to(dtype=torch.int32);
    #end;%if flag_optimize_over_gamma_z==1;

    Z_dwSM____=None;
    X_dwSM____=None;
    if flag_dwSM:
        n_dwSM = n_delta_v*n_w_max*n_S*n_M;
        n_dwSM_GB = n_dwSM*n_byte_per_float32/1e9;
        if (flag_verbose): tmp_str = 'X_dwSM____'; print(f' %% memory: {tmp_str} --> {n_dwSM_GB:.6f} GB');
        if (n_dwSM_GB> memory_limit_GB):
            print(f' %% Warning, n_dwSM_GB {n_dwSM_GB:.2f} in {str_thisfunction}');
            flag_dwSM = 0;
        #end;%if (n_dwSM_GB> memory_limit_GB);
        if (n_dwSM_GB<=memory_limit_GB):
            Z_dwSM____ = torch.zeros(mtr((n_delta_v,n_w_max,n_S,n_M))).to(dtype=torch.float32);
            X_dwSM____ = torch.zeros(mtr((n_delta_v,n_w_max,n_S,n_M))).to(dtype=torch.float32);
            flag_dwSM = 1;
        #end;%if (n_dwSM_GB<=memory_limit_GB);
    #end;%if flag_dwSM;

    if (flag_verbose>0): 
        n_dwSM_GB = n_delta_v*n_w_max*n_S*n_M*n_byte_per_float32/1e9;
        n_wSM_GB = n_w_max*n_S*n_M*n_byte_per_float32/1e9;
        n_wS_GB = n_w_max*n_S*n_byte_per_float32/1e9;
        n_dM_GB = n_delta_v*n_M*n_byte_per_float32/1e9;
        n_M_GB = n_M*n_byte_per_float32/1e9;
        tmp_str = 'UX_CTF_R_S_l2_wS__'; print(f' %% memory: {tmp_str} --> {n_wS_GB} GB');
        tmp_str = 'CTF_R_S_l2_wS__'; print(f' %% memory: {tmp_str} --> {n_wS_GB} GB');
        tmp_str = 'UX_T_M_l2_dM__'; print(f' %% memory: {tmp_str} --> {n_dM_GB} GB');
        tmp_str = 'UX_M_l2_M_'; print(f' %% memory: {tmp_str} --> {n_M_GB} GB');
        if flag_optimize_over_gamma_z==0:
            tmp_str = 'Z_wSM___'; print(f' %% memory: {tmp_str} --> {n_wSM_GB} GB');
            tmp_str = 'X_wSM___'; print(f' %% memory: {tmp_str} --> {n_wSM_GB} GB');
            tmp_str = 'delta_x_wSM___'; print(f' %% memory: {tmp_str} --> {n_wSM_GB} GB');
            tmp_str = 'delta_y_wSM___'; print(f' %% memory: {tmp_str} --> {n_wSM_GB} GB');
            tmp_str = 'gamma_z_wSM___'; print(f' %% memory: {tmp_str} --> {n_wSM_GB} GB');
        #end;%flag_optimize_over_gamma_z==0;
        if flag_optimize_over_gamma_z==1:
            tmp_str = 'Z_SM__'; print(f' %% memory: {tmp_str} --> {n_wSM_GB} GB');
            tmp_str = 'X_SM__'; print(f' %% memory: {tmp_str} --> {n_wSM_GB} GB');
            tmp_str = 'delta_x_SM__'; print(f' %% memory: {tmp_str} --> {n_wSM_GB} GB');
            tmp_str = 'delta_y_SM__'; print(f' %% memory: {tmp_str} --> {n_wSM_GB} GB');
            tmp_str = 'gamma_z_SM__'; print(f' %% memory: {tmp_str} --> {n_wSM_GB} GB');
        #end;%if flag_optimize_over_gamma_z==1;
        if (flag_dwSM): tmp_str = 'Z_dwSM____'; print(f' %% memory: {tmp_str} --> {n_dwSM_GB} GB');
        if (flag_dwSM): tmp_str = 'X_dwSM____'; print(f' %% memory: {tmp_str} --> {n_dwSM_GB} GB');
    #end;%if (flag_verbose>0); 

    #%%%%%%%%;
    #% Now construct the template-norms: ;
    #% <(R(+gamma_z)*CTF_k_p_wk_).*S_k_p_wk_,(R(+gamma_z)*CTF_k_p_wk_).*S_k_p_wk_> ;
    #% Note that this does not involve collapsing onto principal-modes. ;
    #%%%%%%%%;
    CTF_R_S_l2_wS__ = torch.zeros(mtr((n_w_max,n_S))).to(dtype=torch.float32);
    SS_k_p_wkS__ = torch.conj(S_k_p_wkS__)*S_k_p_wkS__;
    SS_k_q_wkS__ = torch.reshape(interp_p_to_q(n_k_p_r,n_w_,n_w_sum,SS_k_p_wkS__),mtr((n_w_sum,n_S)));
    CC_k_p_wk_ = torch.conj(CTF_k_p_wk_)*CTF_k_p_wk_;
    CC_k_q_wk_ = torch.reshape(interp_p_to_q(n_k_p_r,n_w_,n_w_sum,CC_k_p_wk_),mtr((n_w_sum,1)));
    CTF_R_S_l2_wS__ = torch.fft.ifft(torch.reshape(torch.sum(torch.reshape(CC_k_q_wk_*torch.conj(SS_k_q_wkS__),mtr((n_w_max,n_k_p_r,n_S)))*torch.reshape(weight_2d_k_p_r_,mtr((1,n_k_p_r,1))),2-1),mtr((n_w_max,n_S))),dim=1-0);
    CTF_R_S_l2_wS__ = torch.real(CTF_R_S_l2_wS__);
    #%%%%%%%%;
    #% Now re-construct the template-norms, this time limited to radial principal-modes: ;
    #% <((R(+gamma_z)*CTF_k_p_wk_).*S_k_p_wk_) * pm_wUX_kn__),((R(+gamma_z)*CTF_k_p_wk_).*S_k_p_wk_) * pm_wUX_kn__)> ;
    #% Note that this does yes involve collapsing onto principal-modes. ;
    #%%%%%%%%;
    pm_n_k_p_r = pm_n_UX_rank; pm_n_w_max = n_w_max;
    pm_n_w_ = pm_n_w_max*torch.ones(pm_n_k_p_r).to(dtype=torch.int32);
    pm_n_w_sum = int(pm_n_k_p_r*pm_n_w_max);
    UX_CTF_R_S_l2_wS__ = torch.zeros(mtr((n_w_max,n_S))).to(dtype=torch.float32);
    pm_wUX_kn__ = torch.reshape(pm_X_weight_r_,mtr((n_k_p_r,1)))*torch.reshape(pm_UX_kn__,mtr((n_k_p_r,pm_n_k_p_r)));
    str_einsum = msr('kn') + ',' + msr('wkS') + '->' + msr('wnS') ;
    UX_SS_k_q_wnS__ = torch.reshape(torch.einsum(str_einsum,pm_wUX_kn__.to(dtype=torch.complex64),torch.reshape(SS_k_q_wkS__,mtr((n_w_max,n_k_p_r,n_S))).to(dtype=torch.complex64)),mtr((pm_n_w_sum,n_S))).to(dtype=torch.complex64);
    UX_CC_k_q_wn_ = torch.reshape(mmmm( torch.reshape(CC_k_q_wk_.to(dtype=torch.complex64),mtr((n_w_max,n_k_p_r))) , pm_wUX_kn__.to(dtype=torch.complex64) ),mtr((pm_n_w_sum,1))).to(dtype=torch.complex64);
    UX_CTF_R_S_l2_wS__ = torch.real(torch.fft.ifft(torch.reshape(torch.sum(torch.reshape(UX_CC_k_q_wn_ * torch.conj(UX_SS_k_q_wnS__),mtr((pm_n_w_max,pm_n_k_p_r,n_S))),2-1),mtr((pm_n_w_max,n_S))),dim=1-0)).to(dtype=torch.float32);
    #%%%%%%%%

    flag_continue=1;
    n_M_per_Mbatch = int(np.maximum(1,n_M_per_Mbatch));
    n_S_per_Sbatch = int(np.maximum(1,n_S_per_Sbatch));
    n_dwSM = n_delta_v*n_w_max*n_S_per_Sbatch*n_M_per_Mbatch;
    n_dwSM_GB = n_dwSM*n_byte_per_float32/1e9;
    if (n_dwSM_GB> memory_limit_GB): print(f' %% Warning, n_dwSM_GB {n_dwSM_GB:.2f} > {memory_limit_GB:.2f} in {str_thisfunction}');

    n_Mbatch = int(np.ceil(n_M/max(1,n_M_per_Mbatch)));
    if (flag_verbose>1): print(f' %% n_Mbatch %d',n_Mbatch);
    n_Sbatch = int(np.ceil(n_S/max(1,n_S_per_Sbatch)));
    if (flag_verbose>1): print(f' %% n_Sbatch %d',n_Sbatch);
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;
    for nMbatch in range(n_Mbatch):
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;
        index_nM_in_Mbatch_ = int(nMbatch*n_M_per_Mbatch) + torch.arange(0,n_M_per_Mbatch).to(dtype=torch.int32);
        index_nM_in_Mbatch_ = index_nM_in_Mbatch_[efind(index_nM_in_Mbatch_<n_M)]; n_M_sub = numel(index_nM_in_Mbatch_);
        if (flag_verbose>1): print(f' %% nMbatch {nMbatch}/{n_Mbatch} index_nM_in_Mbatch_ {index_nM_in_Mbatch_[0].item()}-->{index_nM_in_Mbatch_[n_M_sub-1].item()}');
        if (flag_verbose>0 and np.mod(nMbatch,1)==0): print(f' %% nMbatch {nMbatch}/{n_Mbatch} index_nM_in_Mbatch_ {index_nM_in_Mbatch_[0].item()}-->{index_nM_in_Mbatch_[n_M_sub-1].item()}');
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;
        if (n_M_sub>0):
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;
            tmp_t = tic();
            tmp_i8_index_rhs_ = matlab_index_2d_0(n_w_sum,':',n_M,index_nM_in_Mbatch_);
            M_sub_k_p_wkM__ = torch.reshape(M_k_p_wkM__.ravel()[tmp_i8_index_rhs_],mtr((n_w_sum,n_M_sub)));
            M_sub_k_q_wkM__ = interp_p_to_q(n_k_p_r,n_w_,n_w_sum,M_sub_k_p_wkM__);
            tmp_t = toc(tmp_t); 
            if (flag_verbose>1): print(f' %% M_sub_k_q_wkM__: %0.2fs',tmp_t);
            UX_M_sub_l2_M_ = torch.zeros(n_M_sub).to(dtype=torch.float32);
            UX_T_M_sub_l2_dM__ = torch.zeros(mtr((n_delta_v,n_M_sub))).to(dtype=torch.float32);
            tmp_t = tic();
            CTF_M_sub_k_p_wkM__ = CTF_k_p_wk_*M_sub_k_p_wkM__;
            CTF_M_sub_k_q_wkM__ = interp_p_to_q(n_k_p_r,n_w_,n_w_sum,CTF_M_sub_k_p_wkM__);
            tmp_t = toc(tmp_t); 
            if (flag_verbose>1): print(f' %% CTF_M_sub_k_q_wkM__: %0.2fs',tmp_t);
            #%%%%;
            #% Prepare quasi-images. ;
            #%%%%;
            tmp_t = tic();
            svd_V_UX_CTF_M_sub_lwnM____ = tpmh_VUXM_lwnM____3(FTK,n_k_p_r,n_w_,n_M_sub,CTF_M_sub_k_q_wkM__,pm_n_UX_rank,pm_UX_kn__,pm_X_weight_r_);
            tmp_t = toc(tmp_t); 
            if (flag_verbose>1): print(f' %% svd_V_UX_CTF_M_sub_lwnM____: %0.2fs',tmp_t);
            tmp_t = tic();
            svd_V_UX_M_sub_lwnM____ = tpmh_VUXM_lwnM____3(FTK,n_k_p_r,n_w_,n_M_sub,M_sub_k_q_wkM__,pm_n_UX_rank,pm_UX_kn__,pm_X_weight_r_);
            tmp_t = toc(tmp_t); 
            if (flag_verbose>1): print(f' %% svd_V_UX_M_sub_lwnM____: %0.2fs',tmp_t);
            tmp_t = tic();
            svd_V_UX_CTF_M_sub_nMwl____ = torch.permute(svd_V_UX_CTF_M_sub_lwnM____.to(dtype=torch.complex64),mtr(mts((2,3,1,0)))).to(dtype=torch.complex64);
            tmp_t = toc(tmp_t); 
            if (flag_verbose>1): print(f' %% svd_V_UX_CTF_M_sub_nMwl____: %0.2fs',tmp_t);
            #%%%%;
            #% Now calculate norms of the translated images. ;
            #%%%%;
            tmp_t = tic();
            UX_T_M_sub_l2_dM__ = tfpmh_UX_T_M_l2_dM__1(FTK,n_w_,n_M_sub,pm_n_UX_rank,svd_V_UX_M_sub_lwnM____);
            tmp_t = toc(tmp_t); 
            if (flag_verbose>1): print(f' %% tfpmh_UX_T_M_sub_l2_dm__1: %0.2fs',tmp_t);
            tmp_index_d0 = intersect_0(efind(torch.abs(FTK['r8_delta_x_'])< 1e-6),efind(torch.abs(FTK['r8_delta_y_'])< 1e-6))[0];
            assert(numel(tmp_index_d0)>=1); #%<-- should be a single index corresponding to zero-displacement. ;
            if numel(tmp_index_d0)> 1: tmp_index_d0 = tmp_index_d0[0]; #end;
            tmp_i8_index_rhs_ = matlab_index_2d_0(FTK['n_delta_v'],tmp_index_d0,n_M_sub,':');
            UX_M_sub_l2_M_ = UX_T_M_sub_l2_dM__.ravel()[tmp_i8_index_rhs_].ravel(); assert(numel(UX_M_sub_l2_M_)==n_M_sub);
            #%%%%;
            #% Store results. ;
            #%%%%;
            UX_M_l2_M_[index_nM_in_Mbatch_] = UX_M_sub_l2_M_;
            tmp_i8_index_lhs_ = matlab_index_2d_0(n_delta_v,':',n_M,index_nM_in_Mbatch_);
            UX_T_M_l2_dM__.ravel()[tmp_i8_index_lhs_] = UX_T_M_sub_l2_dM__.ravel();
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;
            for nSbatch in range(n_Sbatch):
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;
                index_nS_in_Sbatch_ = int(nSbatch*n_S_per_Sbatch) + torch.arange(n_S_per_Sbatch).to(dtype=torch.int32);
                index_nS_in_Sbatch_ = index_nS_in_Sbatch_[efind(index_nS_in_Sbatch_<n_S)]; n_S_sub = numel(index_nS_in_Sbatch_);
                if (flag_verbose>2): print(f' %% nSbatch {nSbatch}/{n_Sbatch} index_nS_in_Sbatch_ {index_nS_in_Sbatch_[0].item()}-->{index_nS_in_Sbatch_[n_S_sub-1].item()}');
                if (flag_verbose>1 and np.mod(nSbatch,32)==0): print(f' %% nSbatch {nSbatch}/{n_Sbatch} index_nS_in_Sbatch_ {index_nS_in_Sbatch_[0].item()}-->{index_nS_in_Sbatch_[n_S_sub-1].item()}');
                #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;
                if (n_S_sub>0):
                #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;
                    tmp_i8_index_rhs_ = matlab_index_2d_0(n_w_sum,':',n_S,index_nS_in_Sbatch_);
                    S_sub_k_p_wkS__ = torch.reshape(S_k_p_wkS__.ravel()[tmp_i8_index_rhs_],mtr((n_w_sum,n_S_sub)));
                    S_sub_k_q_wkS__ = interp_p_to_q(n_k_p_r,n_w_,n_w_sum,S_sub_k_p_wkS__);
                    S_sub_k_q_wSk___ = torch.permute(torch.reshape(S_sub_k_q_wkS__,mtr((n_w_max,n_k_p_r,n_S_sub))),mtr(mts((0,2,1))));
                    #%%%%;
                    #% Prepare UX_S_k_q_wnS__. ;
                    #%%%%;
                    tmp_t = tic();
                    UX_S_sub_k_q_wnS__ = torch.reshape(torch.permute(torch.reshape(mmmm( torch.reshape(S_sub_k_q_wSk___.to(dtype=torch.complex64),mtr((n_w_max*n_S_sub,n_k_p_r)))*torch.reshape(pm_X_weight_r_.to(dtype=torch.complex64),mtr((1,n_k_p_r))) , pm_UX_kn__.to(dtype=torch.complex64) ),mtr((n_w_max,n_S_sub,pm_n_UX_rank))),mtr(mts((0,2,1)))),mtr((n_w_max*pm_n_UX_rank,n_S_sub))).to(dtype=torch.complex64);
                    tmp_t = toc(tmp_t); 
                    if (flag_verbose>1): print(f' %% UX_S_sub_k_q_wnS__: %0.2fs',tmp_t);
                    tmp_t = tic();
                    UX_S_sub_k_q_nSw___ = torch.permute(torch.reshape(UX_S_sub_k_q_wnS__.to(dtype=torch.complex64),mtr((n_w_max,pm_n_UX_rank,n_S_sub))),mtr(mts((1,2,0)))).to(dtype=torch.complex64);
                    tmp_t = toc(tmp_t); 
                    if (flag_verbose>1): print(f' %% UX_S_sub_k_q_nSw__: %0.2fs',tmp_t);
                    #%%%%;
                    #% Calculate innerproduct Z_sub_dwSM____. ;
                    #%%%%;
                    tmp_t = tic();
                    str_einsum = msr('nSw') + ',' + msr('nMwl') + '->' + msr('SMwl') ;
                    svd_S_sub_V_UX_CTF_M_sub_SMwl____ = torch.einsum(str_einsum,torch.conj(UX_S_sub_k_q_nSw___).to(dtype=torch.complex64),svd_V_UX_CTF_M_sub_nMwl____.to(dtype=torch.complex64)).to(dtype=torch.complex64);
                    tmp_t = toc(tmp_t); 
                    if (flag_verbose>1): print(f' %% svd_S_sub_V_UX_CTF_M_sub_SMwl____: %0.6fs',tmp_t);
                    tmp_t = tic();
                    svd_S_sub_V_UX_CTF_M_sub_lwSM____ = torch.fft.ifft(torch.permute(svd_S_sub_V_UX_CTF_M_sub_SMwl____.to(dtype=torch.complex64),mtr(mts((3,2,0,1)))),dim=3-1).to(dtype=torch.complex64)*n_w_max;
                    tmp_t = toc(tmp_t); 
                    if (flag_verbose>1): print(f' %% svd_S_sub_V_UX_CTF_M_sub_lwSM____: %0.6fs',tmp_t);
                    tmp_t = tic();
                    svd_UES_S_sub_V_UX_CTF_M_sub_dwSM____ = torch.reshape( mmmm( torch.reshape(FTK['c16_svd_U_d_expiw_s__'].to(dtype=torch.complex64),mtr((FTK['n_delta_v'],FTK['n_svd_l']))) , torch.reshape(svd_S_sub_V_UX_CTF_M_sub_lwSM____.to(dtype=torch.complex64),mtr((FTK['n_svd_l'],n_w_max*n_S_sub*n_M_sub))) ),mtr((FTK['n_delta_v'],n_w_max,n_S_sub,n_M_sub))).to(dtype=torch.complex64);
                    tmp_t = toc(tmp_t); 
                    if (flag_verbose>1): print(f' %% svd_UES_S_sub_V_UX_CTF_M_sub_dwSM____: %0.6fs',tmp_t);
                    Z_sub_dwSM____  = torch.real(svd_UES_S_sub_V_UX_CTF_M_sub_dwSM____);
                    #%%%%;
                    #% Calculate correlation. ;
                    #%%%%;
                    tmp_t = tic();
                    tmp_i8_index_rhs_ = matlab_index_2d_0(n_w_max,':',n_S,index_nS_in_Sbatch_);
                    UX_CTF_R_S_sub_l2_wS__ = torch.reshape(UX_CTF_R_S_l2_wS__.ravel()[tmp_i8_index_rhs_],mtr((n_w_max,n_S_sub)));
                    X_sub_dwSM____ = Z_sub_dwSM____ / torch.maximum(torch.tensor(1e-6).to(dtype=torch.float32),torch.reshape(torch.sqrt(UX_CTF_R_S_sub_l2_wS__),mtr((1,n_w_max,n_S_sub,1)))) / torch.maximum(torch.tensor(1e-6).to(dtype=torch.float32),torch.reshape(torch.sqrt(UX_T_M_sub_l2_dM__),mtr((n_delta_v,1,1,n_M_sub)))) ;
                    tmp_t = toc(tmp_t); 
                    if (flag_verbose>1): print(f' %% X_sub_dwSM____: %0.6fs',tmp_t);
                    #%%%%;
                    #% Store results. ;
                    #%%%%;
                    if flag_dwSM:
                        tmp_i8_index_lhs_ = matlab_index_4d_0(n_delta_v,':',n_w_max,':',n_S,index_nS_in_Sbatch_,n_M,index_nM_in_Mbatch_);
                        Z_dwSM____.ravel()[tmp_i8_index_lhs_] = Z_sub_dwSM____.ravel();
                        X_dwSM____.ravel()[tmp_i8_index_lhs_] = X_sub_dwSM____.ravel();
                    #%end;%if flag_dwSM;
                    #%%%%%%%%;
                    if flag_optimize_over_gamma_z==0:
                        tmp_t = tic();
                        n_wSM = n_w_max*n_S_sub*n_M_sub;
                        tmp_X_wSM_,tmp_index_delta_wSM_ = torch.max(torch.reshape(X_sub_dwSM____,mtr((n_delta_v,n_wSM))),dim=1-0); #%<-- maximize correlation. ;
                        assert(torch.min(tmp_index_delta_wSM_.ravel()).item()>=0);
                        assert(torch.max(tmp_index_delta_wSM_.ravel()).item()<=n_delta_v-1);
                        tmp_X_wSM___ = torch.reshape(tmp_X_wSM_,mtr((n_w_max,n_S_sub,n_M_sub)));
                        tmp_i8_index_all_wSM_ = tmp_index_delta_wSM_.to(dtype=torch.int64).ravel() + torch.arange(n_wSM).ravel().to(dtype=torch.int64)*int(n_delta_v);
                        assert(torch.min(tmp_i8_index_all_wSM_.ravel()).item()>=0);
                        assert(torch.max(tmp_i8_index_all_wSM_.ravel()).item()<=n_delta_v*n_wSM-1);
                        tmp_Z_wSM___ = torch.reshape(Z_sub_dwSM____.ravel()[tmp_i8_index_all_wSM_],mtr((n_w_max,n_S_sub,n_M_sub)));
                        tmp_index_delta_wSM___ = torch.reshape(tmp_index_delta_wSM_,mtr((n_w_max,n_S_sub,n_M_sub)));
                        tmp_delta_x_wSM___ = FTK['r8_delta_x_'].to(dtype=torch.float32).ravel()[tmp_index_delta_wSM___];
                        tmp_delta_y_wSM___ = FTK['r8_delta_y_'].to(dtype=torch.float32).ravel()[tmp_index_delta_wSM___];
                        tmp_gamma_z_wSM___ = 2*pi*torch.arange(n_w_max).to(dtype=torch.float32)/np.maximum(1,n_w_max);
                        tmp_i8_index_lhs_ = matlab_index_3d_0(n_w_max,':',n_S,index_nS_in_Sbatch_,n_M,index_nM_in_Mbatch_);
                        Z_wSM___.ravel()[tmp_i8_index_lhs_] = tmp_Z_wSM___.ravel();
                        X_wSM___.ravel()[tmp_i8_index_lhs_] = tmp_X_wSM___.ravel();
                        delta_x_wSM___.ravel()[tmp_i8_index_lhs_] = torch.reshape(tmp_delta_x_wSM___,mtr((n_w_max,n_S_sub,n_M_sub))).ravel();
                        delta_y_wSM___.ravel()[tmp_i8_index_lhs_] = torch.reshape(tmp_delta_y_wSM___,mtr((n_w_max,n_S_sub,n_M_sub))).ravel();
                        gamma_z_wSM___.ravel()[tmp_i8_index_lhs_] = (torch.reshape(tmp_gamma_z_wSM___,mtr((n_w_max,1,1)))*torch.ones(mtr((1,n_S_sub,n_M_sub)))).to(dtype=torch.float32).ravel();
                        index_sub_wSM___.ravel()[tmp_i8_index_lhs_] = torch.reshape(tmp_index_delta_wSM_.to(dtype=torch.int32),mtr((n_w_max,n_S_sub,n_M_sub))).ravel();
                        tmp_t = toc(tmp_t); 
                        if (flag_verbose>1): print(f' %% X_wSM___: %0.6f',tmp_t);
                    #end;%if flag_optimize_over_gamma_z==0;
                    #%%%%%%%%;
                    if flag_optimize_over_gamma_z==1:
                        tmp_t = tic();
                        n_dw = n_delta_v*n_w_max; n_SM = n_S_sub*n_M_sub;
                        tmp_X_SM_,tmp_index_dw_SM_ = torch.max(torch.reshape(X_sub_dwSM____,mtr((n_dw,n_SM))),dim=1-0); #%<-- maximize correlation. ;
                        assert(torch.min(tmp_index_dw_SM_.ravel()).item()>=0);
                        assert(torch.max(tmp_index_dw_SM_.ravel()).item()<=n_dw-1);
                        tmp_X_SM__ = torch.reshape(tmp_X_SM_,mtr((n_S_sub,n_M_sub)));
                        tmp_i8_index_all_SM_ = tmp_index_dw_SM_.to(dtype=torch.int64).ravel() + torch.arange(n_SM).ravel().to(dtype=torch.int64)*int(n_dw);
                        assert(torch.min(tmp_i8_index_all_SM_.ravel()).item()>=0);
                        assert(torch.max(tmp_i8_index_all_SM_.ravel()).item()<=n_dw*n_SM-1);
                        tmp_Z_SM__ = torch.reshape(Z_sub_dwSM____.ravel()[tmp_i8_index_all_SM_],mtr((n_S_sub,n_M_sub)));
                        tmp_index_dw_SM__ = torch.reshape(tmp_index_dw_SM_,mtr((n_S_sub,n_M_sub)));
                        tmp_index_delta_SM__ = torch.fmod(tmp_index_dw_SM__,n_delta_v).to(dtype=torch.int32);
                        tmp_index_gamma_SM__ = torch.div(tmp_index_dw_SM__ - tmp_index_delta_SM__,torch.tensor(np.maximum(1,n_delta_v)).to(dtype=torch.int32),rounding_mode='floor').to(dtype=torch.int32);
                        assert(torch.min(tmp_index_delta_SM__.ravel()).item()>=0); 
                        assert(torch.max(tmp_index_delta_SM__.ravel()).item()<=n_delta_v-1);
                        assert(torch.min(tmp_index_gamma_SM__.ravel()).item()>=0);
                        assert(torch.max(tmp_index_gamma_SM__.ravel()).item()<=n_w_max-1);
                        tmp_index_delta_SM__ = torch.reshape(tmp_index_delta_SM__,mtr((n_S_sub,n_M_sub)));
                        tmp_index_gamma_SM__ = torch.reshape(tmp_index_gamma_SM__,mtr((n_S_sub,n_M_sub)));
                        tmp_delta_x_SM__ = FTK['r8_delta_x_'].to(dtype=torch.float32).ravel()[tmp_index_delta_SM__];
                        tmp_delta_y_SM__ = FTK['r8_delta_y_'].to(dtype=torch.float32).ravel()[tmp_index_delta_SM__];
                        tmp_gamma_z_SM__ = 2*pi*tmp_index_gamma_SM__.to(dtype=torch.float32)/np.maximum(1,n_w_max);
                        tmp_i8_index_lhs_ = matlab_index_2d_0(n_S,index_nS_in_Sbatch_,n_M,index_nM_in_Mbatch_);
                        Z_SM__.ravel()[tmp_i8_index_lhs_] = tmp_Z_SM__.ravel();
                        X_SM__.ravel()[tmp_i8_index_lhs_] = tmp_X_SM__.ravel();
                        delta_x_SM__.ravel()[tmp_i8_index_lhs_] = torch.reshape(tmp_delta_x_SM__,mtr((n_S_sub,n_M_sub))).ravel();
                        delta_y_SM__.ravel()[tmp_i8_index_lhs_] = torch.reshape(tmp_delta_y_SM__,mtr((n_S_sub,n_M_sub))).ravel();
                        gamma_z_SM__.ravel()[tmp_i8_index_lhs_] = torch.reshape(tmp_gamma_z_SM__,mtr((n_S_sub,n_M_sub))).ravel();
                        index_sub_SM__.ravel()[tmp_i8_index_lhs_] = torch.reshape(tmp_index_dw_SM_.to(dtype=torch.int32),mtr((n_S_sub,n_M_sub))).ravel();
                        tmp_t = toc(tmp_t); 
                        if (flag_verbose>1): print(f' %% X_SM__: %0.6f',tmp_t);
                    #end;%if flag_optimize_over_gamma_z==1;
                #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;
                #end;%if (n_S_sub>0);
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;
            #end;%for nSbatch=0:n_Sbatch-1;
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;
        #end;%if (n_M_sub>0);
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;
    #end;%for nMbatch=0:n_Mbatch-1;
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;
    
    if flag_optimize_over_gamma_z==1:
        Z_wSM___ = Z_SM__;
        X_wSM___ = X_SM__;
        delta_x_wSM___ = delta_x_SM__;
        delta_y_wSM___ = delta_y_SM__;
        gamma_z_wSM___ = gamma_z_SM__;
        index_sub_wSM___ = index_sub_SM__;
    #end;%if flag_optimize_over_gamma_z==1;

    #%%%%%%%%;
    if (flag_verbose): print(f' %% [finished {str_thisfunction}]');
    #%%%%%%%%;
    return(
        parameter,
        Z_wSM___,
        UX_CTF_R_S_l2_wS__,
        CTF_R_S_l2_wS__,
        UX_T_M_l2_dM__,
        UX_M_l2_M_,
        X_wSM___,
        delta_x_wSM___,
        delta_y_wSM___,
        gamma_z_wSM___,
        index_sub_wSM___,
        Z_dwSM____,
        X_dwSM____,
    );

