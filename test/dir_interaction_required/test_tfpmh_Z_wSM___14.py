from dir_empm.dir_matlab_macros import * ;
from dir_empm.get_weight_3d_1 import get_weight_3d_1 ;
from dir_empm.get_weight_2d_2 import get_weight_2d_2 ;
from dir_empm.principled_marching_empirical_cost_matrix_1 import principled_marching_empirical_cost_matrix_1 ;
from dir_empm.tfh_FTK_4 import tfh_FTK_4 ;
from dir_empm.tfpmh_Z_wSM___14 import tfpmh_Z_wSM___14 ;
from dir_empm.rotate_p_to_p_fftw import rotate_p_to_p_fftw ;
from dir_empm.transf_p_to_p import transf_p_to_p ;

flag_verbose=1; flag_disp=0;nf=0;
flag_speed_vs_error = 0;

disp(sprintf(' %% testing tfpmh_Z_wSM___14: flag_speed_vs_error %d',flag_speed_vs_error));
n_1 = 1; n_2 = 2;
k_p_r_max = 48.0/(2*pi); k_eq_d = 1.0/(2*pi); str_L = 'L';
(
    n_k_p_r,
    k_p_r_,
    weight_3d_k_p_r_,
) = get_weight_3d_1(
    0*flag_verbose,
    k_p_r_max,
    k_eq_d,
    str_L,
)[:3];
#%%%%;
n_w_int = 1;
l_max_upb = matlab_scalar_round(2*pi*k_p_r_max);
l_max_max = int(np.minimum(l_max_upb,1+np.ceil(2*pi*k_p_r_[-1].item())));
n_w_max = int(n_w_int*2*(l_max_max+1)); n_w_0in_ = n_w_max*torch.ones(n_k_p_r).to(dtype=torch.int32);
(
    n_w_,
    weight_2d_k_p_r_,
    weight_2d_k_p_wk_,
    k_p_r_wk_,
    k_p_w_wk_,
    k_c_0_wk_,
    k_c_1_wk_,
) = get_weight_2d_2(
    0*flag_verbose,
    n_k_p_r,
    k_p_r_,
    k_p_r_max,
    -1,
    n_w_0in_,
    weight_3d_k_p_r_,
);
n_w_sum = int(torch.sum(n_w_).item()); n_w_csum_ = cumsum_0(n_w_);
#%%%%;
if flag_speed_vs_error==0: n_M = 17; #end;
if flag_speed_vs_error==1: n_M = 1024; #end;
n_source = 3;
delta_2sM___ = torch.permute(torch.reshape(torch.linspace(-0.05,+0.05,n_2*n_source*n_M),mtr((n_M,n_2,n_source))),mtr(mts((1,2,0))));
M_k_p_wkM__ = torch.zeros(mtr((n_w_sum,n_M))).to(dtype=torch.complex64);
for nM in range(n_M):
    for nsource in range(n_source):
        delta_0 = delta_2sM___[nM,nsource,0]; delta_1 = delta_2sM___[nM,nsource,1];
        M_k_p_wkM__[nM,:] = M_k_p_wkM__[nM,:] + 1*torch.exp(+i*2*pi*(k_c_0_wk_*delta_0 + k_c_1_wk_*delta_1));
    #end;%for nsource=0:n_source-1;
#end;%for nM=0:n_M-1;
#%%%%;
if flag_speed_vs_error==0: n_S = 19; #end;
if flag_speed_vs_error==1: n_S = 993; #end;
n_source = 3;
delta_2sS___ = torch.permute(torch.reshape(torch.linspace(-0.07,+0.07,n_2*n_source*n_S),mtr((n_S,n_2,n_source))),mtr(mts((1,2,0))));
S_k_p_wkS__ = torch.zeros(mtr((n_w_sum,n_S))).to(dtype=torch.complex64);
for nS in range(n_S):
    for nsource in range(n_source):
        delta_0 = delta_2sS___[nS,nsource,0]; delta_1 = delta_2sS___[nS,nsource,1];
        S_k_p_wkS__[nS,:] = S_k_p_wkS__[nS,:] + 1*torch.exp(+i*2*pi*(k_c_0_wk_*delta_0 + k_c_1_wk_*delta_1));
    #end;%for nsource=0:n_source-1;
#end;%for nS=0:n_S-1;
#%%%%;
n_CTF = 1;
CTF_k_p_r_k_ = 0.5 + 1.0*torch.cos(2*pi*k_p_r_);
#%%%%;
(
    X_kk__,
    X_weight_r_,
) = principled_marching_empirical_cost_matrix_1(
    n_k_p_r,
    k_p_r_,
    weight_2d_k_p_r_,
    n_w_,
    n_M,
    M_k_p_wkM__,
);
if flag_speed_vs_error==0: pm_n_UX_rank = int(np.minimum(2,n_k_p_r-1)); #end; #%<-- should be accurate even when radial-compression is lossy. ;
if flag_speed_vs_error==1: pm_n_UX_rank = 17; #end;
tmp_UX_kn__,tmp_SX_k_,tmp_VX_kn__ = matlab_svds(X_kk__,pm_n_UX_rank);
tmp_i8_index_rhs_ = matlab_index_2d_0(n_k_p_r,':',matlab_size(tmp_UX_kn__,1),torch.arange(pm_n_UX_rank));
pm_UX_kn__ = torch.reshape(tmp_UX_kn__.ravel()[tmp_i8_index_rhs_],mtr((n_k_p_r,pm_n_UX_rank)));
pm_X_weight_r_ = torch.sqrt(weight_2d_k_p_r_.ravel());
#%%%%;
r8_delta_r_max = 0.5/np.maximum(1e-12,k_p_r_max); 
if flag_speed_vs_error==0:
    r8_svd_eps = 1e-9; n_delta_v_requested =  9; 
#end;%if flag_speed_vs_error==0:
if flag_speed_vs_error==1:
    r8_svd_eps = 1e-2; n_delta_v_requested = 32;
#end;%if flag_speed_vs_error==1:
#%%%%;
parameter_FTK = {'type':'FTK'};
parameter_FTK['r8_delta_r_max'] = r8_delta_r_max;
parameter_FTK['r8_svd_eps'] = r8_svd_eps;
parameter_FTK['n_delta_v_requested'] = n_delta_v_requested;
parameter_FTK,FTK = tfh_FTK_4(parameter_FTK,n_k_p_r,k_p_r_,k_p_r_max);
n_delta_v = FTK['n_delta_v'];
n_svd_l = FTK['n_svd_l'];
#%%%%%%%%;
if (flag_verbose>0): disp(sprintf(' %% n_k_p_r %d n_w_max %d pm_n_UX_rank %d n_delta_v %d n_svd_l %d n_M %d n_S %d',n_k_p_r,n_w_max,pm_n_UX_rank,n_delta_v,n_svd_l,n_M,n_S)); #end;

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;
if flag_speed_vs_error==1:
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;
    for flag_tf_vs_bf in range(2):
        tmp_t = tic();
        parameter = {'type':'parameter'};
        parameter['flag_verbose'] = 1;
        parameter['n_M_per_Mbatch'] = 24;
        parameter['n_S_per_Sbatch'] = 24;
        parameter['flag_dwSM'] = 0;
        parameter['flag_optimize_over_gamma_z'] = 1;
        parameter['device_use'] = 'cpu';
        FTK['flag_tf_vs_bf'] = flag_tf_vs_bf;
        (
            parameter,
            tfpmh_Z_wSM___,
            tfpmh_UX_T_M_l2_dM__,
            tfpmh_UX_M_l2_M_,
            tfpmh_UX_CTF_S_l2_S_,
            tfpmh_X_wSM___,
            tfpmh_delta_x_wSM___,
            tfpmh_delta_y_wSM___,
            tfpmh_gamma_z_wSM___,
            tfpmh_index_sub_wSM___,
            tfpmh_Z_dwSM____,
            tfpmh_X_dwSM____,
        ) = tfpmh_Z_wSM___14(
            parameter,
            n_k_p_r,
            k_p_r_,
            k_p_r_max,
            n_w_,
            weight_2d_k_p_r_,
            weight_2d_k_p_wk_,
            n_M,
            M_k_p_wkM__,
            CTF_k_p_r_k_,
            n_S,
            S_k_p_wkS__,
            pm_n_UX_rank,
            pm_UX_kn__,
            pm_X_weight_r_,
            FTK,
        );
        tmp_t = toc(tmp_t);
        if (flag_verbose>0): disp(sprintf(' %% tfpmh_Z_wSM___14: %0.6f',tmp_t)); #end;
        parameter_timing_printf(parameter);
    #end;%for flag_tf_vs_bf=[0:1];
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;
#end;%if flag_speed_vs_error==1;
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;
if flag_speed_vs_error==0:
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;
    #%%%%%%%%%%%%%%%%;
    for flag_tf_vs_bf in range(2):
    #%%%%%%%%%%%%%%%%;
        #%%%%%%%%%%%%%%%%;
        for flag_optimize_over_gamma_z in range(2):
        #%%%%%%%%%%%%%%%%;
            parameter = {'type': 'parameter'} ;
            parameter['flag_verbose'] = 1;
            parameter['flag_optimize_over_gamma_z'] = flag_optimize_over_gamma_z;
            parameter['n_M_per_Mbatch'] = 3;
            parameter['n_S_per_Sbatch'] = 5;
            parameter['flag_dwSM'] = 1;
            parameter['flag_verbose'] = np.maximum(0,flag_verbose-1);
            FTK['flag_tf_vs_bf']=flag_tf_vs_bf;
            (
                parameter,
                tfpmh_Z_wSM___,
                tfpmh_UX_T_M_l2_dM__,
                tfpmh_UX_M_l2_M_,
                tfpmh_UX_CTF_S_l2_S_,
                tfpmh_X_wSM___,
                tfpmh_delta_x_wSM___,
                tfpmh_delta_y_wSM___,
                tfpmh_gamma_z_wSM___,
                tfpmh_index_sub_wSM___,
                tfpmh_Z_dwSM____,
                tfpmh_X_dwSM____,
            ) = tfpmh_Z_wSM___14(
                parameter,
                n_k_p_r,
                k_p_r_,
                k_p_r_max,
                n_w_,
                weight_2d_k_p_r_,
                weight_2d_k_p_wk_,
                n_M,
                M_k_p_wkM__,
                CTF_k_p_r_k_,
                n_S,
                S_k_p_wkS__,
                pm_n_UX_rank,
                pm_UX_kn__,
                pm_X_weight_r_,
                FTK,
            );
            if parameter['flag_optimize_over_gamma_z']==0:
                dir_base = '/data/rangan' ;
                dir_pymat = dir_base + '/dir_cryoem/dir_rangan_python/dir_pymat' ;
                fname_pymat = dir_pymat + '/test_tfpmh_Z_wSM___14.mat' ;
                disp(sprintf(' %% writing fname_pymat: %s',fname_pymat));
                matlab_save(
                    fname_mat=fname_pymat,
                    dictionary_original= {
                        "tfpmh_Z_wSM___":tfpmh_Z_wSM___,
                        "tfpmh_UX_T_M_l2_dM__":tfpmh_UX_T_M_l2_dM__,
                        "tfpmh_UX_M_l2_M_":tfpmh_UX_M_l2_M_,
                        "tfpmh_UX_CTF_S_l2_S_":tfpmh_UX_CTF_S_l2_S_,
                        "tfpmh_X_wSM___":tfpmh_X_wSM___,
                        "tfpmh_delta_x_wSM___":tfpmh_delta_x_wSM___,
                        "tfpmh_delta_y_wSM___":tfpmh_delta_y_wSM___,
                        "tfpmh_gamma_z_wSM___":tfpmh_gamma_z_wSM___,
                        "tfpmh_index_sub_wSM___":tfpmh_index_sub_wSM___,
                        "tfpmh_Z_dwSM____":tfpmh_Z_dwSM____,
                        "tfpmh_X_dwSM____":tfpmh_X_dwSM____,
                    },
                );
            #end;%if parameter.flag_optimize_over_gamma_z==0;
            if parameter['flag_optimize_over_gamma_z']==1:
                dir_base = '/data/rangan' ;
                dir_pymat = dir_base + '/dir_cryoem/dir_rangan_python/dir_pymat' ;
                fname_pymat = dir_pymat + '/test_tfpmh_Z_SM__14.mat' ;
                disp(sprintf(' %% writing fname_pymat: %s',fname_pymat));
                matlab_save(
                    fname_mat=fname_pymat,
                    dictionary_original= {
                        "tfpmh_Z_wSM___":tfpmh_Z_wSM___,
                        "tfpmh_UX_T_M_l2_dM__":tfpmh_UX_T_M_l2_dM__,
                        "tfpmh_UX_M_l2_M_":tfpmh_UX_M_l2_M_,
                        "tfpmh_UX_CTF_S_l2_S_":tfpmh_UX_CTF_S_l2_S_,
                        "tfpmh_X_wSM___":tfpmh_X_wSM___,
                        "tfpmh_delta_x_wSM___":tfpmh_delta_x_wSM___,
                        "tfpmh_delta_y_wSM___":tfpmh_delta_y_wSM___,
                        "tfpmh_gamma_z_wSM___":tfpmh_gamma_z_wSM___,
                        "tfpmh_index_sub_wSM___":tfpmh_index_sub_wSM___,
                        "tfpmh_Z_dwSM____":tfpmh_Z_dwSM____,
                        "tfpmh_X_dwSM____":tfpmh_X_dwSM____,
                    },
                );
            #end;%if parameter.flag_optimize_over_gamma_z==1;
            #%%%%%%%%;
            #% check. ;
            #%%%%%%%%;
            gamma_z_ = torch.linspace(0,2*pi,1+n_w_max).to(dtype=torch.float32); gamma_z_ = gamma_z_[0:n_w_max].ravel();
            pm_n_k_p_r = pm_n_UX_rank; pm_n_w_max = n_w_max;
            pm_n_w_ = pm_n_w_max*torch.ones(pm_n_k_p_r).to(dtype=torch.int32);
            pm_n_w_sum = pm_n_k_p_r*pm_n_w_max;
            pm_wUX_kn__ = mmmm( diag(pm_X_weight_r_) , pm_UX_kn__ );
            tmp_Z_errrel = 0.0;
            tmp_X_errrel = 0.0;
            for nS in range(n_S):
                for nM in range(n_M):
                    #%%%%;
                    if parameter['flag_optimize_over_gamma_z']==0:
                        nw = int(np.maximum(0,np.minimum(n_w_max-1,np.floor(n_w_max*np.random.rand())))); gamma_z = (2*pi*nw)/np.maximum(1,n_w_max);
                        delta_x = tfpmh_delta_x_wSM___[nM,nS,nw].item(); delta_y = tfpmh_delta_y_wSM___[nM,nS,nw].item();
                        Z_tfpm = tfpmh_Z_wSM___[nM,nS,nw].item();
                        X_tfpm = tfpmh_X_wSM___[nM,nS,nw].item();
                    #end;%if parameter.flag_optimize_over_gamma_z==0;
                    #%%%%;
                    if parameter['flag_optimize_over_gamma_z']==1:
                        gamma_z = tfpmh_gamma_z_wSM___[nM,nS].item();
                        delta_x = tfpmh_delta_x_wSM___[nM,nS].item(); delta_y = tfpmh_delta_y_wSM___[nM,nS].item();
                        Z_tfpm = tfpmh_Z_wSM___[nM,nS].item();
                        X_tfpm = tfpmh_X_wSM___[nM,nS].item();
                    #end;%if parameter.flag_optimize_over_gamma_z==1;
                    #%%%%;
                    ndelta_v = intersect_0(efind(torch.abs(FTK['r8_delta_x_']-delta_x)<1e-6),efind(torch.abs(FTK['r8_delta_y_']-delta_y)<1e-6))[0];
                    nw = efind(torch.abs(gamma_z_-gamma_z)<1e-6);
                    Z_tens = tfpmh_Z_dwSM____[nM,nS,nw,ndelta_v].item();
                    X_tens = tfpmh_X_dwSM____[nM,nS,nw,ndelta_v].item();
                    #%%%%;
                    S_k_p_wk_ = S_k_p_wkS__[nS,:].ravel();
                    M_k_p_wk_ = M_k_p_wkM__[nM,:].ravel();
                    CTF_k_p_wk_ = torch.tile(torch.reshape(CTF_k_p_r_k_,mtr((1,n_k_p_r))),mtr((n_w_max,1))).ravel(); assert(numel(CTF_k_p_wk_)==n_w_sum);
                    gamma_z = (2*pi*nw)/np.maximum(1,n_w_max);
                    CTF_S_k_p_wk_ = CTF_k_p_wk_*S_k_p_wk_;
                    UX_CTF_S_k_p_wn_ = mmmm( torch.reshape(CTF_S_k_p_wk_.to(dtype=torch.complex64),mtr((n_w_max,n_k_p_r))) , pm_wUX_kn__.to(dtype=torch.complex64) ).ravel(); assert(numel(UX_CTF_S_k_p_wn_)==pm_n_w_sum);
                    UX_CTF_S_l2 = torch.sum(torch.conj(UX_CTF_S_k_p_wn_) * UX_CTF_S_k_p_wn_).item() / np.maximum(1,pm_n_w_max) ;
                    RS_k_p_wk_ = rotate_p_to_p_fftw(n_k_p_r,n_w_,n_w_sum,S_k_p_wk_,+gamma_z);
                    CTF_RS_k_p_wk_ = RS_k_p_wk_*CTF_k_p_wk_;
                    UX_CTF_RS_k_p_wn_ = mmmm( torch.reshape(CTF_RS_k_p_wk_.to(dtype=torch.complex64),mtr((n_w_max,n_k_p_r))) , pm_wUX_kn__.to(dtype=torch.complex64)).ravel(); assert(numel(UX_CTF_RS_k_p_wn_)==pm_n_w_sum);
                    UX_CTF_RS_l2 = torch.sum(torch.conj(UX_CTF_RS_k_p_wn_)*UX_CTF_RS_k_p_wn_).item() / np.maximum(1,pm_n_w_max) ;
                    TM_k_p_wk_ = transf_p_to_p(n_k_p_r,k_p_r_,n_w_,n_w_sum,M_k_p_wk_,+delta_x,+delta_y);
                    UX_TM_k_p_wn_ = mmmm( torch.reshape(TM_k_p_wk_.to(dtype=torch.complex64),mtr((n_w_max,n_k_p_r))) , pm_wUX_kn__.to(dtype=torch.complex64) ).ravel(); assert(numel(UX_TM_k_p_wn_)==pm_n_w_sum);
                    UX_TM_l2 = torch.sum(torch.conj(UX_TM_k_p_wn_)*UX_TM_k_p_wn_).item() / np.maximum(1,pm_n_w_max) ;
                    UX_CTF_RS_k_p_UX_TM_k_p = torch.sum(torch.conj(UX_CTF_RS_k_p_wn_)*UX_TM_k_p_wn_).item() / np.maximum(1,pm_n_w_max) ;
                    Z_quad = UX_CTF_RS_k_p_UX_TM_k_p;
                    X_quad = Z_quad / np.maximum(1e-12,np.sqrt(UX_CTF_RS_l2)) / np.maximum(1e-12,np.sqrt(UX_TM_l2));
                    #%%%%;
                    if (flag_verbose>2): disp(sprintf(' %% nS %.2d/%.2d nM %.2d/%.2d',nS,n_S,nM,n_M)); #end;
                    tmp_Z_errrel = tmp_Z_errrel + fnorm_disp(max(0,flag_verbose-1),'Z_quad',Z_quad,'Z_tfpm',Z_tfpm,' %%<-- should be zero');
                    tmp_Z_errrel = tmp_Z_errrel + fnorm_disp(max(0,flag_verbose-1),'Z_quad',Z_quad,'Z_tens',Z_tens,' %%<-- should be zero');
                    tmp_Z_errrel = tmp_Z_errrel + fnorm_disp(max(0,flag_verbose-1),'Z_tfpm',Z_tfpm,'Z_tens',Z_tens,' %%<-- should be zero');
                    tmp_X_errrel = tmp_X_errrel + fnorm_disp(max(0,flag_verbose-1),'X_quad',X_quad,'X_tfpm',X_tfpm,' %%<-- should be zero');
                    tmp_X_errrel = tmp_X_errrel + fnorm_disp(max(0,flag_verbose-1),'X_quad',X_quad,'X_tens',X_tens,' %%<-- should be zero');
                    tmp_X_errrel = tmp_X_errrel + fnorm_disp(max(0,flag_verbose-1),'X_tfpm',X_tfpm,'X_tens',X_tens,' %%<-- should be zero');
                    #%%%%;
                #end;%for nM=0:n_M-1;
            #end;%for nS=0:n_S-1;
            if (flag_verbose>0): disp(sprintf(' %% parameter.flag_optimize_over_gamma_z = %d; tmp_Z_errrel: %0.16f',parameter['flag_optimize_over_gamma_z'],tmp_Z_errrel)); #end;
            if (flag_verbose>0): disp(sprintf(' %% parameter.flag_optimize_over_gamma_z = %d; tmp_X_errrel: %0.16f',parameter['flag_optimize_over_gamma_z'],tmp_X_errrel)); #end;
        #%%%%%%%%%%%%%%%%;
        #end;%for flag_optimize_over_gamma_z=[0:1];
        #%%%%%%%%%%%%%%%%;
    #%%%%%%%%%%%%%%%%;
    #end;%for flag_tf_vs_tf=[0:1];
    #%%%%%%%%%%%%%%%%;
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;
#end;%if flag_speed_vs_error==0:
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;

