from dir_empm.dir_matlab_macros import * ;
from dir_empm.r8_Jp_chebcoef_0 import r8_Jp_chebcoef_0 ;
from dir_empm.r8_chebval_0 import r8_chebval_0 ;
from scipy.special import roots_jacobi ;
from scipy.special import jv ;

def gen_Jsvd_FTK_8(
        r8_K_max=None,
        r8_N_pixel=None,
        r8_eps_target=None,
        l_max=None,
        n_a_degree=None,
        n_b_degree=None,
):
    flag_verbose=1;
    str_thisfunction = 'gen_Jsvd_FTK_8' ;

    #%%%%%%%%;
    if (flag_verbose): print(f' %% [entering {str_thisfunction}]');
    #%%%%%%%%;

    if r8_eps_target is None: r8_eps_target = 1e-3;
    if l_max is None: l_max = int(24);
    if n_a_degree is None: n_a_degree = int(32);
    if n_b_degree is None: n_b_degree = int(32);

    if (flag_verbose>1): print(f' %% [entering gen_Jsvd_FTK_8] r8_N_pixel {r8_N_pixel:.2f} l_max {l_max}, n_a_degree {n_a_degree}, n_b_degree {n_b_degree}, r8_eps_target {r8_eps_target:.6f}');

    tmp_t=tic();
    #%%%%%%%%;
    r8_K_target = float(r8_K_max);
    r8_z_target = float(r8_N_pixel)*pi*np.sqrt(2);
    r8_D_target = r8_z_target/np.maximum(1e-12,2*pi*r8_K_target);
    #%%%%%%%%;
    r8_r_max = 2*pi*r8_K_target;
    r8_d_max = r8_D_target;
    r8_a_m = r8_r_max/2; r8_a_r = r8_a_m;
    r8_b_m = r8_d_max/2; r8_b_r = r8_b_m;
    #%%%%%%%%;
    _,r8_a_Jp_chebcoef__ = r8_Jp_chebcoef_0([],n_a_degree);
    for na_degree in range(n_a_degree): r8_a_Jp_chebcoef__[:,na_degree] = r8_a_Jp_chebcoef__[:,na_degree]*np.sqrt(na_degree+1)/np.sqrt(2);
    _,r8_b_Jp_chebcoef__ = r8_Jp_chebcoef_0([],n_b_degree);
    for nb_degree in range(n_b_degree): r8_b_Jp_chebcoef__[:,nb_degree] = r8_b_Jp_chebcoef__[:,nb_degree]*np.sqrt(nb_degree+1)/np.sqrt(2);
    #%%%%%%%%;
    np_a_jx_,np_a_jw_ = roots_jacobi(n_a_degree,0,1); #%<-- assume returns float64. ;
    r8_a_jx_ = torch.tensor(np_a_jx_).to(dtype=torch.float64).ravel();
    r8_a_jw_ = torch.tensor(np_a_jw_).to(dtype=torch.float64).ravel();
    r8_a_Jx__ = torch.zeros(mtr((n_a_degree,n_a_degree))).to(dtype=torch.float64);
    for na_degree in range(n_a_degree):
        tmp_i8_index_rhs_ = matlab_index_2d_0(n_a_degree,na_degree,n_a_degree,torch.arange(na_degree+1).to(dtype=torch.int32));
        r8_a_Jx__[:,na_degree] = r8_chebval_0(1+na_degree,r8_a_Jp_chebcoef__.ravel()[tmp_i8_index_rhs_],n_a_degree,r8_a_jx_);
    #end;%for na_degree=0:n_a_degree-1;
    r8_a_jt_ = r8_a_jx_*r8_a_r + r8_a_m;
    #%%%%%%%%;
    np_b_jx_,np_b_jw_ = roots_jacobi(n_b_degree,0,1); #%<-- assume returns float64. ;
    r8_b_jx_ = torch.tensor(np_b_jx_).to(dtype=torch.float64).ravel();
    r8_b_jw_ = torch.tensor(np_b_jw_).to(dtype=torch.float64).ravel();
    r8_b_Jx__ = torch.zeros(mtr((n_b_degree,n_b_degree))).to(dtype=torch.float64);
    for nb_degree in range(n_b_degree):
        tmp_i8_index_rhs_ = matlab_index_2d_0(n_b_degree,nb_degree,n_b_degree,torch.arange(nb_degree+1).to(dtype=torch.int32));
        r8_b_Jx__[:,nb_degree] = r8_chebval_0(1+nb_degree,r8_b_Jp_chebcoef__.ravel()[tmp_i8_index_rhs_],n_b_degree,r8_b_jx_);
    #end;%for nb_degree=0:n_b_degree-1;
    r8_b_jt_ = r8_b_jx_*r8_b_r + r8_b_m;
    #%%%%%%%%;
    [r8_B_jt_ab__,r8_A_jt_ab__] = torch.meshgrid(r8_b_jt_,r8_a_jt_,indexing='ij'); #<-- reversed to match matlab. ;
    #%%%%%%%%;
    tmp_t=toc(tmp_t);
    if (flag_verbose>0): disp(sprintf(' %% gen_Jsvd_FTK_8: phase_0: %0.6fs',tmp_t)); #end;

    tmp_t=tic();
    n_svds = int(np.minimum(n_a_degree,n_b_degree));
    n_sl = n_svds*(1+2*l_max);
    i4_S_l_ = torch.zeros(mtr((1,n_sl))).to(dtype=torch.int32);
    r8_S_u__ = torch.zeros(mtr((n_b_degree,n_sl))).to(dtype=torch.float64);
    r8_S_s_ = torch.zeros(mtr((1,n_sl))).to(dtype=torch.float64);
    r8_S_v__ = torch.zeros(mtr((n_a_degree,n_sl))).to(dtype=torch.float64);
    l=0; n_S=0; continue_flag=1;
    while (continue_flag):
        if (l==0): l_ = [0]; 
        if (l!=0): l_ = [-l,+l];
        for l_tmp in l_:
            r8_F = lambda a,b: jv(l_tmp,a*b).to(dtype=torch.float64);
            r8_F_jt_ab__ = r8_F(r8_A_jt_ab__,r8_B_jt_ab__);
            r8_F_ab__ = torch.zeros(mtr((n_a_degree,n_b_degree))).to(dtype=torch.float64);
            #r8_jw_ab__ = torch.reshape(r8_a_jw_,mtr((n_a_degree,1)))*torch.reshape(r8_b_jw_,mtr((1,n_b_degree)));
            #for na_degree in range(n_a_degree):
            #    for nb_degree in range(n_b_degree):
            #        r8_J_tmp_ab__ = torch.reshape(r8_a_Jx__[:,na_degree],mtr((n_a_degree,1)))*torch.reshape(r8_b_Jx__[:,nb_degree],mtr((1,n_b_degree)));
            #        r8_S_tmp_ab__ = r8_F_jt_ab__*r8_J_tmp_ab__*r8_jw_ab__;
            #        r8_F_ab__[nb_degree,na_degree] = torch.sum(r8_S_tmp_ab__.ravel()).item();
            #    #end;%for nb_degree=0:n_b_degree-1;
            ##end;%for na_degree=0:n_a_degree-1;
            r8_a_Jx_w_a1a1____ = torch.reshape(r8_a_Jx__.T,mtr((n_a_degree,n_1,n_a_degree,n_1)))*torch.reshape(r8_a_jw_,mtr((n_a_degree,n_1,n_1,n_1))) ;
            r8_b_Jx_w_1b1b____ = torch.reshape(r8_b_Jx__.T,mtr((n_1,n_b_degree,n_1,n_b_degree)))*torch.reshape(r8_b_jw_,mtr((n_1,n_b_degree,n_1,n_1))) ;
            r8_F_jt_ab11____ = torch.reshape(r8_F_jt_ab__,mtr((n_a_degree,n_b_degree,n_1,n_1)));
            r8_F_ab__ = torch.reshape(torch.sum(r8_a_Jx_w_a1a1____*r8_b_Jx_w_1b1b____*r8_F_jt_ab11____,dim=(3-0,3-1)),mtr((n_a_degree,n_b_degree)));
            r8_U_t__,r8_Sigma_,r8_V_n__ = torch.linalg.svd(r8_F_ab__.T.T,full_matrices=False); r8_U_n__ = r8_U_t__.T; #%<-- extra transposes to match [U_n__,S__,V_n__] = svds(transpose(r8_F_ab__),n_svds) in matlab. ;
            index_ret_ = efind(r8_Sigma_>r8_eps_target) ;
            if numel(index_ret_)>0:
                if (flag_verbose>1):
                    print(f' %% l {l_tmp:+02d}, found {len(index_ret_)} terms [{r8_Sigma_[index_ret_[0]]:.2f},...,{r8_Sigma_[index_ret_[-1]]:.2f}];');
                #end;%if
                for index in range(numel(index_ret_)):
                    i4_S_l_[n_S,0] = l_tmp;
                    r8_S_u__[n_S,:] = r8_U_n__[index_ret_[index],:];
                    r8_S_s_[n_S,0] = r8_Sigma_[index_ret_[index]];
                    r8_S_v__[n_S,:] = r8_V_n__[index_ret_[index],:];
                    n_S = n_S + 1;
                #end;%for index = 0:length(index_ret_)-1;
            #end;%if ~isempty(index_ret_);
        #end;%for l_tmp = l_;
        l=l+1;
        if (l> l_max): continue_flag=0;
        if (l<=l_max): continue_flag=1;
    #end;%while (continue_flag);
    if (flag_verbose>1): print(f' %% total of n_S {n_S} terms found;');
    i4_S_l_ = i4_S_l_[torch.arange(n_S),0];
    r8_S_u__ = r8_S_u__[torch.arange(n_S),:];
    r8_S_s_ = r8_S_s_[torch.arange(n_S),0];
    r8_S_v__ = r8_S_v__[torch.arange(n_S),:];
    tmp_t=toc(tmp_t);
    if (flag_verbose>0): disp(sprintf(' %% gen_Jsvd_FTK_8: phase_1: %0.6fs',tmp_t)); #end;

    tmp_t=tic();
    FTK = {'type': 'FTK'};
    FTK['n_r_degree'] = n_a_degree; FTK['n_d_degree'] = n_b_degree;
    FTK['n_svd_r'] = numel(r8_a_jt_); FTK['r8_svd_r_'] = r8_a_jt_;
    FTK['r8_svd_r_m'] = r8_a_m; FTK['r8_svd_r_c'] = r8_a_r; FTK['r8_svd_r_w_'] = r8_a_jw_; FTK['r8_svd_r_Jx__'] = r8_a_Jx__; 
    FTK['r8_svd_r_Jp_chebcoef__'] = r8_a_Jp_chebcoef__;
    FTK['n_svd_d'] = numel(r8_b_jt_); FTK['r8_svd_d_'] = r8_b_jt_;
    FTK['r8_svd_d_m'] = r8_b_m; FTK['r8_svd_d_c'] = r8_b_r; FTK['r8_svd_d_w_'] = r8_b_jw_; FTK['r8_svd_d_Jx__'] = r8_b_Jx__;
    FTK['r8_svd_d_Jp_chebcoef__'] = r8_b_Jp_chebcoef__;
    FTK['n_svd_l'] = n_S;
    FTK['svd_l_'] = [];
    if n_S> 0: FTK['i4_svd_l_'] = i4_S_l_;
    FTK['r8_svd_U_d_jacocoef_'] = []; 
    if n_S> 0: FTK['r8_svd_U_d_jacocoef_'] = r8_S_u__;
    FTK['r8_svd_s_'] = [];
    if n_S> 0: FTK['r8_svd_s_'] = r8_S_s_;
    FTK['r8_svd_V_r_jacocoef_'] = [];
    if n_S> 0: FTK['r8_svd_V_r_jacocoef_'] = r8_S_v__;
    tmp_t=toc(tmp_t);
    if (flag_verbose>0): disp(sprintf(' %% gen_Jsvd_FTK_8: phase_2: %0.6fs',tmp_t)); #end;

    tmp_t=tic();
    r8_svd_j_d_chebcoef_ = torch.zeros(mtr((FTK['n_d_degree'],FTK['n_svd_l']))).to(dtype=torch.float64);
    r8_svd_j_r_chebcoef_ = torch.zeros(mtr((FTK['n_r_degree'],FTK['n_svd_l']))).to(dtype=torch.float64);
    for nl in range(FTK['n_svd_l']):
        #%%%%%%%%;
        tmp_r8_p_j_chebcoef_ = torch.zeros(FTK['n_d_degree']).to(dtype=torch.float64);
        for nd_degree in range(FTK['n_d_degree']):
            tmp_i8_index_rhs_ = matlab_index_2d_0(n_b_degree,nd_degree,n_b_degree,torch.arange(nd_degree+1));
            tmp_r8_j_chebcoef_ = FTK['r8_svd_d_Jp_chebcoef__'].ravel()[tmp_i8_index_rhs_]*FTK['r8_svd_U_d_jacocoef_'][nl,nd_degree];
            tmp_index_ = torch.arange(int(np.minimum(FTK['n_d_degree'],numel(tmp_r8_j_chebcoef_)))).to(dtype=torch.int32);
            tmp_r8_p_j_chebcoef_[tmp_index_] = tmp_r8_p_j_chebcoef_.ravel()[tmp_index_] + tmp_r8_j_chebcoef_.ravel()[tmp_index_];
        #end;% for nd_degree=0:FTK['n_d_degree']-1;
        r8_svd_j_d_chebcoef_[nl,:] = tmp_r8_p_j_chebcoef_;
        #%%%%%%%%;
        tmp_r8_p_j_chebcoef_ = torch.zeros(FTK['n_r_degree']).to(dtype=torch.float64);
        for nr_degree in range(FTK['n_r_degree']):
            tmp_i8_index_rhs_ = matlab_index_2d_0(n_a_degree,nr_degree,n_a_degree,torch.arange(nr_degree+1));
            tmp_r8_j_chebcoef_ = FTK['r8_svd_r_Jp_chebcoef__'].ravel()[tmp_i8_index_rhs_]*FTK['r8_svd_V_r_jacocoef_'][nl,nr_degree];
            tmp_index_ = torch.arange(int(np.minimum(FTK['n_r_degree'],numel(tmp_r8_j_chebcoef_)))).to(dtype=torch.int32);
            tmp_r8_p_j_chebcoef_[tmp_index_] = tmp_r8_p_j_chebcoef_.ravel()[tmp_index_] + tmp_r8_j_chebcoef_.ravel()[tmp_index_];
        #end;%for nr_degree=0:FTK['n_r_degree']-1;
        r8_svd_j_r_chebcoef_[nl,:] = tmp_r8_p_j_chebcoef_;
        #%%%%%%%%;
    #end;%for nl=0:FTK.n_svd_l-1;
    tmp_t=toc(tmp_t);
    if (flag_verbose>0): disp(sprintf(' %% gen_Jsvd_FTK_8: phase_3: %0.6fs',tmp_t)); #end;

    FTK['r8_svd_U_d_chebcoef_'] = r8_svd_j_d_chebcoef_.ravel();
    FTK['r8_svd_V_r_chebcoef_'] = r8_svd_j_r_chebcoef_.ravel();

    #%%%%%%%%;
    if (flag_verbose): print(f' %% [finished {str_thisfunction}]');
    #%%%%%%%%;
    return(FTK);

