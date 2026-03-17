from dir_empm.dir_matlab_macros import * ;
from dir_empm.cg_rhs_2 import cg_rhs_2 ;
from dir_empm.sample_sphere_7 import sample_sphere_7 ;
from dir_empm.sample_shell_6 import sample_shell_6 ;
from dir_empm.get_weight_2d_2 import get_weight_2d_2 ;
from dir_empm.rotate_p_to_p_fftw_using_numpy import rotate_p_to_p_fftw_using_numpy ;
from dir_empm.rotate_p_to_p_fftw import rotate_p_to_p_fftw ;

def R2(gamma_z):
    R2__ = torch.zeros(mtr((2,2))).to(dtype=torch.float32);
    na=0;
    #R2__.ravel()[na] = +np.cos(gamma_z); na=na+1;
    #R2__.ravel()[na] = +np.sin(gamma_z); na=na+1;
    #R2__.ravel()[na] = -np.sin(gamma_z); na=na+1;
    #R2__.ravel()[na] = +np.cos(gamma_z); na=na+1;
    R2__ = matlab_assign_lhs_from_rhs_(R2__,na, +np.cos(gamma_z)); na=na+1;
    R2__ = matlab_assign_lhs_from_rhs_(R2__,na, +np.sin(gamma_z)); na=na+1;
    R2__ = matlab_assign_lhs_from_rhs_(R2__,na, -np.sin(gamma_z)); na=na+1;
    R2__ = matlab_assign_lhs_from_rhs_(R2__,na, +np.cos(gamma_z)); na=na+1;
    assert(na==4);
    return(R2__);
#end;def;

def Rz(azimu_b):
    Rz__ = torch.zeros(mtr((3,3))).to(dtype=torch.float32);
    na=0;
    #Rz__.ravel()[na] = +np.cos(azimu_b); na=na+1;
    #Rz__.ravel()[na] = +np.sin(azimu_b); na=na+1;
    #Rz__.ravel()[na] =                0; na=na+1;
    #Rz__.ravel()[na] = -np.sin(azimu_b); na=na+1;
    #Rz__.ravel()[na] = +np.cos(azimu_b); na=na+1;
    #Rz__.ravel()[na] =                0; na=na+1;
    #Rz__.ravel()[na] =                0; na=na+1;
    #Rz__.ravel()[na] =                0; na=na+1;
    #Rz__.ravel()[na] =                1; na=na+1;
    Rz__ = matlab_assign_lhs_from_rhs_(Rz__,na, +np.cos(azimu_b)); na=na+1;
    Rz__ = matlab_assign_lhs_from_rhs_(Rz__,na, +np.sin(azimu_b)); na=na+1;
    Rz__ = matlab_assign_lhs_from_rhs_(Rz__,na,                0); na=na+1;
    Rz__ = matlab_assign_lhs_from_rhs_(Rz__,na, -np.sin(azimu_b)); na=na+1;
    Rz__ = matlab_assign_lhs_from_rhs_(Rz__,na, +np.cos(azimu_b)); na=na+1;
    Rz__ = matlab_assign_lhs_from_rhs_(Rz__,na,                0); na=na+1;
    Rz__ = matlab_assign_lhs_from_rhs_(Rz__,na,                0); na=na+1;
    Rz__ = matlab_assign_lhs_from_rhs_(Rz__,na,                0); na=na+1;
    Rz__ = matlab_assign_lhs_from_rhs_(Rz__,na,                1); na=na+1;
    assert(na==9);
    return(Rz__);
#end;def;

def Ry(polar_a):
    Ry__ = torch.zeros(mtr((3,3))).to(dtype=torch.float32);
    na=0;
    #Ry__.ravel()[na] = +np.cos(polar_a); na=na+1;
    #Ry__.ravel()[na] =                0; na=na+1;
    #Ry__.ravel()[na] = -np.sin(polar_a); na=na+1;
    #Ry__.ravel()[na] =                0; na=na+1;
    #Ry__.ravel()[na] =                1; na=na+1;
    #Ry__.ravel()[na] =                0; na=na+1;
    #Ry__.ravel()[na] = +np.sin(polar_a); na=na+1;
    #Ry__.ravel()[na] =                0; na=na+1;
    #Ry__.ravel()[na] = +np.cos(polar_a); na=na+1;
    Ry__ = matlab_assign_lhs_from_rhs_(Ry__,na, +np.cos(polar_a)); na=na+1;
    Ry__ = matlab_assign_lhs_from_rhs_(Ry__,na,                0); na=na+1;
    Ry__ = matlab_assign_lhs_from_rhs_(Ry__,na, -np.sin(polar_a)); na=na+1;
    Ry__ = matlab_assign_lhs_from_rhs_(Ry__,na,                0); na=na+1;
    Ry__ = matlab_assign_lhs_from_rhs_(Ry__,na,                1); na=na+1;
    Ry__ = matlab_assign_lhs_from_rhs_(Ry__,na,                0); na=na+1;
    Ry__ = matlab_assign_lhs_from_rhs_(Ry__,na, +np.sin(polar_a)); na=na+1;
    Ry__ = matlab_assign_lhs_from_rhs_(Ry__,na,                0); na=na+1;
    Ry__ = matlab_assign_lhs_from_rhs_(Ry__,na, +np.cos(polar_a)); na=na+1;
    assert(na==9);
    return(Ry__);
#end;def;

def test_rbp_0(
        k_eq_d_double = None,
        k_int = None,
        viewing_k_eq_d_double = None,
):
    str_thisfunction = 'test_rbp_0';
    flag_verbose=1;
    if (flag_verbose> 0): print(f" %% [entering {str_thisfunction}]");
    np.random.seed(0);
    if k_eq_d_double is None: k_eq_d_double = 1.0;
    if k_int is None: k_int = 48;
    if viewing_k_eq_d_double is None: viewing_k_eq_d_double = 1.0;

    if (flag_verbose>0):
        print(" %% %% %% %% %% %% %% %% ");
        print(f" %% setting k_eq_d_double = {k_eq_d_double:.6f}");
        print(f" %% setting k_int = {k_int} ");
        print(f" %% setting viewing_k_eq_d_double = {viewing_k_eq_d_double:.6f}");
    #end;%if (flag_verbose>0);

    k_p_r_max = k_int/(2*pi); k_eq_d = k_eq_d_double/(2*pi); str_T_vs_L = 'L';
    flag_uniform_over_n_k_p_r = 1; flag_uniform_over_polar_a = 0;
    (
        n_qk,
        n_qk_csum_,
        k_p_r_qk_,
        k_p_azimu_b_qk_,
        k_p_polar_a_qk_,
        weight_3d_k_p_qk_,
        weight_shell_qk_,
        n_k_p_r,
        k_p_r_,
        weight_3d_k_p_r_,
        k_c_0_qk_,
        k_c_1_qk_,
        k_c_2_qk_,
        J_node_,
        J_weight_,
        J_chebfun_,
        J_polyval_,
        n_polar_a_k_,
        polar_a_ka__,
        n_azimu_b_ka__,
    ) = sample_sphere_7(
        flag_verbose,
        k_p_r_max,
        k_eq_d,
        str_T_vs_L,
        flag_uniform_over_n_k_p_r,
        flag_uniform_over_polar_a,
        );

    if (flag_verbose>0):
        print(f" %% setting k_p_r_max = {k_p_r_max:.6f}");
        print(f" %% setting k_eq_d = {k_eq_d:.6f}");
        print(f" %% n_qk = {n_qk}");
    #end;%if (flag_verbose>0);
        
    index_outer_shell_ = torch.arange(int(n_qk_csum_[n_k_p_r-1].item()),int(n_qk_csum_[n_k_p_r-1+1].item())).to(dtype=torch.int32);
    n_outer_shell = numel(index_outer_shell_);
    k_outer_shell_r_q_ = k_p_r_qk_[index_outer_shell_];
    assert(np.std(k_outer_shell_r_q_.numpy(),ddof=1)< 1e-3);
    k_outer_shell_azimu_b_q_ = k_p_azimu_b_qk_[index_outer_shell_];
    k_outer_shell_polar_a_q_ = k_p_polar_a_qk_[index_outer_shell_];

    if (flag_verbose>2):
        np.set_printoptions(threshold=10);
        print('index_outer_shell_');print(index_outer_shell_);
        print('n_outer_shell');print(n_outer_shell);
        print('k_outer_shell_r_q_');print(k_outer_shell_r_q_);
        print('k_outer_shell_azimu_b_q_');print(k_outer_shell_azimu_b_q_);
        print('k_outer_shell_polar_a_q_');print(k_outer_shell_polar_a_q_);
    #end;%if (flag_verbose>2);

    n_w_int = int(1);
    l_max_upb = matlab_scalar_round(2*pi*k_p_r_max);
    n_w_max = int(n_w_int*2*(l_max_upb+1));
    template_k_eq_d = int(-1);
    n_w_0in_ = n_w_max*torch.ones(n_k_p_r).to(dtype=torch.int32);
    (
        n_w_,
        weight_2d_k_p_r_,
        weight_2d_k_p_wk_,
        k_p_r_wk_,
        k_p_w_wk_,
        k_c_0_wk_,
        k_c_1_wk_,
    ) = get_weight_2d_2(
        flag_verbose,
        n_k_p_r,
        k_p_r_,
        k_p_r_max,
        template_k_eq_d,
        n_w_0in_,
        weight_3d_k_p_r_,
    );
    n_w_max = int(torch.max(n_w_).item());
    n_w_sum = int(torch.sum(n_w_).item());
    n_w_csum_ = cumsum_0(n_w_);

    if (flag_verbose>2):
        np.set_printoptions(threshold=10);
        print('n_w_');print(n_w_);
        print('weight_2d_k_p_r_');print(weight_2d_k_p_r_);
        print('weight_2d_k_p_wk_');print(weight_2d_k_p_wk_);
        print('k_p_r_wk_');print(k_p_r_wk_);
        print('k_p_w_wk_');print(k_p_w_wk_);
        print('k_c_0_wk_');print(k_c_0_wk_);
        print('k_c_1_wk_');print(k_c_1_wk_);
    #end;%if (flag_verbose>2);

    n_3 = 3;
    delta_a_c_3s__ = torch.transpose(
        torch.tensor([
            [ +1.5, -0.5 ],
            [ -0.5, -1.5 ],
            [ +0.3, +2.0 ]
        ]) / (2 * k_p_r_max) 
        , 1 , 0 ).to(torch.float32);
    n_source = delta_a_c_3s__.shape[0];

    if (flag_verbose>2):
        np.set_printoptions(threshold=10);
        print('delta_a_c_3s__');print(delta_a_c_3s__);
        print('n_source');print(n_source);
    #end;%if (flag_verbose>2);

    a_k_p_form_ = torch.zeros(n_qk).to(dtype=torch.complex64);
    for nsource in range(n_source):
        delta_a_c_ = delta_a_c_3s__[nsource,:];
        a_k_p_form_ += torch.exp(
            i * 2 * pi * (
                k_c_0_qk_ * delta_a_c_[0] +
                k_c_1_qk_ * delta_a_c_[1] +
                k_c_2_qk_ * delta_a_c_[2]
            )
        );
    #end;%for;

    if (flag_verbose>2):
        np.set_printoptions(threshold=10);
        print('a_k_p_form_');print(a_k_p_form_);
    #end;%if (flag_verbose>2);

    viewing_k_eq_d = viewing_k_eq_d_double*1.0/np.maximum(1e-12,k_p_r_max);
    flag_uniform_over_polar_a = 0;
    (
        n_viewing_S,
        viewing_azimu_b_S_,
        viewing_polar_a_S_,
        viewing_weight_S_,
        _,
        _,
        _,
        _,
        _,
        _,
    )= sample_shell_6(
        1.0,
        viewing_k_eq_d,
        'L',
        flag_uniform_over_polar_a,
    );
    n_S = n_viewing_S;
    if (flag_verbose>0):
        print(f" %% viewing_k_eq_d_double: {viewing_k_eq_d_double:.2f}, n_S: {n_S}");
    #end;%if (flag_verbose>0);

    S_k_p_wkS__ = torch.zeros(mtr((n_w_sum,n_S))).to(dtype=torch.complex64);
    for nS in range(n_S):
        tmp_azimu_b = viewing_azimu_b_S_[nS].item();
        tmp_polar_a = viewing_polar_a_S_[nS].item();
        tmp_gamma_z = 0.0;
        tmp_R__ = mmmm(Rz(-tmp_gamma_z),mmmm(Ry(-tmp_polar_a),Rz(-tmp_azimu_b)));
        S_k_p_wk_ = torch.zeros(n_w_sum).to(dtype=torch.complex64);
        for nsource in range(n_source):
            tmp_delta_ = mmvm(tmp_R__,delta_a_c_3s__[nsource,:].ravel());
            S_k_p_wk_ = S_k_p_wk_ + torch.exp(+i*2*pi*(k_c_0_wk_*tmp_delta_[0].item() + k_c_1_wk_*tmp_delta_[1].item()));
        #end;for nsource in range(n_source):
        tmp_i8_index_lhs_ = matlab_index_2d_0(n_w_sum,':',n_S,nS);
        #S_k_p_wkS__.ravel()[tmp_i8_index_lhs_] = S_k_p_wk_.ravel();
        S_k_p_wkS__ = matlab_assign_lhs_from_rhs_(S_k_p_wkS__,tmp_i8_index_lhs_, S_k_p_wk_.ravel());
    #end;%for nS=0:n_S-1;

    if (flag_verbose>2):
        np.set_printoptions(threshold=10);
        print('S_k_p_wkS__');print(S_k_p_wkS__);
    #end;%if (flag_verbose>2);

    n_M = n_S*2;
    index_nS_from_nM_ = torch.tensor(np.maximum(0,np.minimum(n_S-1,np.floor(np.mod(np.arange(n_M),n_S)))).astype(int)).to(dtype=torch.int32);
    euler_polar_a_M_ = viewing_polar_a_S_[index_nS_from_nM_];
    euler_azimu_b_M_ = viewing_azimu_b_S_[index_nS_from_nM_];
    euler_gamma_z_M_ = 2*pi*torch.arange(n_M).to(dtype=torch.float32) / np.maximum(1,n_M) + pi/2.0;

    if (flag_verbose>2):
        np.set_printoptions(threshold=10);
        print('n_M');print(n_M);
        print('index_nS_from_nM_');print(index_nS_from_nM_);
        print('euler_polar_a_M_');print(euler_polar_a_M_);
        print('euler_azimu_b_M_');print(euler_azimu_b_M_);
        print('euler_gamma_z_M_');print(euler_gamma_z_M_);
    #end;%if (flag_verbose>2);

    n_CTF = 8  # <-- some number of CTF-functions. ;
    CTF_phi_C_ = torch.zeros(n_CTF).to(dtype=torch.float32);
    CTF_k_p_wkC__ = torch.zeros(mtr((n_w_sum,n_CTF))).to(dtype=torch.complex64);
    for nCTF in range(n_CTF):
        CTF_phi = 2*pi*nCTF / np.maximum(1,n_CTF) + pi/2.0;
        CTF_phi_C_[nCTF] = CTF_phi;
        CTF_k_p_wk_ = 2*k_p_r_wk_ * torch.cos(k_p_w_wk_-CTF_phi);
        tmp_i8_index_lhs_ = matlab_index_2d_0(n_w_sum,':',n_CTF,nCTF);
        #CTF_k_p_wkC__.ravel()[tmp_i8_index_lhs_] = CTF_k_p_wk_.ravel().to(dtype=torch.complex64);
        CTF_k_p_wkC__ = matlab_assign_lhs_from_rhs_(CTF_k_p_wkC__,tmp_i8_index_lhs_, CTF_k_p_wk_.ravel().to(dtype=torch.complex64));
    #end;%for nCTF=0:n_CTF-1;
    index_nCTF_from_nM_ = torch.tensor(np.mod(np.arange(n_M),n_CTF).astype(int)).to(dtype=torch.int32);

    if (flag_verbose>2):
        np.set_printoptions(threshold=10);
        print('n_CTF');print(n_CTF);
        print('index_nCTF_from_nM_');print(index_nCTF_from_nM_);
        print('CTF_phi_C_');print(CTF_phi_C_);
        print('CTF_k_p_wkC__');print(CTF_k_p_wkC__);
    #end;%if (flag_verbose>2);

    #%%%%;
    #% Since n_w_ is uniform, we can perform the rotations all at once. ;
    #%%%%;
    tmp_q_ = torch.concatenate((torch.arange(0,+n_w_max/2-1+1),torch.arange(-n_w_max/2,-1+1)),0).to(dtype=torch.int32);
    N_k_p_wkM__ = torch.zeros(mtr((n_w_sum,n_M))).to(dtype=torch.complex64);
    N_k_p_wkM__ = CTF_k_p_wkC__[index_nCTF_from_nM_,:] * torch.reshape( torch.fft.ifft( torch.fft.fft( torch.reshape(S_k_p_wkS__[index_nS_from_nM_,:],mtr((n_w_max,n_k_p_r,n_M))) , dim=2-0) * torch.reshape( torch.exp(-i*torch.reshape(tmp_q_,mtr((n_w_max,1)))*torch.reshape(euler_gamma_z_M_,mtr((1,n_M)))) , mtr((n_w_max,1,n_M)) ) , dim=2-0) , mtr((n_w_sum,n_M)) ).to(dtype=torch.complex64) ;

    #%%%%;
    #% alternatively, we can generate the M_k_p_wk_ individually. ;
    #%%%%;
    M_k_p_wkM__ = torch.zeros(mtr((n_w_sum,n_M))).to(dtype=torch.complex64);
    for nM in range(n_M):
        if (flag_verbose>0):
            if (np.mod(nM,128)==0): print(f' nM {nM}/{n_M}');
        nCTF = index_nCTF_from_nM_[nM].item();
        tmp_i8_index_rhs_ = matlab_index_2d_0(n_w_sum,':',n_CTF,nCTF);
        CTF_k_p_wk_ = CTF_k_p_wkC__.ravel()[tmp_i8_index_rhs_];
        nS = index_nS_from_nM_[nM].item();
        gamma_z = euler_gamma_z_M_[nM].item();
        tmp_i8_index_rhs_ = matlab_index_2d_0(n_w_sum,':',n_S,nS);
        S_k_p_wk_ = S_k_p_wkS__.ravel()[tmp_i8_index_rhs_];
        M_k_p_wk_ = torch.zeros(n_w_sum).to(dtype=torch.complex64);
        M_k_p_wk_ = CTF_k_p_wk_ * torch.tensor(rotate_p_to_p_fftw_using_numpy(n_k_p_r,n_w_.numpy(),n_w_sum,S_k_p_wk_.numpy(),+gamma_z));
        M_k_p_wkM__[nM,:] = M_k_p_wk_;
    #end;%for nM=0:n_M-1;
    #%%%%;
    fnorm_disp(flag_verbose,'N_k_p_wkM__',N_k_p_wkM__,'M_k_p_wkM__',M_k_p_wkM__,' %%<-- should be zero');
        
    if (flag_verbose>2):
        np.set_printoptions(threshold=10);
        print('M_k_p_wkM__');print(M_k_p_wkM__);
    #end;%if (flag_verbose>2);

    CTF_k_p_wkM__ = CTF_k_p_wkC__[index_nCTF_from_nM_,:];
    M0C2_k_p_wkM__ = torch.abs(CTF_k_p_wkM__)**2;
    M1C1_k_p_wkM__ = M_k_p_wkM__*CTF_k_p_wkM__;
    M0C2_k_p_wMk__ = torch.reshape(torch.permute(torch.reshape(M0C2_k_p_wkM__,mtr((n_w_max,n_k_p_r,n_M))),mtr(mts((0,2,1)))),mtr((n_w_max*n_M,n_k_p_r)));
    M1C1_k_p_wMk__ = torch.reshape(torch.permute(torch.reshape(M1C1_k_p_wkM__,mtr((n_w_max,n_k_p_r,n_M))),mtr(mts((0,2,1)))),mtr((n_w_max*n_M,n_k_p_r)));

    if (flag_verbose>2):
        np.set_printoptions(threshold=10);
        print('CTF_k_p_wkM__');print(CTF_k_p_wkM__);
        tmp_index_ = (1e6*np.arange(10)).astype(int);
        print('M0C2_k_p_wkM__[tmp_index_]');print(M0C2_k_p_wkM__.flatten()[tmp_index_]);
        print('M0C2_k_p_wMk__[tmp_index_]');print(M0C2_k_p_wMk__.flatten()[tmp_index_]);
        print('M1C1_k_p_wkM__[tmp_index_]');print(M1C1_k_p_wkM__.flatten()[tmp_index_]);
        print('M1C1_k_p_wMk__[tmp_index_]');print(M1C1_k_p_wMk__.flatten()[tmp_index_]);
    #end;%if (flag_verbose>2);

    n_3 = 3;
    (
        k_p_polar_a_wM__,
        k_p_azimu_b_wM__,
        k_c_0_wM__,
        k_c_1_wM__,
        k_c_2_wM__,
        k_p_r01_wM__,
    )= cg_rhs_2(
        n_M,
        n_w_max,
        euler_polar_a_M_,
        euler_azimu_b_M_,
        euler_gamma_z_M_,
    );

    if (flag_verbose>2):
        np.set_printoptions(threshold=10);
        print('k_p_polar_a_wM__');print(k_p_polar_a_wM__);
        print('k_p_azimu_b_wM__');print(k_p_azimu_b_wM__);
        print('k_c_0_wM__');print(k_c_0_wM__);
        print('k_c_1_wM__');print(k_c_1_wM__);
        print('k_c_2_wM__');print(k_c_2_wM__);
        print('k_p_r01_wM__');print(k_p_r01_wM__);
    #end;%if (flag_verbose>2);

    polar_a_a_ = polar_a_ka__[0].ravel();
    n_polar_a = numel(polar_a_a_);
    assert(np.std(np.diff(polar_a_a_.numpy()),ddof=1)<1e-3);
    dpolar_a = np.mean(np.diff(polar_a_a_.numpy())) #<-- will be negative. ;
    n_azimu_b_a_ = n_azimu_b_ka__[0].ravel();
    dazimu_b_a_ = (2*pi) / torch.maximum(torch.tensor([1]),n_azimu_b_a_);
    n_azimu_b_csum_ = cumsum_0(n_azimu_b_a_);
    assert(n_outer_shell==int(n_azimu_b_csum_[n_polar_a].item()));

    #%%%%%%%%;
    #% Here we switch to numpy to use the csr_matrix functions. ;
    #%%%%%%%%;
    npolar_a_from_data_wM_ = np.reshape(
        np.clip(
            np.round((pi - k_p_polar_a_wM__.numpy()) / max(1e-12,-dpolar_a)), #<-- the banker-round should not impact the results much. ;
            0,
            n_polar_a - 1,
        ),
        (n_w_max*n_M),
    );
        
    nazimu_b_from_data_wMa__ = np.zeros((n_polar_a,n_w_max*n_M));
    for npolar_a in range(n_polar_a):
        n_azimu_b = n_azimu_b_a_[npolar_a].item();
        dazimu_b = dazimu_b_a_[npolar_a].item();
        nazimu_b_from_data_wM_ = np.reshape(
            np.mod(
                np.round(k_p_azimu_b_wM__.numpy() / max(1e-12,dazimu_b)), #<-- the banker-round should not impact the results much. ;
                n_azimu_b,
            ),
            (n_w_max*n_M),
        );
        nazimu_b_from_data_wMa__[npolar_a,:] = nazimu_b_from_data_wM_.flatten();
    #end;%for npolar_a=0:n_polar_a-1;

    index_babin_shell_from_data_wM_ = np.zeros((n_w_max*n_M),dtype=int);
    for npolar_a in range(n_polar_a):
        tmp_index_ = np.where(npolar_a_from_data_wM_.flatten() == npolar_a)[0];
        index_babin_shell_from_data_wM_[tmp_index_] = ( n_azimu_b_csum_.numpy()[npolar_a] + nazimu_b_from_data_wMa__[npolar_a,tmp_index_] );
    #end;%for npolar_a=0:n_polar_a-1;

    babin_shell_from_data_qwM__ = csr_matrix(
        (
            np.ones(n_w_max*n_M),
            (np.arange(n_w_max*n_M),index_babin_shell_from_data_wM_),
        ),
        shape=(n_w_max*n_M,n_outer_shell),
    );

    if (flag_verbose>2):
        np.set_printoptions(threshold=10);
        print('polar_a_a_');print(polar_a_a_);
        print('n_polar_a');print(n_polar_a);
        print('dpolar_a');print(dpolar_a);
        print('n_azimu_b_a_');print(n_azimu_b_a_);
        print('dazimu_b_a_');print(dazimu_b_a_);
        print('n_azimu_b_csum_');print(n_azimu_b_csum_);
        print('npolar_a_from_data_wM_');print(npolar_a_from_data_wM_);
        print('nazimu_b_from_data_wMa__');print(nazimu_b_from_data_wMa__);
        print('index_babin_shell_from_data_wM_');print(index_babin_shell_from_data_wM_);
        print('babin_shell_from_data_qwM__');print(babin_shell_from_data_qwM__);
    #end;%if (flag_verbose>2);

    a_k_p_babin_ = np.reshape((babin_shell_from_data_qwM__.T.dot(M1C1_k_p_wMk__.numpy().T).T)/np.maximum(1e-12,babin_shell_from_data_qwM__.T.dot(M0C2_k_p_wMk__.numpy().T).T),n_qk); #<-- all these non-conjugate transposes so that this matches the matlab code in line 307 of test_CryoLike_rbp_20250730.m. ;

    fnorm_disp(flag_verbose,'a_k_p_form_',a_k_p_form_,'a_k_p_babin_',torch.tensor(a_k_p_babin_).to(dtype=torch.complex64),' %<-- total error can be 1e-1 or so');
        
    if (flag_verbose> 0): print(f" %% [finished {str_thisfunction}]");

if __name__ == '__main__':
    print('running test_rbp_0');
    test_rbp_0();
    print('returning');



