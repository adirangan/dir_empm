# M-x reb-change-syntax --> string ;
# M-x query-replace-regexp ;
# (\([^(),]*\),\([^(),]*\),\([^(),]*\)) ;
# [\3,\2,\1] ;

from dir_empm.dir_matlab_macros import * ;
from dir_empm.sample_sphere_7 import sample_sphere_7 ;
from dir_empm.get_Ylm__2 import get_Ylm__2 ;
from dir_empm.get_Ylm_condense_wrap_0 import get_Ylm_condense_wrap_0 ;

flag_verbose=1;
print(' %% testing get_Ylm_condense_wrap_0');
dir_base = '/data/rangan' ;
dir_ascii = dir_base + '/dir_cryoem/dir_rangan_python/dir_ascii' ;
#for nk_int in range(3,8+1):
for nk_int in range(3,3+1):
    k_int = int(2**nk_int);
#    for nk_eq_d_double in range(2):
    for nk_eq_d_double in range(1):
        if nk_eq_d_double==0: k_eq_d_double=1.0;
        if nk_eq_d_double==1: k_eq_d_double=0.5;
        k_p_r_max = k_int/(2*pi); k_eq_d = k_eq_d_double/(2*pi); str_L = 'L';
        flag_uniform_over_n_k_p_r = 1;
        flag_uniform_over_polar_a = 0; #<-- This is set to match test_ssnll_from_a_k_p_14 ;
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
            _,
            _,
            _,
            _,
            n_polar_a_k_,
            polar_a_ka__,
            n_azimu_b_ka__,
        ) = sample_sphere_7(
            0*flag_verbose,
            k_p_r_max,
            k_eq_d,
            str_L,
            flag_uniform_over_n_k_p_r,
            flag_uniform_over_polar_a,
        ) ;
        #%%%%;
        l_max_upb = matlab_scalar_round(2*pi*k_p_r_max); #<-- typically sufficient for 2-3 digits of precision. ;
        l_max_ = torch.zeros(n_k_p_r);
        for nk_p_r in range(n_k_p_r):
            l_max_[nk_p_r] = int(np.maximum(0,np.minimum(l_max_upb,1+np.ceil(2*pi*k_p_r_[nk_p_r].item()))));
        #end;%for nk_p_r=0:n_k_p_r-1;
        (
            Ylm_uklma___,
            k_p_azimu_b_sub_uka__,
            k_p_polar_a_sub_uka__,
            l_max_uk_,
            index_nu_n_k_per_shell_from_nk_p_r_,
            index_k_per_shell_uka__,
        ) = get_Ylm_condense_wrap_0(
            0*flag_verbose,
            n_qk,
            n_qk_csum_,
            k_p_azimu_b_qk_,
            k_p_polar_a_qk_,
            n_k_p_r,
            l_max_,
        );
        n_u = len(Ylm_uklma___);
        assert n_u==1;
        if (flag_verbose>0): print(f' %% l_max_uk_[0] {l_max_uk_[0].item()}');
        fname_ascii = dir_ascii + '/Ylm_0lma__.ascii' ;
        print(f" %% writing {fname_ascii}");
        np.savetxt(fname_ascii,Ylm_uklma___[0].numpy().ravel());
        fname_ascii = dir_ascii + '/k_p_azimu_b_sub_0a_.ascii' ;
        print(f" %% writing {fname_ascii}");
        np.savetxt(fname_ascii,k_p_azimu_b_sub_uka__[0].numpy().ravel());
        fname_ascii = dir_ascii + '/k_p_polar_a_sub_0a_.ascii' ;
        print(f" %% writing {fname_ascii}");
        np.savetxt(fname_ascii,k_p_polar_a_sub_uka__[0].numpy().ravel());
        fname_ascii = dir_ascii + '/index_nu_n_k_per_shell_from_nk_p_r_.ascii' ;
        print(f" %% writing {fname_ascii}");
        np.savetxt(fname_ascii,index_nu_n_k_per_shell_from_nk_p_r_.numpy().ravel());
        fname_ascii = dir_ascii + '/index_k_per_shell_0a_.ascii' ;
        print(f" %% writing {fname_ascii}");
        np.savetxt(fname_ascii,index_k_per_shell_uka__[0].numpy().ravel());
        n_uklma = numel(Ylm_uklma___[0]);
        print(f' %% k_int {k_int:02d} --> n_k_p_r {n_k_p_r:03d} l_max_upb {l_max_upb:03d} n_u {n_u:02d} n_uklma {n_uklma:06d} --> Ylm_uklma___: {(n_uklma*16/1e9):0.6f}GB')
        #%%%%;
    #end;%for k_int = [ 8,16,32,48,64];
#end;%for k_eq_d_double = [1.0,0.5];
#%%%%;
print(' %% returning');

r'''
dir_base = '/data/rangan' ;
dir_ascii = dir_base + '/dir_cryoem/dir_rangan_python/dir_ascii' ;
for nl in range(n_l):
    l_val = l_val_[nl].item();
    fname_ascii = dir_ascii + '/d0Y_mid_' + str(l_val) + '_mj__.ascii' ;
    print(f" %% writing {fname_ascii}");
    np.savetxt(fname_ascii,d0Y_mid_lmj___[nl].numpy().ravel());
for nl in range(n_l):
    l_val = l_val_[nl].item();
    fname_ascii = dir_ascii + '/d1Y_mid_' + str(l_val) + '_mj__.ascii' ;
    print(f" %% writing {fname_ascii}");
    np.savetxt(fname_ascii,d1Y_mid_lmj___[nl].numpy().ravel());
for nl in range(n_l):
    l_val = l_val_[nl].item();
    fname_ascii = dir_ascii + '/d2Y_mid_' + str(l_val) + '_mj__.ascii' ;
    print(f" %% writing {fname_ascii}");
    np.savetxt(fname_ascii,d2Y_mid_lmj___[nl].numpy().ravel());
'''
