from dir_empm.dir_matlab_macros import * ;
from dir_empm.sample_shell_6 import sample_shell_6
from dir_empm.h2d import h2d ; from dir_empm.h3d import h3d ;
from dir_empm.sample_sphere_7 import sample_sphere_7

def test_weight_3d_k_p_qk_from_sample_sphere_7(
        k_int: int = 48, #<-- highest frequency (2*pi*k_p_r_max), watch out for aliasing! 
        k_eq_d_double: np.float64 = 1.0, #<-- prefactor for k_eq_d, determines density of sampling in frequency-space. 
        flag_uniform_over_n_k_p_r: int= 0, #<-- SamplingStrategy = SamplingStrategy.ADAPTIVE,
        flag_uniform_over_polar_a: int= 0, #<-- SamplingStrategy = SamplingStrategy.ADAPTIVE,
        n_frequency: int = 1+3*8, #<-- 8 plane-waves tested per unit-interval in k-space.
        frequency_max: np.float64 = 3.0, #<-- test delta_T in [0,3].
):
    '''
    This function tests the spherical-quadrature-grid in 3-dimensions.
    '''   
    ####
    flag_verbose = 2 ;
    if flag_verbose: print('flag_verbose: ',flag_verbose) ;
    ####
    k_p_r_max = k_int/(2.0*pi); k_eq_d = k_eq_d_double/(2*pi); str_T_vs_L = 'L';
    ####
    # define quadrature grid. 
    ####
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
    ) = sample_sphere_7(
        0*flag_verbose,
        k_p_r_max,
        k_eq_d,
        str_T_vs_L,
        flag_uniform_over_n_k_p_r,
        flag_uniform_over_polar_a,
    )[:13] ;
    if flag_verbose>1: print('2*pi*k_p_r_max',2*pi*k_p_r_max,'k_p_r_max',k_p_r_max) ;
    if flag_verbose>1: print('n_k_p_r ',n_k_p_r) ;
    if flag_verbose>1: print('k_p_r_.shape',k_p_r_.shape) ;
    if flag_verbose>2: print('weight_3d_k_p_r_.shape',weight_3d_k_p_r_.shape) ;
    if flag_verbose>1: print('n_qk ',n_qk) ;
    ####
    # Here we just check the basic integral for various plane-waves of varying frequency: 
    ####
    frequency_ = torch.linspace(0, frequency_max, n_frequency).to(dtype=torch.float32) ;
    I_T_errrel_max = 0.0
    for nfrequency in range(n_frequency):
        frequency = frequency_[nfrequency].item();
        delta_T = frequency ;
        tmp_polar_a = 1 * pi * np.random.rand() ;
        tmp_azimu_b = 2 * pi * np.random.rand() ;
        delta_T_0 = delta_T * np.cos(tmp_azimu_b) * np.sin(tmp_polar_a) ;
        delta_T_1 = delta_T * np.sin(tmp_azimu_b) * np.sin(tmp_polar_a) ;
        delta_T_2 = delta_T * np.cos(tmp_polar_a) ;
        T_k_p_qk_ = torch.exp( 2*pi*i* (k_c_0_qk_*delta_T_0 + k_c_1_qk_*delta_T_1 + k_c_2_qk_*delta_T_2) ) ;
        tmp_T_kd = 2*pi*k_p_r_max*delta_T ;
        I_T_quad = torch.sum(T_k_p_qk_ * weight_3d_k_p_qk_).item() ;
        if np.abs(tmp_T_kd)< 1e-6: I_T_form = (4*pi/3) * k_p_r_max**3 ;
        if np.abs(tmp_T_kd)>=1e-6: I_T_form = h3d(tmp_T_kd).item() * k_p_r_max**3 ;
        I_T_errrel = np.abs(I_T_form - I_T_quad)/max(1e-12,np.abs(I_T_form)) ;
        if flag_verbose>1: print(' %%',' nfrequency ',nfrequency,'/',n_frequency,' frequency ',frequency) ;
        if flag_verbose>1: print(' %%',' I_T_form: ',I_T_form,' I_T_quad: ',I_T_quad,' I_T_errrel: ',I_T_errrel) ;
        I_T_errrel_max = max(I_T_errrel_max,I_T_errrel) ;
    #end;%for;
    if flag_verbose>1: print(' %%',' I_T_errrel_max: ',I_T_errrel_max) ;
    assert(np.abs(I_T_errrel_max) < 1e-2) ;

if __name__ == '__main__':
    print('running test_weight_3d_k_p_qk_from_sample_sphere_7') ;
    test_weight_3d_k_p_qk_from_sample_sphere_7() ;
    print('returning') ;
