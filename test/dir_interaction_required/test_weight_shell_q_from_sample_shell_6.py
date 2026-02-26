from dir_empm.dir_matlab_macros import * ;
from dir_empm.hshell import hshell ;
from dir_empm.sample_shell_6 import sample_shell_6 ;

def test_weight_shell_q_from_sample_shell_6(
        k_int: int = 48, #<-- highest frequency (2*pi*k_p_r_max), watch out for aliasing! 
        k_eq_d_double: np.float64 = 1.0, #<-- prefactor for k_eq_d, determines density of sampling in frequency-space. 
        flag_uniform_over_polar_a: int = 0, #<-- SamplingStrategy = SamplingStrategy.ADAPTIVE,
        n_frequency: int = 1+3*8, #<-- 8 plane-waves tested per unit-interval in k-space.
        frequency_max: np.float64 = 3.0, #<-- test delta_T in [0,3].
):
    '''
    This function tests the spherical-quadrature-grid on a single shell in 3-dimensions.
    '''   
    ####
    flag_verbose = 2 ;
    if flag_verbose: print('flag_verbose: ',flag_verbose) ;
    ####
    k_p_r_max = k_int/(2.0*pi); k_eq_d = k_eq_d_double/(2*pi) ;
    ####
    # define quadrature grid. 
    ####
    (
        n_q,
        azimu_b_q_,
        polar_a_q_,
        weight_shell_q_,
        k_c_0_q_,
        k_c_1_q_,
        k_c_2_q_,
    ) = sample_shell_6(
        k_p_r_max = k_p_r_max,
        k_eq_d = k_eq_d,
        str_T_vs_L = 'L',
        flag_uniform_over_polar_a = flag_uniform_over_polar_a,
    )[:7] ;
    if flag_verbose>1: print('2*pi*k_p_r_max',2*pi*k_p_r_max,'k_p_r_max',k_p_r_max) ;
    if flag_verbose>1: print('n_q ',n_q) ;
    if flag_verbose>2: print('polar_a_q_.shape',polar_a_q_.shape) ;
    if flag_verbose>2: print('azimu_b_q_.shape',azimu_b_q_.shape) ;
    if flag_verbose>2: print('weight_shell_q_.shape',weight_shell_q_.shape) ;
    ####
    # Here we just check the basic integral for various plane-waves of varying frequency: 
    ####
    frequency_ = torch.linspace(0, frequency_max, n_frequency).to(dtype=torch.float32) ;
    I_T_errrel_max = 0.0 ;
    for nfrequency in range(n_frequency):
        frequency = frequency_[nfrequency].item() ;
        delta_T = frequency ;
        tmp_polar_a = 1 * pi * np.random.rand() ;
        tmp_azimu_b = 2 * pi * np.random.rand() ;
        delta_T_0 = delta_T * np.cos(tmp_azimu_b) * np.sin(tmp_polar_a) ;
        delta_T_1 = delta_T * np.sin(tmp_azimu_b) * np.sin(tmp_polar_a) ;
        delta_T_2 = delta_T * np.cos(tmp_polar_a) ;
        T_k_p_q_ = torch.exp(2 * pi * i * (k_c_0_q_ * delta_T_0 + k_c_1_q_ * delta_T_1 + k_c_2_q_ * delta_T_2)) ;
        tmp_T_kd = 2*pi*k_p_r_max*delta_T ;
        I_T_quad = torch.sum(T_k_p_q_ * weight_shell_q_).item() ;
        if np.abs(tmp_T_kd)< 1e-6: I_T_form = (4*pi) * k_p_r_max**2 ;
        if np.abs(tmp_T_kd)>=1e-6: I_T_form = hshell(tmp_T_kd).item() * k_p_r_max**2 ;
        I_T_errrel = np.abs(I_T_form - I_T_quad)/np.maximum(1e-12,np.abs(I_T_form)) ;
        if flag_verbose>1: print(' %%',' nfrequency ',nfrequency,'/',n_frequency,' frequency ',frequency) ;
        if flag_verbose>1: print(' %%',' I_T_form: ',I_T_form,' I_T_quad: ',I_T_quad,' I_T_errrel: ',I_T_errrel) ;
        I_T_errrel_max = max(I_T_errrel_max,I_T_errrel) ;
    #end;%for ;
    if flag_verbose>1: print(' %%',' I_T_errrel_max: ',I_T_errrel_max) ;
    assert(np.abs(I_T_errrel_max) < 1e-2) ;

if __name__ == '__main__':
    print('running test_weight_shell_q_from_sample_shell_6') ;
    test_weight_shell_q_from_sample_shell_6() ;
    print('returning') ;
