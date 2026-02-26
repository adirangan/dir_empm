from dir_empm.dir_matlab_macros import * ;

def tfpmhh_pm_wUX_0(
        parameter=None,
        n_k_p_r=None,
        pm_n_k_p_r=None,
        pm_wUX_kn__=None,
        n_w_max=None,
        n_M=None,
        M_k_p_wkM__=None,
):
    str_thisfunction = 'tfpmhh_pm_wUX_0';

    if not isinstance(parameter, dict): parameter = {'type': 'parameter'};
    if 'flag_verbose' not in parameter: parameter['flag_verbose'] = 0 ;
    flag_verbose = parameter['flag_verbose'];

    if flag_verbose > 0: print(f' %% [entering {str_thisfunction}]');

    pm_n_w_max = n_w_max; pm_n_w_sum = pm_n_w_max*pm_n_k_p_r;
    str_einsum = msr('kn') + ',' + msr('wkM') + '->' + msr('wnM') ;
    UX_M_k_p_wnM__ = torch.reshape(torch.einsum( str_einsum , torch.reshape(pm_wUX_kn__.to(dtype=torch.complex64),mtr((n_k_p_r,pm_n_k_p_r))) , torch.reshape(M_k_p_wkM__.to(dtype=torch.complex64),mtr((n_w_max,n_k_p_r,n_M))) ),mtr((pm_n_w_sum,n_M)));

    if flag_verbose > 0: print(f' %% [finished {str_thisfunction}]');

    return(
        parameter,
        UX_M_k_p_wnM__,
    );
