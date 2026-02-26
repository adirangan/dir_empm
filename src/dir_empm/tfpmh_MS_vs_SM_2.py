from dir_empm.dir_matlab_macros import * ;
from dir_empm.tfpmh_MS_uniform_2 import tfpmh_MS_uniform_2 ;
from dir_empm.tfpmh_SM_uniform_2 import tfpmh_SM_uniform_2 ;
from dir_empm.tfpmh_SM_uniform_3 import tfpmh_SM_uniform_3 ;

def tfpmh_MS_vs_SM_2(
        parameter =None,
        n_w_max=None,
        n_S=None,
        viewing_azimu_b_S_=None,
        viewing_polar_a_S_=None,
        n_M=None,
        X_SM__=None,
        delta_x_SM__=None,
        delta_y_SM__=None,
        gamma_z_SM__=None,
        I_value_SM__=None,
):

    #%%%%%%%%;
    flag_verbose=0;
    str_thisfunction = 'tfpmh_MS_vs_SM_2';
    if (flag_verbose>0): disp(sprintf(' %% [entering %s],str_thisfunction')); #end;
    #%%%%%%%%;

    if isempty(parameter): parameter = {'type':'parameter'}; #end;%if isempty(parameter);
    if isempty(I_value_SM__): I_value_SM__ = 1 + 0*X_SM__; #end;
    #%%%%%%%%;
    if 'flag_MS_vs_SM' not in parameter: parameter['flag_MS_vs_SM'] = 1; #end; %<-- parameter_bookmark. ;
    flag_MS_vs_SM = parameter['flag_MS_vs_SM'];

    #%%%%%%%%%%%%%%%%;
    if flag_MS_vs_SM==1: #%<-- assign images to templates, ensuring a uniform distribution of viewing angles. ;
        tmp_t = tic();
        (
            parameter,
            euler_polar_a_M_,
            euler_azimu_b_M_,
            euler_gamma_z_M_,
            image_delta_x_M_,
            image_delta_y_M_,
            image_I_value_M_,
            image_X_value_M_,
            image_S_index_M_,
        ) = tfpmh_MS_uniform_2(
            parameter,
            n_w_max,
            n_S,
            viewing_azimu_b_S_,
            viewing_polar_a_S_,
            n_M,
            X_SM__,
            delta_x_SM__,
            delta_y_SM__,
            gamma_z_SM__,
            I_value_SM__,
        );
        tmp_t = toc(tmp_t);
        if (flag_verbose>1): disp(sprintf(' %% MS: update euler_polar_a_ euler_azimu_b_ euler_gamma_z_ : %0.3fs',tmp_t)); #end;
    #end;%if flag_MS_vs_SM==1; %<-- assign images to templates, ensuring a uniform distribution of viewing angles. ;
    if flag_MS_vs_SM==0: #%<-- assign templates to images (uses leslie f_rand). ;
        if 'f_rand' not in parameter: parameter['f_rand'] = 0.05; #end; %<-- parameter_bookmark. ;
        f_rand = parameter['f_rand'];
        tmp_t = tic();
        (
            parameter,
            euler_polar_a_M_,
            euler_azimu_b_M_,
            euler_gamma_z_M_,
            image_delta_x_M_,
            image_delta_y_M_,
            image_I_value_M_,
            image_X_value_M_,
            image_S_index_M_,
        ) = tfpmh_SM_uniform_3(
            parameter,
            n_S,
            viewing_azimu_b_S_,
            viewing_polar_a_S_,
            n_M,
            X_SM__,
            delta_x_SM__,
            delta_y_SM__,
            gamma_z_SM__,
            I_value_SM__,
        );
        tmp_t = toc(tmp_t);
        if (flag_verbose>1): disp(sprintf(' %% SM: update euler_polar_a_ euler_azimu_b_ euler_gamma_z_ : %0.3fs',tmp_t)); #end;
    #end;%if flag_MS_vs_SM==0; %<-- assign templates to images (uses leslie f_rand). ;
    #%%%%%%%%%%%%%%%%;
    
    return(
        parameter,
        euler_polar_a_M_,
        euler_azimu_b_M_,
        euler_gamma_z_M_,
        image_delta_x_M_,
        image_delta_y_M_,
        image_I_value_M_,
        image_X_value_M_,
        image_S_index_M_,
    );
