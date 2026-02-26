def function_template(parameter=None):
    str_thisfunction = 'function_template';

    if not isinstance(parameter, dict): parameter = {'type': 'parameter'};
    if 'flag_verbose' not in parameter: parameter['flag_verbose'] = 0 ;
    flag_verbose = parameter['flag_verbose'];

    if flag_verbose > 0: print(f' %% [entering {str_thisfunction}]');

    #%%%%;

    if flag_verbose > 0: print(f' %% [finished {str_thisfunction}]');
