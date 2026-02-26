#import os; os.chdir('/data/rangan/dir_cryoem/dir_rangan_python');
import numpy as np ;
from . disp_sprintf import disp ; from . disp_sprintf import sprintf ;

def parameter_timing_update(
        parameter=None,
        str_field=None,
        dt=None,
        ncall=None,
        nop=None,
):
    if dt is None: dt=0.0;
    if ncall is None: ncall=int(0);
    if nop is None: nop=int(0);
    if parameter is None: parameter = {'type': 'parameter'};
    if 'timing' not in parameter or parameter['timing'] is None:
        parameter['timing'] = [
            ['label', 'total time', 'number of calls', 'number of ops', 'MHz', 'GHz']
        ];
    #end;%if ;
    timing = parameter['timing'];
    tmp_index = None;
    for idx in range(1, len(timing)):
        if timing[idx][0] == str_field:
            tmp_index = idx ;
            break ;
        #end;%if;
    #end;%for;

    if tmp_index is not None:
        timing[tmp_index][1] = timing[tmp_index][1] + dt ;
        timing[tmp_index][2] = timing[tmp_index][2] + ncall ;
        timing[tmp_index][3] = timing[tmp_index][3] + nop ;
    else:
        timing.append([str_field, dt, ncall, nop, 0.0, 0.0]) ;
        tmp_index = len(timing) - 1 ;
    #end;%if;

    n_op = timing[tmp_index][3] ;
    d_t = timing[tmp_index][1] ;
    if (n_op > 0) and (d_t > 0):
        timing[tmp_index][4] = (n_op / np.maximum(1e-12,d_t)) / 1e6 ; # MHz
        timing[tmp_index][5] = (n_op / np.maximum(1e-12,d_t)) / 1e9 ; # GHz
    #end;%if;

    parameter['timing'] = timing;
    return(parameter);


