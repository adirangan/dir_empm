#import os; os.chdir('/data/rangan/dir_cryoem/dir_rangan_python');
import numpy as np ;
from . disp_sprintf import disp ; from . disp_sprintf import sprintf ;

def parameter_timing_printf(
        parameter=None,
):
    if parameter is None: parameter = {'type': 'parameter'};
    if 'timing' not in parameter or parameter['timing'] is None:
        parameter['timing'] = [
            ['label', 'total time', 'number of calls', 'number of ops', 'MHz', 'GHz']
        ];
    #end;%if ;
    timing = parameter['timing'];
    n_label = len(timing)-1;
    t_tot_ = np.zeros(n_label);
    for nlabel in range(n_label):
        nt=1; t_tot = timing[1+nlabel][nt];
        t_tot_[nlabel] = t_tot;
    #end;%for nlabel in range(n_label): ;
    index_srt_ = np.flip(np.argsort(t_tot_),axis=0);
    disp(sprintf(' %% %16.16ss %16.16s %16.16s %s','t_tot','n_cal','n_ops','label'));
    for nlabel in range(n_label):
        nt=0;
        label = timing[1+index_srt_[nlabel]][nt]; nt=nt+1;
        t_tot = timing[1+index_srt_[nlabel]][nt]; nt=nt+1;
        n_cal = timing[1+index_srt_[nlabel]][nt]; nt=nt+1;
        n_ops = timing[1+index_srt_[nlabel]][nt]; nt=nt+1;
        MHz_ = timing[1+index_srt_[nlabel]][nt]; nt=nt+1;
        GHz_ = timing[1+index_srt_[nlabel]][nt]; nt=nt+1;
        disp(sprintf(' %% %+16.6fs %+16.4d %+16.4d %s',t_tot,n_cal,n_ops,label));
    #end;%for nlabel in range(n_label): ;


