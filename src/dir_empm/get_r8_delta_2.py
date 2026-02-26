from dir_empm.dir_matlab_macros import * ;

def get_r8_delta_2(
        r8_delta_r_max=None,
        n_delta_v_0in=None,
):
    flag_verbose=0; tolerance_margin = 1e-6;
    str_thisfunction = 'get_r8_delta_2'

    if flag_verbose > 0: print(f' %% [entering {str_thisfunction}]');

    if r8_delta_r_max is None: r8_delta_r_max = 0.0;
    if n_delta_v_0in is None: n_delta_v_0in = int(0);

    if n_delta_v_0in<=1:
        n_delta_v_out = 1; r8_delta_x_ = torch.tensor(0.0).to(dtype=torch.float64); r8_delta_y_ = torch.tensor(0.0).to(dtype=torch.float64);
    #end;%if n_delta_v_0in<=1;

    if n_delta_v_0in> 1:
        n_x = int(1 + np.floor(np.sqrt(n_delta_v_0in))); continue_flag=1;
        while continue_flag:
            r8_x_ = torch.linspace(-r8_delta_r_max,+r8_delta_r_max,n_x).to(dtype=torch.float64);
            r8_y_ = torch.linspace(-r8_delta_r_max,+r8_delta_r_max,n_x).to(dtype=torch.float64);
            [r8_Y__,r8_X__] = torch.meshgrid(r8_y_,r8_x_,indexing='ij'); #%<-- reversed to match matlab. ;
            R__ = torch.sqrt(r8_X__**2 + r8_Y__**2);
            tmp_index_ = efind(R__.ravel()<=r8_delta_r_max+tolerance_margin);
            if numel(tmp_index_)>=n_delta_v_0in: continue_flag=0;
            if numel(tmp_index_)< n_delta_v_0in:
                n_x = n_x + 1; continue_flag=1;
            #end;%if;
        #end;%while;
        n_delta_v_out = numel(tmp_index_);
        r8_delta_x_ = r8_X__.ravel()[tmp_index_];
        r8_delta_y_ = r8_Y__.ravel()[tmp_index_];
        r8_0 = torch.zeros(1).to(dtype=torch.float64);
        if (np.mod(n_x,2)==0): n_delta_v_out = n_delta_v_out + 1;
        if (np.mod(n_x,2)==0): r8_delta_x_ = torch.concatenate((r8_0,r8_delta_x_.ravel()),0).to(dtype=torch.float64);
        if (np.mod(n_x,2)==0): r8_delta_y_ = torch.concatenate((r8_0,r8_delta_y_.ravel()),0).to(dtype=torch.float64);
    #end;%if n_delta_v_0in> 1;

    if flag_verbose > 0: print(f' %% [finished {str_thisfunction}]');
    return(
        n_delta_v_out,
        r8_delta_x_,
        r8_delta_y_,
    );
