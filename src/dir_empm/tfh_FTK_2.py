from dir_empm.dir_matlab_macros import * ;
from dir_empm.gen_Jsvd_FTK_8 import gen_Jsvd_FTK_8 ;
from dir_empm.get_r8_delta_2 import get_r8_delta_2 ;
from dir_empm.get_r8_svd_chebval_U_d_0 import get_r8_svd_chebval_U_d_0 ;
from dir_empm.get_r8_svd_chebval_V_r_0 import get_r8_svd_chebval_V_r_0 ;

def tfh_FTK_2(
        n_k_p_r=None,
        r8_k_p_r_=None,
        r8_k_p_r_max=None,
        r8_delta_r_max=None,
        r8_svd_eps=None,
        n_delta_v_requested=None,
        r8_delta_x_0in_=None,
        r8_delta_y_0in_=None,
        l_max=None,
        n_a_degree=None,
        n_b_degree=None,
):
    flag_verbose=0;
    str_thisfunction = 'tfh_FTK_2' ;

    #%%%%%%%%;
    if (flag_verbose): print(f' %% [entering {str_thisfunction}]');
    #%%%%%%%%;

    if r8_delta_r_max is None: r8_delta_r_max = 0.0;
    if r8_svd_eps is None: r8_svd_eps = 1e-4;
    if l_max is None: l_max = int(24);
    if n_a_degree is None: n_a_degree = int(64);
    if n_b_degree is None: n_b_degree = int(65);
    r8_k_p_r_ = r8_k_p_r_.to(dtype=torch.float64);
    r8_k_p_r_max = float(r8_k_p_r_max);
    r8_delta_r_max = float(r8_delta_r_max);
    r8_svd_eps = float(r8_svd_eps);
    
    pm_N_pixel = r8_delta_r_max * (2*pi*r8_k_p_r_max) / (pi*np.sqrt(2)) ; 
    tmp_t=tic();
    FTK = gen_Jsvd_FTK_8(r8_k_p_r_max,pm_N_pixel,r8_svd_eps,l_max,n_a_degree,n_b_degree);
    tmp_t=toc(tmp_t);
    if (flag_verbose>0): disp(sprintf(' %% gen_Jsvd_FTK_8: %0.6fs',tmp_t)); #end;
    #%%%%;
    if n_delta_v_requested is None: n_delta_v_requested = 2*FTK['n_svd_l'];
    if (n_delta_v_requested<=0): n_delta_v_requested = 2*FTK['n_svd_l'];
    #%%%%;
    if r8_delta_r_max==0:
        tmp_t=tic();
        FTK['n_delta_v'],FTK['r8_delta_x_'],FTK['r8_delta_y_'] = get_r8_delta_2(r8_delta_r_max,1);
        tmp_t=toc(tmp_t);
        if (flag_verbose>0): disp(sprintf(' %% get_r8_delta_2: %0.6fs',tmp_t)); #end;
    #end;%if r8_delta_r_max==0;
    #%%%%;
    if r8_delta_r_max> 0:
        if (r8_delta_x_0in_ is not None) and (r8_delta_y_0in_ is not None):
            r8_delta_x_0in_ = r8_delta_x_0in_.to(dtype=torch.float64);
            r8_delta_y_0in_ = r8_delta_y_0in_.to(dtype=torch.float64);
            assert(numel(r8_delta_x_0in_)==n_delta_v_requested);
            assert(numel(r8_delta_y_0in_)==n_delta_v_requested);
            FTK['n_delta_v'] = n_delta_v_requested;
            FTK['r8_delta_x_'] = r8_delta_x_0in_.ravel();
            FTK['r8_delta_y_'] = r8_delta_y_0in_.ravel();
        #end;%if ~isempty(r8_delta_x_0in_) & ~isempty(r8_delta_y_0in_);
        if (r8_delta_x_0in_ is     None)  or (r8_delta_y_0in_ is     None):
            tmp_t=tic();
            FTK['n_delta_v'],FTK['r8_delta_x_'],FTK['r8_delta_y_'] = get_r8_delta_2(r8_delta_r_max,n_delta_v_requested);
            tmp_t=toc(tmp_t);
            if (flag_verbose>0): disp(sprintf(' %% get_r8_delta_2: %0.6fs',tmp_t)); #end;
        #end;%if  isempty(r8_delta_x_0in_) |  isempty(r8_delta_y_0in_);
    #end;%if r8_delta_r_max> 0;
    #%%%%;

    FTK['r8_svd_d_max'] = r8_delta_r_max;
    tmp_t=tic();
    FTK['r8_svd_chebval_U_d_'] = get_r8_svd_chebval_U_d_0(FTK['r8_svd_d_max'],FTK['n_svd_d'],FTK['r8_svd_d_'],FTK['n_svd_l'],FTK['i4_svd_l_'],FTK['r8_svd_U_d_chebcoef_'],FTK['n_delta_v'],FTK['r8_delta_x_'],FTK['r8_delta_y_']).ravel();
    tmp_t=toc(tmp_t);
    if (flag_verbose>0): disp(sprintf(' %% get_r8_svd_chebval_U_d_0: %0.6fs',tmp_t)); #end;
    assert(numel(FTK['r8_svd_chebval_U_d_'])==FTK['n_svd_l']*FTK['n_delta_v']);
    FTK['r8_svd_r_max'] = 2*pi*r8_k_p_r_max;
    tmp_t=tic();
    FTK['r8_svd_chebval_V_r_'] = get_r8_svd_chebval_V_r_0(FTK['r8_svd_r_max'],FTK['n_svd_r'],FTK['r8_svd_r_'],FTK['n_svd_l'],FTK['i4_svd_l_'],FTK['r8_svd_V_r_chebcoef_'],n_k_p_r,2*pi*r8_k_p_r_).ravel();
    tmp_t=toc(tmp_t);
    if (flag_verbose>0): disp(sprintf(' %% get_r8_svd_chebval_V_r_0: %0.6fs',tmp_t)); #end;
    assert(numel(FTK['r8_svd_chebval_V_r_'])==FTK['n_svd_l']*n_k_p_r);
    tmp_t=tic();
    FTK['c16_svd_expiw__'] = torch.reshape(torch.exp(-i*(pi/2 - torch.reshape(torch.atan2(FTK['r8_delta_y_'],FTK['r8_delta_x_']),mtr((FTK['n_delta_v'],1))))*torch.reshape(FTK['i4_svd_l_'],mtr((1,FTK['n_svd_l'])))),mtr((FTK['n_delta_v'],FTK['n_svd_l'])));
    tmp_t=toc(tmp_t);
    if (flag_verbose>0): disp(sprintf(' %% c16_svd_expiw__: %0.6fs',tmp_t)); #end;
    tmp_t=tic();
    FTK['c16_svd_U_d_expiw_s__'] = torch.permute(torch.reshape(FTK['r8_svd_chebval_U_d_'],mtr((FTK['n_svd_l'],FTK['n_delta_v']))), mtr(mts((1,0)))) * mmmm( FTK['c16_svd_expiw__'] , torch.diagflat(FTK['r8_svd_s_']).to(dtype=torch.complex128) );
    tmp_t=toc(tmp_t);
    if (flag_verbose>0): disp(sprintf(' %% c16_svd_U_d_expiw_s__: %0.6fs',tmp_t)); #end;

    #%%%%%%%%;
    if (flag_verbose): print(f' %% [finished {str_thisfunction}]');
    #%%%%%%%%;
    return(FTK);

