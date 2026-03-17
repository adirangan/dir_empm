from dir_empm.dir_matlab_macros import * ;
from dir_empm.ylgndr_2 import ylgndr_2 ;
from scipy.special import jv

def plane_wave_expansion_2(
        parameter=None,
        n_k_p_r=None,
        k_p_r_=None,
        n_x=None,
        x_3x__=None,
        l_max_=None,
        sqrt_2lp1_=None,
        sqrt_2mp1_=None,
        sqrt_rat0_m_=None,
        sqrt_rat3_lm__=None,
        sqrt_rat4_lm__=None,
):
    
    str_thisfunction = 'plane_wave_expansion_2';
    
    if isempty(parameter): parameter={'type':'parameter'}; #end;
    if 'flag_verbose' not in parameter: parameter['flag_verbose']=0; #end;
    flag_verbose=parameter['flag_verbose'];

    if (flag_verbose>0): disp(sprintf(' %% [entering %s]',str_thisfunction)); #end;

    l_max_max = int(torch.max(l_max_).item());
    n_y_ = (1+l_max_)**2;
    n_y_sum = int(torch.sum(n_y_).item());
    n_y_csum_ = cumsum_0(n_y_);

    tmp_i8_index_rhs_ = matlab_index_2d_0(n_3,0,n_x,':'); x_0_x_ = x_3x__.ravel()[tmp_i8_index_rhs_];
    tmp_i8_index_rhs_ = matlab_index_2d_0(n_3,1,n_x,':'); x_1_x_ = x_3x__.ravel()[tmp_i8_index_rhs_];
    tmp_i8_index_rhs_ = matlab_index_2d_0(n_3,2,n_x,':'); x_2_x_ = x_3x__.ravel()[tmp_i8_index_rhs_];
    x_p_r_x_ = torch.sqrt(x_0_x_**2 + x_1_x_**2 + x_2_x_**2);
    x_p_r01_x_ = torch.sqrt(x_0_x_**2 + x_1_x_**2);
    x_p_azimu_b_x_ = torch.atan2(x_1_x_,x_0_x_);
    x_p_polar_a_x_ = torch.atan2(x_p_r01_x_,x_2_x_); #%<-- assumed to have no special clustered structure. ;

    tmp_t=tic();
    parameter_ylgndr = parameter;
    parameter_ylgndr['flag_verbose'] = 0;
    parameter_ylgndr['flag_d'] = 0;
    parameter_ylgndr['flag_dd'] = 0;
    if 'sqrt_2lp1_' not in locals(): sqrt_2lp1_=None; #end;
    if 'sqrt_2mp1_' not in locals(): sqrt_2mp1_=None; #end;
    if 'sqrt_rat0_m_' not in locals(): sqrt_rat0_m_=None; #end;
    if 'sqrt_rat3_lm__' not in locals(): sqrt_rat3_lm__=None; #end;
    if 'sqrt_rat4_lm__' not in locals(): sqrt_rat4_lm__=None; #end;
    (
        parameter_ylgndr,
        d0y_xlm___,
        _,
        _,
        sqrt_2lp1_,
        sqrt_2mp1_,
        sqrt_rat0_m_,
        sqrt_rat3_lm__,
        sqrt_rat4_lm__,
    ) = ylgndr_2(
        parameter_ylgndr,
        l_max_max,
        torch.cos(x_p_polar_a_x_),
        sqrt_2lp1_,
        sqrt_2mp1_,
        sqrt_rat0_m_,
        sqrt_rat3_lm__,
        sqrt_rat4_lm__,
    );
    tmp_t=toc(tmp_t);
    if (flag_verbose>0): disp(sprintf(' %% ylgndr_2: time %0.6fs',tmp_t)); #end;

    tmp_t=tic();
    t_p_kx__ = 2*pi* (torch.reshape(k_p_r_,mtr((n_k_p_r,1))) * torch.reshape(x_p_r_x_,mtr((1,n_x)))).to(dtype=torch.float32);
    tmp_index_big_ = efind(torch.abs(t_p_kx__.ravel())>=1e-12);
    t_p_kx_big_ = t_p_kx__.ravel()[tmp_index_big_];
    #tmp_index_sml_ = efind(torch.abs(t_p_kx__.ravel())< 1e-12);
    jl_kxl___ = torch.zeros(mtr((n_k_p_r,n_x,1+l_max_max))).to(dtype=torch.float32);
    for l_val in range(l_max_max+1):
        tmp_jl_kx__ = torch.zeros(mtr((n_k_p_r,n_x))).to(dtype=torch.float32);
        if (l_val==0): tmp_jl_kx__ = tmp_jl_kx__ + torch.tensor([1.0]); #end;
        #tmp_jl_kx__.ravel()[tmp_index_big_] = jv(l_val+0.5,t_p_kx_big_)*torch.sqrt(pi/(2*t_p_kx_big_));
        tmp_jl_kx__ = matlab_assign_lhs_from_rhs_(tmp_jl_kx__,tmp_index_big_, jv(l_val+0.5,t_p_kx_big_)*torch.sqrt(pi/(2*t_p_kx_big_)));
        tmp_i8_index_lhs_ = matlab_index_3d_0(n_k_p_r,':',n_x,':',1+l_max_max,l_val);
        #jl_kxl___.ravel()[tmp_i8_index_lhs_] = tmp_jl_kx__.ravel();
        jl_kxl___ = matlab_assign_lhs_from_rhs_(jl_kxl___,tmp_i8_index_lhs_, tmp_jl_kx__.ravel());
    #end;%for l_val=0:l_max_max;
    jl_xlk___ = torch.permute(jl_kxl___,mtr(mts((1,2,0))));
    tmp_t=toc(tmp_t);
    if (flag_verbose>0): disp(sprintf(' %% jl_xlk__: time %0.6fs',tmp_t)); #end;

    tmp_t=tic();
    m_val_ = torch.arange(-l_max_max,+l_max_max+1).to(dtype=torch.int32);
    #%expimb_xm__ = exp(+i*bsxfun(@times,reshape(x_p_azimu_b_x_,[n_x,1]),reshape(m_val_,[1,1+2*l_max_max])));
    exnimb_xm__ = torch.exp(-i*torch.reshape(x_p_azimu_b_x_,mtr((n_x,1)))*torch.reshape(m_val_,mtr((1,1+2*l_max_max)))).to(dtype=torch.complex64);
    tmp_t=toc(tmp_t);
    if (flag_verbose>0): disp(sprintf(' %% exnimb_xm__: time %0.6fs',tmp_t)); #end;

    tmp_t=tic();
    a_k_Y_lmk___ = torch.zeros(mtr((1+l_max_max,1+2*l_max_max,n_k_p_r))).to(dtype=torch.complex64);
    tmp_i8_index_rhs_xlm_ = matlab_index_3d_0(n_x,':',1+l_max_max,':',1+l_max_max,torch.abs(torch.arange(-l_max_max,+l_max_max+1)).to(dtype=torch.int32));
    a_k_Y_lmk___=torch.reshape(torch.sum(4*pi*torch.reshape(torch.tensor([i]).to(dtype=torch.complex64)**torch.arange(l_max_max+1),mtr((n_1,1+l_max_max,n_1,n_1)))*torch.reshape(jl_xlk___,mtr((n_x,1+l_max_max,n_1,n_k_p_r)))*torch.reshape(d0y_xlm___.ravel()[tmp_i8_index_rhs_xlm_],mtr((n_x,1+l_max_max,1+2*l_max_max,n_1)))*torch.reshape(exnimb_xm__,mtr((n_x,n_1,1+2*l_max_max,n_1)))/np.sqrt(4*pi),dim=3-0),mtr((1+l_max_max,1+2*l_max_max,n_k_p_r))).to(dtype=torch.complex64);
    tmp_t=toc(tmp_t);
    if (flag_verbose>0): disp(sprintf(' %% a_k_Y_lmk___: time %0.6fs',tmp_t)); #end;

    tmp_t=tic();
    a_k_Y_yk_ = torch.zeros(n_y_sum).to(dtype=torch.complex64);
    na=0;
    for nk_p_r in range(n_k_p_r):
        l_max = int(l_max_[nk_p_r].item());
        for l_val in range(l_max+1):
            m_val_ = torch.arange(-l_val,+l_val+1).to(dtype=torch.int32);
            tmp_i8_index_rhs_lmk_ = matlab_index_3d_0(1+l_max_max,l_val,1+2*l_max_max,l_max_max+m_val_,n_k_p_r,nk_p_r);
            a_k_Y_yk_[na+l_val+m_val_] = a_k_Y_lmk___.ravel()[tmp_i8_index_rhs_lmk_];
            na=na+1+2*l_val;
        #end;%for l_val=0:l_max;
    #end;%for nk_p_r=0:n_k_p_r-1;
    assert(na==n_y_sum);
    tmp_t=toc(tmp_t);
    if (flag_verbose>0): disp(sprintf(' %% a_k_Y_yk_: time %0.6fs',tmp_t)); #end;

    if (flag_verbose>0): disp(sprintf(' %% [finished %s]',str_thisfunction)); #end;

    return(
        parameter,
        a_k_Y_yk_,
        sqrt_2lp1_,
        sqrt_2mp1_,
        sqrt_rat0_m_,
        sqrt_rat3_lm__,
        sqrt_rat4_lm__,
    );
