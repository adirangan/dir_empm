from dir_empm.dir_matlab_macros import * ;
from dir_empm.ylgndr_2 import ylgndr_2 ;

def wignerd_lsq_b(
        n_l=None,
        beta=None,
):
    flag_verbose = 0;
    n_oversample = 5;
    l_max = n_l;
    m_max_ = torch.arange(-l_max,+l_max+1).to(dtype=torch.int32);
    n_m_max = 1+2*l_max;
    n_azimu_b = int(np.ceil(np.sqrt(n_oversample*2*n_m_max)));
    azimu_b_ = torch.tensor(np.sort(2*pi*np.random.rand(n_azimu_b))).to(dtype=torch.float32);
    n_polar_a = int(np.ceil(np.sqrt(n_oversample*1*n_m_max)));
    polar_a_ = torch.tensor(np.sort(1*pi*np.random.rand(n_polar_a))).to(dtype=torch.float32);
    [azimu_b_ori__,polar_a_ori__] = torch.meshgrid(azimu_b_,polar_a_,indexing='ij'); #<-- reversed to match matlab. ;
    #%%%%;
    tmp_t = tic();
    (
        _,
        d0y_jlm___,
        _,
        _,
        sqrt_2lp1_,
        sqrt_2mp1_,
        sqrt_rat0_m__,
        sqrt_rat3_lm__,
        sqrt_rat4_lm__,
    ) = ylgndr_2(
        [],
        l_max,
        torch.cos(polar_a_),
    )[:9];
    L_ori_lmab____ = torch.tile(torch.permute(torch.reshape(d0y_jlm___,mtr((n_polar_a,1+l_max,1+l_max))),mtr(mts((1,2,0)))),mtr((1,1,1,n_azimu_b)));
    tmp_t = toc(tmp_t); 
    if (flag_verbose): print(f' %% L_ori_lmab____: {tmp_t:.2f}s');
    #%%%%;
    index_rhs_lmab_ = matlab_index_4d_0(1+l_max,':',1+l_max,torch.arange(1+l_max-1,0,-1),n_polar_a,':',n_azimu_b,':')
    L_ori_lmab____ = torch.concatenate( ( torch.reshape(L_ori_lmab____.ravel()[index_rhs_lmab_],mtr((1+l_max,1+l_max-1,n_polar_a,n_azimu_b))) , L_ori_lmab____ ) , axis=4-2)/np.sqrt(4*pi);
    #s_ = torch.ones(n_m_max).to(dtype=torch.int32);
    #%s_ = (-1).^((m_max_<0).*m_max_); %<-- unnecessary here. ;
    expi_mb__ = torch.exp(+i*torch.reshape(m_max_,mtr((n_m_max,1)))*torch.reshape(azimu_b_,mtr((1,n_azimu_b))));
    Y_ori_lmab____ = L_ori_lmab____ * torch.reshape(expi_mb__,mtr((1,n_m_max,1,n_azimu_b))) ;
    #%%%%;
    cb = np.cos(+beta); sb = np.sin(+beta); sg = -1;
    Xn__ = torch.sin(polar_a_ori__)*torch.cos(azimu_b_ori__);
    Yn__ = torch.sin(polar_a_ori__)*torch.sin(azimu_b_ori__);
    Zn__ = torch.cos(polar_a_ori__);
    Xt__ = +cb*Xn__ + sg*sb*Zn__;
    Yt__ = Yn__;
    Zt__ = -sg*sb*Xn__ + cb*Zn__;
    azimu_b_rot__ = torch.atan2(Yt__,Xt__);
    polar_a_rot__ = torch.acos(Zt__);
    #%%%%;
    tmp_t = tic();
    tmp_parameter = {'type': 'parameter'}; tmp_parameter['flag_d'] = 0;  tmp_parameter['flag_dd'] = 0;
    (
        _,
        tmp_d0y_jlm___,
    ) = ylgndr_2(
        tmp_parameter,
        l_max,
        torch.cos(polar_a_rot__.ravel()),
        sqrt_2lp1_,
        sqrt_2mp1_,
        sqrt_rat0_m__,
        sqrt_rat3_lm__,
        sqrt_rat4_lm__,
    )[:2];    
    L_rot_lmab____ = torch.permute(torch.reshape(tmp_d0y_jlm___,mtr((n_polar_a,n_azimu_b,1+l_max,1+l_max))),mtr(mts((2,3,0,1))));
    tmp_t = toc(tmp_t); 
    if (flag_verbose): print(f' %% L_rot_lmab____: {tmp_t:.2f}s');
    index_rhs_lmab_ = matlab_index_4d_0(1+l_max,':',1+l_max,torch.arange(1+l_max-1,0,-1),n_polar_a,':',n_azimu_b,':')
    L_rot_lmab____ = torch.concatenate( ( torch.reshape(L_rot_lmab____.ravel()[index_rhs_lmab_],mtr((1+l_max,1+l_max-1,n_polar_a,n_azimu_b))) , L_rot_lmab____ ) , axis=4-2)/np.sqrt(4*pi);
    #s_ = torch.ones(n_m_max,1).to(dtype=torch.int32);
    #%s_ = (-1).^((m_max_<0).*m_max_); %<-- unnecessary here. ;
    expi_mab___ = torch.exp(+i*torch.reshape(m_max_,mtr((n_m_max,1)))*torch.reshape(azimu_b_rot__,mtr((1,n_polar_a,n_azimu_b))));
    Y_rot_lmab____ = L_rot_lmab____ * torch.reshape(expi_mab___,mtr((1,n_m_max,n_polar_a,n_azimu_b)));
    #%%%%;
    W_ = cell(1+l_max); #<-- cell array. ;
    tmp_t = tic();
    for l_val in range(l_max+1):
        tmp_i8_index_rhs_ = matlab_index_4d_0(1+l_max,l_val,n_m_max,l_max+torch.arange(-l_val,+l_val+1),n_polar_a,':',n_azimu_b,':');
        Y_ori_mab__ = torch.reshape(Y_ori_lmab____.ravel()[tmp_i8_index_rhs_],mtr((1+2*l_val,n_polar_a*n_azimu_b)));
        Y_rot_mab__ = torch.reshape(Y_rot_lmab____.ravel()[tmp_i8_index_rhs_],mtr((1+2*l_val,n_polar_a*n_azimu_b)));
        W_[l_val] = torch.real( torch.reshape(torch.linalg.lstsq(Y_ori_mab__,Y_rot_mab__).solution.ravel(),mtr((1+2*l_val,1+2*l_val))).T ) ; #<-- I believe that, as of 20251021, the torch.linalg.lstsq().solution should be ordered with matlab-compatible indexing. ;
    #end;%for l_val=0:l_max;
    tmp_t = toc(tmp_t);
    if (flag_verbose): print(f' %% W_: {tmp_t:.2f}s');

    return(W_);


