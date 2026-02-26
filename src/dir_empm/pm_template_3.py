from dir_empm.dir_matlab_macros import * ;
from dir_empm.sample_shell_6 import sample_shell_6 ;
from dir_empm.ylgndr_2 import ylgndr_2 ;

def pm_template_3(
        flag_verbose=None,
        l_max=None,
        n_k=None,
        a_k_Y_yk__=None,
        viewing_euler_k_eq_d=None,
        template_inplane_k_eq_d=None,
        n_w_input=None,
        n_S=None,
        viewing_azimu_b_S_=None,
        viewing_polar_a_S_=None,
        viewing_weight_S_=None,
        n_viewing_polar_a=None,
        viewing_polar_a_=None,
        n_viewing_azimu_b_=None,
        sqrt_2lp1_=None,
        sqrt_2mp1_=None,
        sqrt_rat0_m_=None,
        sqrt_rat3_lm__=None,
        sqrt_rat4_lm__=None,
):
    str_thisfunction = 'pm_template_3' ;

    #%%%%%%%%;
    if (flag_verbose>0): disp(sprintf(' %% [entering %s]',str_thisfunction)); #end;
    #%%%%%%%%;


    memory_limit_GB = 0.5; #%<-- 0.5GB for n_k_per_k_batch. ;

    n_y = (l_max+1)**2;
    m_max_ = torch.arange(-l_max,+l_max+1).to(dtype=torch.int32);
    n_m_max = numel(m_max_); #%<-- 2*l_max+1;
    if (flag_verbose>0): disp(sprintf(' %% l_max %d n_y %d',l_max,n_y)); #end;
    if isempty(a_k_Y_yk__): a_k_Y_yk__ = torch.zeros(n_y).to(dtype=torch.complex64); #end;

    #%%%%%%%%;
    if (flag_verbose): print(f' %% First determine the viewing angles.');
    #%%%%%%%%;
    tmp_t=tic();
    if n_S is None:
        k_p_r = 1;
        flag_uniform_over_polar_a = 0;
        (
            n_S,
            viewing_azimu_b_S_,
            viewing_polar_a_S_,
            viewing_weight_S_,
            _,
            _,
            _,
            n_viewing_polar_a,
            viewing_polar_a_,
            n_viewing_azimu_b_,
        ) = sample_shell_6(
            k_p_r,
            viewing_euler_k_eq_d,
            'L',
        ) ; #<-- obtain viewing angles. ;
    #end;%if isempty(n_S);
    #%%%%;
    if viewing_weight_S_ is None: viewing_weight_S_ = torch.ones(n_S).to(dtype=torch.float32);
    #%%%%;
    if n_viewing_polar_a is None:
        np_viewing_polar_a_ = np.unique(viewing_polar_a_S_.numpy().ravel(),return_index=False,return_inverse=False);
        viewing_polar_a_ = torch.tensor(np_viewing_polar_a_).to(dtype=torch.float32);
        n_viewing_polar_a = numel(viewing_polar_a_);
        n_viewing_azimu_b_ = torch.zeros(n_viewing_polar_a).to(dtype=torch.int32);
        for nviewing_polar_a in range(n_viewing_polar_a):
            viewing_polar_a = viewing_polar_a_[nviewing_polar_a].item();
            n_viewing_azimu_b_[nviewing_polar_a] = numel(efind(torch.abs(viewing_polar_a_S_-viewing_polar_a)< 1e-12));
        #end;%for nviewing_polar_a=0:n_viewing_polar_a-1;
    #end;%if isempty(n_viewing_polar_a);
    #%%%%;    
    n_viewing_azimu_b_sum = int(torch.sum(n_viewing_azimu_b_).item());
    n_viewing_azimu_b_csum_ = cumsum_0(n_viewing_azimu_b_);
    tmp_t = toc(tmp_t);
    if (flag_verbose>1): disp(sprintf(' %% determine viewing-angles: %0.6fs',tmp_t)); #end;
    if (flag_verbose>0): disp(sprintf(' %% n_S %d n_viewing_polar_a %d n_viewing_azimu_b_sum %d',n_S,n_viewing_polar_a,n_viewing_azimu_b_sum)); #end;
    #%%%%%%%%;
    if (flag_verbose>0): disp(sprintf(' %% Now determine the points along each equatorial plane (i.e., the points for each template).')); #end;
    #%%%%%%%%;
    tmp_t=tic();
    n_w = 0;
    if (template_inplane_k_eq_d>0):
        k_p_r = 1.0;
        n_equator = int(3+matlab_scalar_round(2*pi*k_p_r/template_inplane_k_eq_d));
        n_polar_a = int(3+matlab_scalar_round(n_equator/2));
        n_w = int(2*n_polar_a);
    #end;%if (template_inplane_k_eq_d>0);
    if (template_inplane_k_eq_d<=0):
        n_w = int(np.maximum(6,n_w_input));
    #end;%if (template_inplane_k_eq_d<=0);
    tmp_t = toc(tmp_t);
    if (flag_verbose>1): disp(sprintf(' %% determine n_w: %0.6fs',tmp_t)); #end;
    if (flag_verbose>0): disp(sprintf(' %% n_w %d',n_w)); #end;
    #%%%%%%%%;
    #% Set up inner gamma_z for the templates. ;
    #%%%%%%%%;
    gamma_z_ = torch.zeros(n_w).to(dtype=torch.float32); n_gamma_z = n_w;
    gamma_z_ = torch.linspace(0,2*pi,n_w+1).to(dtype=torch.float32)[:n_gamma_z];
    cc_ = torch.cos(gamma_z_); sc_ = torch.sin(gamma_z_);
    #%%%%%%%%;
    #% initialize template_wkS___. ;
    #%%%%%%%%;
    template_wkS___ = torch.zeros(mtr((n_w,n_k,n_S))).to(dtype=torch.complex64);

    #%%%%%%%%;
    #% The general formula used here is as follows. ;
    #% let sa and ca be sin(polar_a) and cos(polar_a), respectively. ;
    #% let sb and cb be sin(azimu_b) and cos(azimu_b), respectively. ;
    #% let sc and cc be sin(gamma_z) and cos(gamma_z), respectively. ;
    #% And rotation by azimu_b about the +z-axis is represented as: ;
    #% Rz(azimu_b) = ;
    #% [ +cb -sb 0 ] ;
    #% [ +sb +cb 0 ] ;
    #% [  0   0  1 ] ;
    #% And rotation by polar_a about the +y-axis is represented as: ;
    #% Ry(polar_a) = ;
    #% [ +ca 0 +sa ] ;
    #% [  0  1  0  ] ;
    #% [ -sa 0 +ca ] ;
    #% And rotation by gamma_z about the +z-axis is represented as: ;
    #% Rz(gamma_z) = ;
    #% [ +cc -sc 0 ] ;
    #% [ +sc +cc 0 ] ;
    #% [  0   0  1 ] ;
    #% Which, collectively, implies that under the transform: ;
    #% Rz(azimu_b) * Ry(polar_a) * Rz(gamma_z), ;
    #% Which is the same as: ;
    #% [ +cb -sb 0 ] [ +ca*cc -ca*sc +sa ]   [ +cb*ca*cc - sb*sc , -cb*ca*sc -sb*cc , +cb*sa ];
    #% [ +sb +cb 0 ] [ +sc    +cc    0   ] = [ +sb*ca*cc + cb*sc , -sb*ca*sc +cb*cc , +sb*sa ];
    #% [  0   0  1 ] [ -sa*cc +sa*sc +ca ]   [ -sa*cc            , +sa*sc           , +ca    ];
    #% the point [1;0;0] is mapped to: ;
    #% [ template_k_c_0 ; template_k_c_1 ; template_k_c_2 ] = [ +cb*ca*cc - sb*sc ; +sb*ca*cc + cb*sc ; -sa*cc ];
    #%%%%%%%%;

    if (flag_verbose>0): disp(sprintf(' %% template: (%d,%d,%d)=%d (%0.2f GB)',n_w,n_S,n_k,n_w*n_S*n_k,n_w*n_S*n_k*16/1e9)); #end;

    #%%%%%%%%;
    if (flag_verbose): print(f' %% Now construct array of k_c_?_ values for the templates.');
    #%%%%%%%%;
    tmp_t=tic();
    template_k_p_r = 1.0;
    #%%%%%%%%;
    template_k_c_0__ = torch.zeros(mtr((n_w,n_S))).to(dtype=torch.float32);
    template_k_c_1__ = torch.zeros(mtr((n_w,n_S))).to(dtype=torch.float32);
    #template_k_c_2__ = torch.zeros(mtr((n_w,n_S))).to(dtype=torch.float32);
    if (flag_verbose>0): disp(sprintf(' %% template_k_c___: (%d,%d)=%d (%0.2f GB)',n_w,n_S,n_w*n_S,n_w*n_S*8/1e9)); #end;
    ca_ = torch.cos(viewing_polar_a_S_); sa_ = torch.sin(viewing_polar_a_S_);
    cb_ = torch.cos(viewing_azimu_b_S_); sb_ = torch.sin(viewing_azimu_b_S_);
    ca__ = torch.tile(torch.reshape(ca_,mtr((1,n_S))),mtr((n_w,1))); sa__ = torch.tile(torch.reshape(sa_,mtr((1,n_S))),mtr((n_w,1)));
    cb__ = torch.tile(torch.reshape(cb_,mtr((1,n_S))),mtr((n_w,1))); sb__ = torch.tile(torch.reshape(sb_,mtr((1,n_S))),mtr((n_w,1)));
    cc__ = torch.tile(torch.reshape(cc_,mtr((n_w,1))),mtr((1,n_S))); sc__ = torch.tile(torch.reshape(sc_,mtr((n_w,1))),mtr((1,n_S)));
    template_k_c_0__ = (+cb__*ca__*cc__ - sb__*sc__)*template_k_p_r;
    template_k_c_1__ = (+sb__*ca__*cc__ + cb__*sc__)*template_k_p_r;
    #%template_k_c_2__ = (-sa__*cc__                   )*template_k_p_r;
    template_azimu_b__ = torch.atan2(template_k_c_1__,template_k_c_0__);
    expi_template_azimu_b__ = torch.exp(i*template_azimu_b__).to(dtype=torch.complex64);
    del template_azimu_b__;
    del template_k_c_0__ ; del template_k_c_1__ ; #del template_k_c_2__ ;
    tmp_t = toc(tmp_t);
    if (flag_verbose>1): disp(sprintf(' %% expi_template_azimu_b__: %0.6fs',tmp_t)); #end;
    #%%%%%%%%;
    #% We also use a condensed array, called condense_k_c_2__, which only depends on the polar_a, and not on azimu_b. ;
    #%%%%%%%%;
    condense_k_c_2__ = torch.zeros(mtr((n_w,n_viewing_polar_a))).to(dtype=torch.float32);
    if (flag_verbose>0): disp(sprintf(' %% condense_k_c_2__: (%d,%d)=%d (%0.2f GB)',n_w,n_viewing_polar_a,n_w*n_viewing_polar_a,n_w*n_viewing_polar_a*8/1e9)); #end;
    condense_k_c_2__ = -(torch.reshape(torch.sin(viewing_polar_a_),mtr((1,n_viewing_polar_a))) * torch.reshape(cc_,mtr((n_w,1)))).to(dtype=torch.float32);
    tmp_t = toc(tmp_t);
    if (flag_verbose>1): disp(sprintf(' %% condense_k_c_2__: %0.6fs',tmp_t)); #end;

    #%%%%%%%%;
    if (flag_verbose>0): disp(sprintf(' %% Now evaluate associated legendre polynomials at the various k_c_2 values.')); #end;
    #%%%%%%%%;
    #% we use ylgndr_2 (i.e., yrecursion.f). ;
    #%%%%%%%%;
    tmp_t=tic();
    (
        _,
        y_jlm___,
        _,
        _,
        sqrt_2lp1_,
        sqrt_2mp1_,
        sqrt_rat0_m_,
        sqrt_rat3_lm__,
        sqrt_rat4_lm__,
    ) = ylgndr_2(
        {'type':'parameter','flag_verbose':0,'flag_d':0,'flag_dd':0},
        l_max,
        condense_k_c_2__.ravel(),
        sqrt_2lp1_,
        sqrt_2mp1_,
        sqrt_rat0_m_,
        sqrt_rat3_lm__,
        sqrt_rat4_lm__,
    );
    y_lmwa____ = torch.reshape(torch.permute(y_jlm___,mtr(mts((1,2,0)))),mtr((1+l_max,1+l_max,n_w,n_viewing_polar_a)))/np.sqrt(4*pi);
    legendre_use_evaluate_normalized_lmwa____ = torch.zeros(mtr((1+l_max,n_m_max,n_w,n_viewing_polar_a))).to(dtype=torch.float32);
    tmp_i8_index_lhs_ = matlab_index_4d_0(1+l_max,':',n_m_max,l_max+torch.arange(0,+l_max+1,+1).to(dtype=torch.int32),n_w,':',n_viewing_polar_a,':');
    legendre_use_evaluate_normalized_lmwa____.ravel()[tmp_i8_index_lhs_] = y_lmwa____.ravel();
    tmp_i8_index_lhs_ = matlab_index_4d_0(1+l_max,':',n_m_max,l_max+torch.arange(0,-l_max-1,-1).to(dtype=torch.int32),n_w,':',n_viewing_polar_a,':');
    legendre_use_evaluate_normalized_lmwa____.ravel()[tmp_i8_index_lhs_] = y_lmwa____.ravel();
    tmp_t = toc(tmp_t);
    if (flag_verbose>1): disp(sprintf(' %% legendre_use_evaluate_normalize_lmwa____: %0.6fs',tmp_t)); #end;
    #%%%%%%%%;

    #%%%%%%%%;
    tmp_t=tic();
    legendre_evaluate_normalized_lwam____ = torch.reshape(torch.permute(legendre_use_evaluate_normalized_lmwa____,mtr(mts((0,2,3,1)))),mtr((1+l_max,n_w,n_viewing_polar_a,n_m_max)));
    if (flag_verbose>0): disp(sprintf(' %% legendre_evaluate_normalized_lwam____: (%d,%d,%d,%d)=%d (%0.2f GB)',(1+l_max),n_w,n_viewing_polar_a,n_m_max,(1+l_max)*n_w*n_viewing_polar_a*n_m_max,(1+l_max)*n_w*n_viewing_polar_a*n_m_max*8/1e9)); #end;
    tmp_t = toc(tmp_t);
    if (flag_verbose>1): disp(sprintf(' %% legendre_use_evaluate_normalize_lwam____: %0.6fs',tmp_t)); #end;
    #%%%%%%%%;
    #% unroll a_k_Y_yk__. ;
    #%%%%%%%%;
    a_k_Y_lmk___ = torch.zeros(mtr((1+l_max,n_m_max,n_k))).to(dtype=torch.complex64);
    if (flag_verbose>0): disp(sprintf(' %% a_k_Y_lmk___: (%d,%d,%d)=%d (%0.2f GB)',(1+l_max),n_m_max,n_k,(1+l_max)*n_m_max*n_k,(1+l_max)*n_m_max*n_k*16/1e9)); #end;
    for l_val in range(l_max+1):
        index_m_0in_ = ((1+l_val-1)**2+l_val+torch.arange(-l_val,+l_val+1).to(dtype=torch.int32)).to(dtype=torch.int32);
        index_m_out_ = l_max + torch.arange(-l_val,+l_val+1).to(dtype=torch.int32);
        tmp_i8_index_lhs_ = matlab_index_3d_0(1+l_max,l_val,n_m_max,index_m_out_,n_k,':');
        tmp_i8_index_rhs_ = matlab_index_2d_0((1+l_max)**2,index_m_0in_,n_k,':');
        a_k_Y_lmk___.ravel()[tmp_i8_index_lhs_] = a_k_Y_yk__.ravel()[tmp_i8_index_rhs_];
    #end;%for l_val=0:l_max;
    a_k_Y_lkm___ = torch.permute(a_k_Y_lmk___,mtr(mts((0,2,1))));
    if (flag_verbose): print(f' %% a_k_Y_lkm___: ({(1+l_max)},{n_k},{n_m_max})={((1+l_max)*n_k*n_m_max)} ({((1+l_max)*n_k*n_m_max*n_byte_per_complex64/1e9):.2f} GB)');
    #%%%%%%%%;
    if (flag_verbose>0): disp(sprintf(' %% Now accumulate the legendre_evaluates over l_val, for each m_val.')); #end;
    #% We account for the normalization coefficients here, ;
    #% so that later we can apply the complex exponential to produce the spherical-harmonics. ;
    #% More specifically, for each na: ;
    #% spherical_harmonic_unphased_(1+l_max+m_val,1+nw,1+nviewing_polar_a) ;
    #% contains the sum: ;
    #% \sum_{l_val=0}^{l_max} ... ;
    #%             legendre_normalization_{1+l_val}(1+l_val+m_val) ... ;
    #%           * legendre_evaluate_lmwS____{1+l_val}(1+l_val+m_val,1+nw,1+nviewing_polar_a) ... ;
    #%           * a_k_Y_(1+(1+l_val-1)^2+l_val+m_val). ;
    #% Note that this is 'unphased', in the sense that the full evaluate requires: ;
    #% spherical_harmonic_evaluate__(1+nw,1+nS) = ... ;
    #% \sum_{m_val=-l_max}^{+l_max} ... ;
    #%             spherical_harmonic_unphased_(1+l_max+m_val,1+nw,1+nviewing_polar_a) ... ;
    #%           * exp(+i*m_val*template_azimu_b__(1+nw,1+nS)). ;
    #% Note that this final exponential can be calculated as: ;
    #% expi_template_azimu_b__(1+nw,1+nS)^m_val. ;
    #%%%%%%%%;

    n_k_per_k_batch = int(np.minimum(n_k,np.maximum(1,np.ceil( memory_limit_GB * 1e9 / np.maximum(1,n_w*n_viewing_polar_a*(1+2*l_max)*n_byte_per_float32) )))); #%<-- 0.5GB limit. ;
    n_k_batch = int(np.ceil(n_k/np.maximum(1,n_k_per_k_batch)));
    if (flag_verbose>0): disp(sprintf(' %% n_k_per_k_batch %d, n_k_batch %d',n_k_per_k_batch,n_k_batch)); #end;
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;
    for nk_batch in range(n_k_batch):
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;
        index_k_ = n_k_per_k_batch*nk_batch + torch.arange(n_k_per_k_batch).to(dtype=torch.int32);
        index_k_ = index_k_[index_k_<n_k];
        n_k_sub = numel(index_k_);
        #%%%%%%%%%%%%%%%%;
        if (n_k_sub>0):
        #%%%%%%%%%%%%%%%%;
            tmp_i8_index_rhs_ = matlab_index_3d_0(1+l_max,':',n_k,index_k_,n_m_max,':');
            tmp_a_k_Y_lkm___ = torch.reshape(a_k_Y_lkm___.ravel()[tmp_i8_index_rhs_],mtr((1+l_max,n_k_sub,n_m_max)));

            tmp_t=tic();
            spherical_harmonic_unphased_wakm____ = torch.zeros(mtr((n_w,n_viewing_polar_a,n_k_sub,1+2*l_max))).to(dtype=torch.complex64);
            if (flag_verbose>0): disp(sprintf(' %% spherical_harmonic_unphased_wakm____: (%d,%d,%d,%d)=%d (%0.2f GB)',n_w,n_viewing_polar_a,n_k_sub,1+2*l_max,n_w*n_viewing_polar_a*n_k_sub*(1+2*l_max),n_w*n_viewing_polar_a*n_k_sub*(1+2*l_max)*8/1e9)); #end;
            str_einsum = msr('lwam') + ',' + msr('lkm') + '->' + msr('wakm') ;
            spherical_harmonic_unphased_wakm____ = torch.reshape(torch.einsum( str_einsum , legendre_evaluate_normalized_lwam____.to(dtype=torch.complex64) , tmp_a_k_Y_lkm___.to(dtype=torch.complex64) ),mtr((n_w,n_viewing_polar_a,n_k_sub,1+2*l_max)));
            spherical_harmonic_unphased_mkwa____ = torch.permute(spherical_harmonic_unphased_wakm____,mtr(mts((3,2,0,1))));
            if (flag_verbose>0): disp(sprintf(' %% spherical_harmonic_unphased_mkwa____: (%d,%d,%d,%d)=%d (%0.2f GB)',(1+2*l_max),n_k_sub,n_w,n_viewing_polar_a,(1+2*l_max)*n_k_sub*n_w*n_viewing_polar_a,(1+2*l_max)*n_k_sub*n_w*n_viewing_polar_a*8/1e9)); #end;
            tmp_t = toc(tmp_t);
            if (flag_verbose>1): disp(sprintf(' %% spherical_harmonic_unphased_mkwa____: %0.6fs',tmp_t)); #end;

#            tmp_t=tic();
#            spherical_harmonic_unphased_wakm____ = torch.zeros(mtr((n_w,n_viewing_polar_a,n_k_sub,1+2*l_max))).to(dtype=torch.complex64);
#            if (flag_verbose): print(f' %% spherical_harmonic_unphased_wakm____: ({n_w},{n_viewing_polar_a},{n_k_sub},{1+2*l_max})={n_w*n_viewing_polar_a*n_k_sub*(1+2*l_max)} ({n_w*n_viewing_polar_a*n_k_sub*(1+2*l_max)*n_byte_per_float32/1e9:.2f} GB)');
#            for nm in range(n_m_max):
#                tmp_i8_index_rhs_lwam_ = matlab_index_3d_0(1+l_max,':',n_w*n_viewing_polar_a,':',n_m_max,nm);
#                tmp_i8_index_rhs_lkm_ = matlab_index_3d_0(1+l_max,':',n_k_sub,':',n_m_max,nm);
#                tmp_i8_index_lhs_wakm_ = matlab_index_4d_0(n_w,':',n_viewing_polar_a,':',n_k_sub,':',n_m_max,nm);
#                str_einsum = msr('lb') + ',' + msr('la') + '->' + msr('ba') ; #<-- symbol 'b' is the multi-index 'wp'. ;
#                spherical_harmonic_unphased_wakm____.ravel()[tmp_i8_index_lhs_wakm_] = torch.einsum( str_einsum , torch.reshape(legendre_evaluate_normalized_lwam___.to(dtype=torch.complex64).ravel()[tmp_i8_index_rhs_lwam_],mtr((1+l_max,n_w*n_viewing_polar_a))) , torch.reshape(tmp_a_k_Y_lkm___.ravel()[tmp_i8_index_rhs_lkm_],mtr((1+l_max,n_k_sub))) ).ravel() ;
#            #end;%for nm=0:n_m_max-1;
#            spherical_harmonic_unphased_mkwa____ = torch.permute(spherical_harmonic_unphased_wakm____,mtr(mts((3,2,0,1))));
#            if (flag_verbose>0):
#                disp(sprintf(' %% spherical_harmonic_unphased_mkwa____: (%d,%d,%d,%d)=%d (%0.2f GB)',(1+2*l_max),n_k_sub,n_w,n_viewing_polar_a,(1+2*l_max)*n_k_sub*n_w*n_viewing_polar_a,(1+2*l_max)*n_k_sub*n_w*n_viewing_polar_a*8/1e9)); #end;
#            tmp_t = toc(tmp_t);
#            if (flag_verbose>1): disp(sprintf(' %% spherical_harmonic_unphased_mkwa____: %0.6fs',tmp_t)); #end;

            #%%%%%%%%;
            #% Here we rewrite the final sum from pm_template_2 to use pagemtimes. ;
            #%%%%%%%%;
            if (flag_verbose>0): disp(sprintf(' %% now perform the final sum over m_val.')); #end;
            #%%%%%%%%;
            tmp_t=tic();
            spherical_harmonic_evaluate_wkS___ = torch.zeros(mtr((n_w,n_k_sub,n_S))).to(dtype=torch.complex64);
            if (flag_verbose>0):
                disp(sprintf(' %% spherical_harmonic_evaluate_wkS___: (%d,%d,%d)=%d (%0.2f GB)',n_w,n_k_sub,n_S,n_w*n_k_sub*n_S,n_w*n_k_sub*n_S*16/1e9)); #end;
            nS=0;
            for nviewing_polar_a in range(n_viewing_polar_a):
                tmp_i8_index_rhs_ = matlab_index_4d_0(1+2*l_max,':',n_k_sub,':',n_w,':',n_viewing_polar_a,nviewing_polar_a);
                tmp_shu_mkw1____ = torch.reshape(spherical_harmonic_unphased_mkwa____.ravel()[tmp_i8_index_rhs_],mtr((n_m_max,n_k_sub,n_w,n_1)));
                n_viewing_azimu_b = int(n_viewing_azimu_b_[nviewing_polar_a].item());
                n_S_sub = n_viewing_azimu_b;
                index_nS_sub_ = nS + torch.arange(n_S_sub);
                tmp_i8_index_rhs_ = matlab_index_2d_0(n_w,':',n_S,index_nS_sub_);
                tmp_expi_sub_wSm___ = (torch.reshape(expi_template_azimu_b__.ravel()[tmp_i8_index_rhs_],mtr((n_w,n_S_sub,1)))**torch.reshape(torch.arange(-l_max,+l_max+1),mtr((1,1,n_m_max)))).to(dtype=torch.complex64);
                tmp_expi_sub_1mwS____ = torch.reshape(torch.permute(tmp_expi_sub_wSm___,mtr(mts((2,0,1)))),mtr((n_1,n_m_max,n_w,n_S_sub)));
                tmp_i8_index_lhs_ = matlab_index_3d_0(n_w,':',n_k_sub,':',n_S,index_nS_sub_);
                str_einsum = msr('mwS') + ',' + msr('mkw') + '->' + msr('wkS') ;
                spherical_harmonic_evaluate_wkS___.ravel()[tmp_i8_index_lhs_] = torch.einsum( str_einsum , torch.reshape(tmp_expi_sub_1mwS____.to(dtype=torch.complex64),mtr((n_m_max,n_w,n_S_sub))) , torch.reshape(tmp_shu_mkw1____.to(dtype=torch.complex64),mtr((n_m_max,n_k_sub,n_w))) ).ravel() ;
                nS=nS+n_S_sub;
            #end;%for nviewing_polar_a=0:n_viewing_polar_a-1;
            assert(nS==n_S);
            tmp_t = toc(tmp_t);
            if (flag_verbose>1): disp(sprintf(' %% spherical_harmonic_evaluate_wkS___: %0.6fs',tmp_t)); #end;
            #%%%%%%%%;
            
            tmp_i8_index_lhs_ = matlab_index_3d_0(n_w,':',n_k,index_k_,n_S,':');
            template_wkS___.ravel()[tmp_i8_index_lhs_] = spherical_harmonic_evaluate_wkS___.ravel();

        #%%%%%%%%%%%%%%%%;
        #end;%if (n_k_sub>0);
        #%%%%%%%%%%%%%%%%;

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;
    #end;%for nk_batch=0:n_k_batch-1;
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;
    #% finished batches. ;
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;

    #%%%%%%%%;
    if (flag_verbose>0): disp(sprintf(' %% [finished %s]',str_thisfunction)); #end;
    #%%%%%%%%;
    return(
        template_wkS___,
        n_w,
        n_S,
        viewing_azimu_b_S_,
        viewing_polar_a_S_,
        viewing_weight_S_,
        n_viewing_polar_a,
        viewing_polar_a_,
        n_viewing_azimu_b_,
        sqrt_2lp1_,
        sqrt_2mp1_,
        sqrt_rat0_m_,
        sqrt_rat3_lm__,
        sqrt_rat4_lm__,
    );

