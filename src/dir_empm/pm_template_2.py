from dir_empm.dir_matlab_macros import * ;
from dir_empm.sample_shell_6 import sample_shell_6 ;
from dir_empm.ylgndr_2 import ylgndr_2 ;

def pm_template_2(
        flag_verbose=None,
        l_max=None,
        n_a=None,
        a_k_Y_ya__=None,
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
    str_thisfunction = 'pm_template_2' ;

    #%%%%%%%%;
    if (flag_verbose): print(f' %% [entering {str_thisfunction}]');
    #%%%%%%%%;

    n_y = (l_max+1)**2;
    m_max_ = torch.arange(-l_max,+l_max+1).to(dtype=torch.int32);
    n_m_max = numel(m_max_); #%<-- 2*l_max+1;
    if (flag_verbose): print(f' %% l_max {l_max} n_y {n_y}');
    if a_k_Y_ya__ is None: a_k_Y_ya__ = torch.zeros(mtr((n_y,1))).to(dtype=torch.complex64);

    #%%%%%%%%;
    if (flag_verbose): print(f' %% First determine the viewing angles.');
    #%%%%%%%%;
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
    if (flag_verbose): print(f' %% n_S {n_S} n_viewing_polar_a {n_viewing_polar_a} n_viewing_azimu_b_sum {n_viewing_azimu_b_sum}');
    #%%%%%%%%;
    if (flag_verbose): print(f' %% Now determine the points along each equatorial plane (i.e., the points for each template).');
    #%%%%%%%%;

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
    if (flag_verbose): print(f' %% n_w {n_w}');
    #%%%%%%%%;
    #% Set up inner gamma_z for the templates. ;
    #%%%%%%%%;
    gamma_z_ = torch.zeros(n_w).to(dtype=torch.float32); n_gamma_z = n_w;
    gamma_z_ = torch.linspace(0,2*pi,n_w+1).to(dtype=torch.float32)[:n_gamma_z];
    cc_ = torch.cos(gamma_z_); sc_ = torch.sin(gamma_z_);
    #%%%%%%%%;
    #% initialize template_waS___. ;
    #%%%%%%%%;
    template_waS___ = torch.zeros(mtr((n_w,n_a,n_S))).to(dtype=torch.complex64);

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

    if (flag_verbose): print(f' %% template: ({n_w},{n_S},{n_a})={n_w*n_S*n_a} ({n_w*n_S*n_a*n_byte_per_complex64/1e9:.2f} GB)');

    #%%%%%%%%;
    if (flag_verbose): print(f' %% Now construct array of k_c_?_ values for the templates.');
    #%%%%%%%%;
    template_k_p_r = 1.0;
    #%%%%%%%%;
    template_k_c_0__ = torch.zeros(mtr((n_w,n_S))).to(dtype=torch.float32);
    template_k_c_1__ = torch.zeros(mtr((n_w,n_S))).to(dtype=torch.float32);
    template_k_c_2__ = torch.zeros(mtr((n_w,n_S))).to(dtype=torch.float32);
    if (flag_verbose): print(f' %% template_k_c___: ({n_w},{n_S})={n_w*n_S} ({n_w*n_S*n_byte_per_float32/1e9:.2f} GB)');
    for nS in range(n_S):
        viewing_polar_a = viewing_polar_a_S_[nS].item(); ca = np.cos(viewing_polar_a); sa = np.sin(viewing_polar_a);
        viewing_azimu_b = viewing_azimu_b_S_[nS].item(); cb = np.cos(viewing_azimu_b); sb = np.sin(viewing_azimu_b);
        tmp_i8_index_rhs_ = matlab_index_2d_0(n_w,':',n_S,nS);
        #template_k_c_0__.ravel()[tmp_i8_index_rhs_] = (+cb*ca*cc_ - sb*sc_)*template_k_p_r;
        #template_k_c_1__.ravel()[tmp_i8_index_rhs_] = (+sb*ca*cc_ + cb*sc_)*template_k_p_r;
        #template_k_c_2__.ravel()[tmp_i8_index_rhs_] = (-sa*cc_            )*template_k_p_r;
        template_k_c_0__ = matlab_assign_lhs_from_rhs_(template_k_c_0__,tmp_i8_index_rhs_, (+cb*ca*cc_ - sb*sc_)*template_k_p_r);
        template_k_c_1__ = matlab_assign_lhs_from_rhs_(template_k_c_1__,tmp_i8_index_rhs_, (+sb*ca*cc_ + cb*sc_)*template_k_p_r);
        template_k_c_2__ = matlab_assign_lhs_from_rhs_(template_k_c_2__,tmp_i8_index_rhs_, (-sa*cc_            )*template_k_p_r);
    #end;%for nS=0:n_S-1;
    template_azimu_b__ = torch.atan2(template_k_c_1__,template_k_c_0__);
    expi_template_azimu_b__ = torch.exp(i*template_azimu_b__);
    del template_azimu_b__;
    del template_k_c_0__; del template_k_c_1__; del template_k_c_2__ ;
    #%%%%%%%%;
    #% We also use a condensed array, called condense_k_c_2__, which only depends on the polar_a, and not on azimu_b. ;
    #%%%%%%%%;
    condense_k_c_2__ = torch.zeros(mtr((n_w,n_viewing_polar_a))).to(dtype=torch.float32);
    if (flag_verbose): print(f' %% condense_k_c_2__: ({n_w},{n_viewing_polar_a})={n_w*n_viewing_polar_a} ({n_w*n_viewing_polar_a*n_byte_per_float32/1e9:.2f} GB)');
    for nviewing_polar_a in range(n_viewing_polar_a):
        viewing_polar_a = viewing_polar_a_[nviewing_polar_a].item(); ca = np.cos(viewing_polar_a); sa = np.sin(viewing_polar_a);
        tmp_i8_index_rhs_ = matlab_index_2d_0(n_w,':',n_viewing_polar_a,nviewing_polar_a);
        #condense_k_c_2__.ravel()[tmp_i8_index_rhs_] = -sa*cc_ ;
        condense_k_c_2__ = matlab_assign_lhs_from_rhs_(condense_k_c_2__,tmp_i8_index_rhs_, -sa*cc_ );
    #end;%for nviewing_polar_a=0:n_S-1;
    #%%%%%%%%;
    if (flag_verbose): print(f' %% Now evaluate associated legendre polynomials at the various k_c_2 values.');
    #% Here legendre_evaluate_lmwS____{1+l_val}(1+l_val+m_val,1+nw,1+nviewing_polar_a) contains ;
    #% the associated legendre-function of degree l_val and order abs(m_val) (ranging from 0 to +l_val) ;
    #% evaluated at the k_c_2 value stored in condense_k_c_2__(1+nw,1+nviewing_polar_a). ;
    #% Note that this is associated with viewing_polar_a_(1+nviewing_polar_a). ;
    #% The legendre_normalization_{1+l_val}(1+abs(m_val)) contains ;
    #% The normalization coefficient for the spherical harmonics associated with l_val and m_val. ;
    #%%%%%%%%;
    parameter = {'type': 'parameter'} ;
    parameter['flag_verbose'] = int(np.maximum(0,flag_verbose-1)) ;
    parameter['flag_d'] = 0 ;
    parameter['flag_dd'] = 0 ;
    (
        parameter,
        y_jlm___,
        _,
        _,
        sqrt_2lp1_,
        sqrt_2mp1_,
        sqrt_rat0_m_,
        sqrt_rat3_lm__,
        sqrt_rat4_lm__,
    ) = ylgndr_2(
        parameter,
        l_max,
        condense_k_c_2__.ravel(),
        sqrt_2lp1_,
        sqrt_2mp1_,
        sqrt_rat0_m_,
        sqrt_rat3_lm__,
        sqrt_rat4_lm__,
    );
    y_lmwp____ = torch.reshape(torch.permute(y_jlm___,mtr(mts((1,2,0)))),mtr((1+l_max,1+l_max,n_w,n_viewing_polar_a)))/np.sqrt(4*pi);
    legendre_use_evaluate_normalized_lmwp____ = torch.zeros(mtr((1+l_max,n_m_max,n_w,n_viewing_polar_a))).to(dtype=torch.float32);
    tmp_i8_index_lhs_ = matlab_index_4d_0(1+l_max,':',n_m_max,l_max+torch.arange(0,+l_max+1,+1).to(dtype=torch.int32),n_w,':',n_viewing_polar_a,':');
    #legendre_use_evaluate_normalized_lmwp____.ravel()[tmp_i8_index_lhs_] = y_lmwp____.ravel();
    legendre_use_evaluate_normalized_lmwp____ = matlab_assign_lhs_from_rhs_(legendre_use_evaluate_normalized_lmwp____,tmp_i8_index_lhs_, y_lmwp____.ravel());
    tmp_i8_index_lhs_ = matlab_index_4d_0(1+l_max,':',n_m_max,l_max+torch.arange(0,-l_max-1,-1).to(dtype=torch.int32),n_w,':',n_viewing_polar_a,':');
    #legendre_use_evaluate_normalized_lmwp____.ravel()[tmp_i8_index_lhs_] = y_lmwp____.ravel();
    legendre_use_evaluate_normalized_lmwp____ = matlab_assign_lhs_from_rhs_(legendre_use_evaluate_normalized_lmwp____,tmp_i8_index_lhs_, y_lmwp____.ravel());

    #%%%%%%%%;
    legendre_evaluate_normalized_lwpm___ = torch.reshape(torch.permute(legendre_use_evaluate_normalized_lmwp____,mtr(mts((0,2,3,1)))),mtr((1+l_max,n_w*n_viewing_polar_a,n_m_max)));
    if (flag_verbose): print(f' %% legendre_evaluate_normalized_lwpm____: ({(1+l_max)},{n_w},{n_viewing_polar_a},{n_m_max})={((1+l_max)*n_w*n_viewing_polar_a*n_m_max)} ({((1+l_max)*n_w*n_viewing_polar_a*n_m_max*n_byte_per_float32/1e9):.2f} GB)');
    #%%%%%%%%;
    #% unroll a_k_Y_ya__. ;
    #%%%%%%%%;
    a_k_Y_lma___ = torch.zeros(mtr((1+l_max,n_m_max,n_a))).to(dtype=torch.complex64);
    if (flag_verbose): print(f' %% a_k_Y_lma___: ({(1+l_max)},{n_m_max},{n_a})={((1+l_max)*n_m_max*n_a)} ({((1+l_max)*n_m_max*n_a*n_byte_per_complex64/1e9):.2f} GB)');
    for l_val in range(l_max+1):
        index_m_0in_ = ((1+l_val-1)**2+l_val+torch.arange(-l_val,+l_val+1).to(dtype=torch.int32)).to(dtype=torch.int32);
        index_m_out_ = l_max + torch.arange(-l_val,+l_val+1).to(dtype=torch.int32);
        tmp_i8_index_lhs_ = matlab_index_3d_0(1+l_max,l_val,n_m_max,index_m_out_,n_a,':');
        tmp_i8_index_rhs_ = matlab_index_2d_0((1+l_max)**2,index_m_0in_,n_a,':');
        #a_k_Y_lma___.ravel()[tmp_i8_index_lhs_] = a_k_Y_ya__.ravel()[tmp_i8_index_rhs_];
        a_k_Y_lma___ = matlab_assign_lhs_from_rhs_(a_k_Y_lma___,tmp_i8_index_lhs_, a_k_Y_ya__.ravel()[tmp_i8_index_rhs_]);
    #end;%for l_val=0:l_max;
    a_k_Y_lam___ = torch.permute(a_k_Y_lma___,mtr(mts((0,2,1))));
    if (flag_verbose): print(f' %% a_k_Y_lam___: ({(1+l_max)},{n_a},{n_m_max})={((1+l_max)*n_a*n_m_max)} ({((1+l_max)*n_a*n_m_max*n_byte_per_complex64/1e9):.2f} GB)');
    #%%%%%%%%;
    if (flag_verbose): print(f' %% Now accumulate the legendre_evaluates over l_val, for each m_val.');
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

    n_a_per_a_batch = int(np.minimum(n_a,np.maximum(1,np.ceil( 0.5e9 / np.maximum(1,n_w*n_viewing_polar_a*(1+2*l_max)*n_byte_per_float32) )))); #%<-- 0.5GB limit. ;
    n_a_batch = int(np.ceil(n_a/np.maximum(1,n_a_per_a_batch)));
    if (flag_verbose): print(f' %% n_a_per_a_batch {n_a_per_a_batch}, n_a_batch {n_a_batch}');
    for na_batch in range(n_a_batch):
        index_a_ = n_a_per_a_batch*na_batch + torch.arange(n_a_per_a_batch).to(dtype=torch.int32);
        index_a_ = index_a_[index_a_<n_a];
        n_a_sub = numel(index_a_);
        if (n_a_sub>0):
            tmp_i8_index_rhs_ = matlab_index_3d_0(1+l_max,':',n_a,index_a_,n_m_max,':');
            tmp_a_k_Y_lam___ = torch.reshape(a_k_Y_lam___.ravel()[tmp_i8_index_rhs_],mtr((1+l_max,n_a_sub,n_m_max)));

            spherical_harmonic_unphased_wpam____ = torch.zeros(mtr((n_w,n_viewing_polar_a,n_a_sub,1+2*l_max))).to(dtype=torch.complex64);
            if (flag_verbose): print(f' %% spherical_harmonic_unphased_wpam____: ({n_w},{n_viewing_polar_a},{n_a_sub},{1+2*l_max})={n_w*n_viewing_polar_a*n_a_sub*(1+2*l_max)} ({n_w*n_viewing_polar_a*n_a_sub*(1+2*l_max)*n_byte_per_float32/1e9:.2f} GB)');
            for nm in range(n_m_max):
                tmp_i8_index_rhs_lwpm_ = matlab_index_3d_0(1+l_max,':',n_w*n_viewing_polar_a,':',n_m_max,nm);
                tmp_i8_index_rhs_lam_ = matlab_index_3d_0(1+l_max,':',n_a_sub,':',n_m_max,nm);
                tmp_i8_index_lhs_wpam_ = matlab_index_4d_0(n_w,':',n_viewing_polar_a,':',n_a_sub,':',n_m_max,nm);
                str_einsum = msr('lb') + ',' + msr('la') + '->' + msr('ba') ; #<-- symbol 'b' is the multi-index 'wp'. ;
                #spherical_harmonic_unphased_wpam____.ravel()[tmp_i8_index_lhs_wpam_] = torch.einsum( str_einsum , torch.reshape(legendre_evaluate_normalized_lwpm___.to(dtype=torch.complex64).ravel()[tmp_i8_index_rhs_lwpm_],mtr((1+l_max,n_w*n_viewing_polar_a))) , torch.reshape(tmp_a_k_Y_lam___.ravel()[tmp_i8_index_rhs_lam_],mtr((1+l_max,n_a_sub))) ).ravel() ;
                spherical_harmonic_unphased_wpam____ = matlab_assign_lhs_from_rhs_(spherical_harmonic_unphased_wpam____,tmp_i8_index_lhs_wpam_, torch.einsum( str_einsum , torch.reshape(legendre_evaluate_normalized_lwpm___.to(dtype=torch.complex64).ravel()[tmp_i8_index_rhs_lwpm_],mtr((1+l_max,n_w*n_viewing_polar_a))) , torch.reshape(tmp_a_k_Y_lam___.ravel()[tmp_i8_index_rhs_lam_],mtr((1+l_max,n_a_sub))) ).ravel() );
            #end;%for nm=0:n_m_max-1;
            spherical_harmonic_unphased_mawp____ = torch.permute(spherical_harmonic_unphased_wpam____,mtr(mts((3,2,0,1))));
            if (flag_verbose): print(f' %% spherical_harmonic_unphased_mawp____: ({(1+2*l_max)},{n_a_sub},{n_w},{n_viewing_polar_a})={((1+2*l_max)*n_a_sub*n_w*n_viewing_polar_a)} ({((1+2*l_max)*n_a_sub*n_w*n_viewing_polar_a*n_byte_per_float32/1e9):.2f} GB)');

            #%%%%%%%%;
            if (flag_verbose): print(f' %% now perform the final sum over m_val.');
            #%%%%%%%%;

            spherical_harmonic_evaluate_waS___ = torch.zeros(mtr((n_w,n_a_sub,n_S))).to(dtype=torch.complex64);
            if (flag_verbose): print(f' %% spherical_harmonic_evaluate_waS___: ({n_w},{n_a_sub},{n_S})={n_w*n_a_sub*n_S} ({n_w*n_a_sub*n_S*n_byte_per_complex64/1e9:.2f} GB)');
            nS=0;
            for nviewing_polar_a in range(n_viewing_polar_a):
                n_viewing_azimu_b = int(n_viewing_azimu_b_[nviewing_polar_a].item());
                for nviewing_azimu_b in range(n_viewing_azimu_b):
                    tmp_i8_index_rhs_ = matlab_index_2d_0(n_w,':',n_S,nS);
                    tmp_expi_sub_ = expi_template_azimu_b__.ravel()[tmp_i8_index_rhs_];
                    tmp_expi_pre_ = tmp_expi_sub_**(-l_max);
                    tmp_expi_pos_ = tmp_expi_pre_;
                    #%%%%;
                    tmp_expi_wm__ = torch.zeros(n_w,n_m_max).to(dtype=torch.complex64); 
                    tmp_i8_index_lhs_ = matlab_index_2d_0(n_w,':',n_m_max,0);
                    #tmp_expi_wm__.ravel()[tmp_i8_index_lhs_] = tmp_expi_pos_.ravel();
                    tmp_expi_wm__ = matlab_assign_lhs_from_rhs_(tmp_expi_wm__,tmp_i8_index_lhs_, tmp_expi_pos_.ravel());
                    for nm in range(1,n_m_max):
                        tmp_i8_index_rhs_ = matlab_index_2d_0(n_w,':',n_m_max,nm-1);
                        tmp_i8_index_lhs_ = matlab_index_2d_0(n_w,':',n_m_max,nm-0);
                        #tmp_expi_wm__.ravel()[tmp_i8_index_lhs_] = tmp_expi_wm__.ravel()[tmp_i8_index_rhs_]*tmp_expi_sub_;
                        tmp_expi_wm__ = matlab_assign_lhs_from_rhs_(tmp_expi_wm__,tmp_i8_index_lhs_, tmp_expi_wm__.ravel()[tmp_i8_index_rhs_]*tmp_expi_sub_);
                    #end;%for nm=1:n_m_max-1;
                    #%%%%;
                    tmp_sum_wa__ = torch.zeros(n_w,n_a_sub).to(dtype=torch.complex64);
                    for nw in range(n_w):
                        tmp_i8_index_rhs_wm_ = matlab_index_2d_0(n_w,nw,n_m_max,':');
                        tmp_i8_index_rhs_mawp_ = matlab_index_4d_0(n_m_max,':',n_a_sub,':',n_w,nw,n_viewing_polar_a,nviewing_polar_a);
                        tmp_i8_index_lhs_wa_ = matlab_index_2d_0(n_w,nw,n_a_sub,':');
                        str_einsum = msr('m') + ',' + msr('ma') + '->' + msr('a');
                        #tmp_sum_wa__.ravel()[tmp_i8_index_lhs_wa_] = torch.einsum( str_einsum , tmp_expi_wm__.ravel()[tmp_i8_index_rhs_wm_] , torch.reshape(spherical_harmonic_unphased_mawp____.ravel()[tmp_i8_index_rhs_mawp_],mtr((n_m_max,n_a_sub))) );
                        tmp_sum_wa__ = matlab_assign_lhs_from_rhs_(tmp_sum_wa__,tmp_i8_index_lhs_wa_, torch.einsum( str_einsum , tmp_expi_wm__.ravel()[tmp_i8_index_rhs_wm_] , torch.reshape(spherical_harmonic_unphased_mawp____.ravel()[tmp_i8_index_rhs_mawp_],mtr((n_m_max,n_a_sub))) ));
                    #end;%for nw=0:n_w-1;
                    tmp_i8_index_lhs_ = matlab_index_3d_0(n_w,':',n_a_sub,':',n_S,nS);
                    #spherical_harmonic_evaluate_waS___.ravel()[tmp_i8_index_lhs_] = tmp_sum_wa__.ravel();
                    spherical_harmonic_evaluate_waS___ = matlab_assign_lhs_from_rhs_(spherical_harmonic_evaluate_waS___,tmp_i8_index_lhs_, tmp_sum_wa__.ravel());
                    #%%%%;
                    nS=nS+1;
                #end;%for nviewing_azimu_b=0:n_viewing_azimu_b-1;
            #end;%for nviewing_polar_a=0:n_viewing_polar_a-1;

            tmp_i8_index_lhs_ = matlab_index_3d_0(n_w,':',n_a,index_a_,n_S,':');
            #template_waS___.ravel()[tmp_i8_index_lhs_] = spherical_harmonic_evaluate_waS___.ravel();
            template_waS___ = matlab_assign_lhs_from_rhs_(template_waS___,tmp_i8_index_lhs_, spherical_harmonic_evaluate_waS___.ravel());

        #%%%%%%%%%%%%%%%%;
        #end;%if (n_a_sub>0);
        #%%%%%%%%%%%%%%%%;

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;
    #end;%for na_batch=0:n_a_batch-1;
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;
    #% finished batches. ;
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;

    #%%%%%%%%;
    if (flag_verbose): print(f' %% [finished {str_thisfunction}]');
    #%%%%%%%%;
    return(
        template_waS___,
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

