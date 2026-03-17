from dir_empm.dir_matlab_macros import * ;
from dir_empm.ylgndr_2 import ylgndr_2 ;
from dir_empm.local_yk__from_yk_ import local_yk__from_yk_;
from dir_empm.local_yk_from_yk__ import local_yk_from_yk__;

def convert_spharm_to_k_p_uniform_over_n_k_p_r_5(
        flag_verbose=None,
        n_qk=None,
        n_qk_csum_=None,
        k_p_r_qk_=None,
        k_p_azimu_b_qk_=None,
        k_p_polar_a_qk_=None,
        weight_3d_k_p_qk_=None,
        weight_shell_qk_=None,
        n_k_p_r=None,
        k_p_r_=None,
        weight_3d_k_p_r_=None,
        l_max_=None,
        a_k_Y_yk_=None,
        sqrt_2lp1_=None,
        sqrt_2mp1_=None,
        sqrt_rat0_m_=None,
        sqrt_rat3_lm__=None,
        sqrt_rat4_lm__=None,
):
    #% uses spherical-harmonic-expansion a_k_Y_ to evaluate a_k_p_ on a collection of points on spherical shells determined by k_p_r_. ;
    #% We assume that the polar-representation and quadrature weights associated with these points have been previously calculated. ; 
    #% We also assume that flag_uniform_over_n_k_p_r==1 when generating the spherical grid. ;
    #% ;
    #% inputs: ;
    #% ;
    #% flag_verbose = integer verbosity_level. ;
    #% n_qk = integer total number of points. ;
    #% n_qk_csum_ = integer array of starting indices associated with each k-value. ;
    #% k_p_r_qk_ = real array of k-values for each point. ;
    #% k_p_azimu_b_qk_ = real array of azimu_b-values for each point. ;
    #% k_p_polar_a_qk_ = real array of polar_a-values for each point. ;
    #% weight_3d_k_p_qk_ = real array of quadrature weights for volume integral (for each point) (unused). ;
    #% weight_shell_qk_ = real array of quadrature weights for shell integral (for each point). ;
    #% n_k_p_r = integer maximum k. ;
    #% k_p_r_ = real array of length n_k_p_r; k_p_r_(nk_p_r) = k_p_r_value for shell nk_p_r. ;
    #% weight_3d_k_p_r_ = real array of length n_k_p_r; radial quadrature weights (already assumed to be a factor of weight_3d_k_p_qk_) (unused). ;
    #% l_max_ = integer array of length n_k_p_r; l_max_(nk_p_r) = spherical harmonic order on shell nk_p_r; l_max_(nk_p_r) corresponds to n_y_(nk_p_r) = (l_max_(nk_p_r)+1)^2 coefficients. ;
    #% a_k_Y_yk_ = complex array of length \sum_{nk_p_r} (n_y_(nk_p_r)+1)^2 ; coefficients are ordered in a row, with m varying quickly, l varying slowly and k varying most slowly. ;
    #% ;
    #% outputs: ;
    #% ;
    #% a_k_p_qk_ = complex array of a-values for each point. ;

    str_thisfunction = 'convert_spharm_to_k_p_uniform_over_n_k_p_r_5';
    tolerance_machine = 1e-6;
    n_y_ = ((1+l_max_.ravel())**2).to(dtype=torch.int32); n_y_sum = int(torch.sum(n_y_).item()); n_y_max = int(torch.max(n_y_).item()); n_y_csum_ = cumsum_0(n_y_);
    if (flag_verbose>0): disp(sprintf(' %% [entering %s] n_qk %d, n_y_sum %d',str_thisfunction,n_qk,sum(n_y_))); #end;

    if (torch.std(torch.diff(n_qk_csum_).to(dtype=torch.float32)).item()>1e-6): disp(sprintf(' %% Warning, flag_uniform_over_n_k_p_r==1 expected in %s',str_thisfunction)); #end;
    n_q = int(n_qk_csum_[1].item());
    assert(n_qk==n_q*n_k_p_r);
    k_p_r_qk__ = torch.reshape(k_p_r_qk_,mtr((n_q,n_k_p_r)));
    if (torch.max(torch.std(k_p_r_qk__,dim=1-0)).item()>1e-6): disp(sprintf(' %% Warning, flag_uniform_over_n_k_p_r==1 expected in %s',str_thisfunction)); #end;
    k_p_azimu_b_qk__ = torch.reshape(k_p_azimu_b_qk_,mtr((n_q,n_k_p_r)));
    if (torch.max(torch.std(k_p_azimu_b_qk__,dim=1-1)).item()>1e-6): disp(sprintf(' %% Warning, flag_uniform_over_n_k_p_r==1 expected in %s',str_thisfunction)); #end;
    k_p_polar_a_qk__ = torch.reshape(k_p_polar_a_qk_,mtr((n_q,n_k_p_r)));
    if (torch.max(torch.std(k_p_polar_a_qk__,dim=1-1)).item()>1e-6): disp(sprintf(' %% Warning, flag_uniform_over_n_k_p_r==1 expected in %s',str_thisfunction)); #end;

    k_p_azimu_b_q_ = k_p_azimu_b_qk__[0,:].ravel();
    k_p_polar_a_q_ = k_p_polar_a_qk__[0,:].ravel();
    k_p_polar_a_unique_,index_unique_,index_return_ = unique_0(k_p_polar_a_q_); 
    n_polar_a_unique = numel(k_p_polar_a_unique_);
    l_max_max = int(torch.max(l_max_).item());
    if 'sqrt_2lp1_' not in locals(): sqrt_2lp1_=None; #end;
    if 'sqrt_2mp1_' not in locals(): sqrt_2mp1_=None; #end;
    if 'sqrt_rat0_m_' not in locals(): sqrt_rat0_m_=None; #end;
    if 'sqrt_rat3_lm__' not in locals(): sqrt_rat3_lm__=None; #end;
    if 'sqrt_rat4_lm__' not in locals(): sqrt_rat4_lm__=None; #end;
    tmp_t = tic();
    parameter_ylgndr = {'type':'ylgndr'};
    parameter_ylgndr['flag_verbose'] = 0;
    parameter_ylgndr['flag_d'] = 0;
    parameter_ylgndr['flag_dd'] = 0;
    (
        parameter_ylgndr,
        d0y_jlm___,
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
        torch.cos(k_p_polar_a_unique_),
        sqrt_2lp1_,
        sqrt_2mp1_,
        sqrt_rat0_m_,
        sqrt_rat3_lm__,
        sqrt_rat4_lm__,
    );
    tmp_t = toc(tmp_t);
    if (flag_verbose>0): disp(sprintf(' %% ylgndr_2: %0.6fs',tmp_t)); #end;

    #%%%%%%%%;
    #% This version involves preliminary inflation of m_val_, ;
    #% as well as extraneous exp evaluations, ;
    #% all to avoid memory movement. ;
    #%%%%%%%%;

    tmp_t = tic();
    d0y_lmj___ = torch.permute(d0y_jlm___,mtr(mts((1,2,0)))); #%<-- permutation before inflation is faster. ;
    tmp_i8_index_rhs_ = matlab_index_3d_0(1+l_max_max,':',1+l_max_max,torch.abs(torch.arange(-l_max_max,+l_max_max+1)).to(dtype=torch.int32),n_polar_a_unique,':');
    d0y_jml___ = torch.permute(torch.reshape(d0y_lmj___.ravel()[tmp_i8_index_rhs_],mtr((1+l_max_max,1+2*l_max_max,n_polar_a_unique))),mtr(mts((2,1,0)))); #%<-- retain unique cos(polar_a_{j}) for now.; 
    d0y_jy__ = torch.zeros(mtr((n_polar_a_unique,n_y_max))).to(dtype=torch.float32);
    for nl in range(numel(l_max_)):
        l_max = int(l_max_[nl].item());
        for l_val in range(l_max+1):
            m_val_ = torch.arange(-l_val,+l_val+1).to(dtype=torch.int32);
            tmp_i8_index_lhs_ = matlab_index_2d_0(n_polar_a_unique,':',n_y_max,l_val**2+l_val+m_val_);
            tmp_i8_index_rhs_ = matlab_index_3d_0(n_polar_a_unique,':',1+2*l_max_max,l_max_max+m_val_,1+l_max_max,l_val);
            #d0y_jy__.ravel()[tmp_i8_index_lhs_] = d0y_jml___.ravel()[tmp_i8_index_rhs_];
            d0y_jy__ = matlab_assign_lhs_from_rhs_(d0y_jy__,tmp_i8_index_lhs_, d0y_jml___.ravel()[tmp_i8_index_rhs_]);
        #end;%for l_val=0:l_max;
    #end;%for nl=0:numel(l_max_)-1;
    tmp_t = toc(tmp_t);
    if (flag_verbose>0): disp(sprintf(' %% d0y_jy__: %0.6fs',tmp_t)); #end;

    tmp_t = tic();
    tmp_i8_index_rhs_ = matlab_index_2d_0(n_polar_a_unique,index_return_,n_y_max,':');
    d0y_qy__ = torch.reshape(d0y_jy__.ravel()[tmp_i8_index_rhs_],mtr((n_q,n_y_max)));
    tmp_t = toc(tmp_t);
    if (flag_verbose>0): disp(sprintf(' %% d0y_jy__: %0.6fs',tmp_t)); #end;

    tmp_t = tic();
    m_val_y_ = torch.zeros(n_y_max).to(dtype=torch.int32);
    for nl in range(numel(l_max_)):
        l_max = int(l_max_[nl].item());
        for l_val in range(l_max+1):
            m_val_ = torch.arange(-l_val,+l_val+1).to(dtype=torch.int32);
            m_val_y_[l_val**2+l_val+m_val_] = m_val_;
        #end;%for l_val=0:l_max;
    #end;%for nl=0:numel(l_max_)-1;
    expimb_qy__ = torch.permute(torch.reshape(torch.exp(+i*(torch.reshape(m_val_y_,mtr((n_y_max,1))) * torch.reshape(k_p_azimu_b_q_,mtr((1,n_q))))),mtr((n_y_max,n_q))),mtr(mts((1,0)))).to(dtype=torch.complex64);
    tmp_t = toc(tmp_t);
    if (flag_verbose>0): disp(sprintf(' %% expimb_qy__: %0.6fs',tmp_t)); #end;

    tmp_t = tic();
    d0Y_qy__ = d0y_qy__ * expimb_qy__ / np.sqrt(4*pi);
    tmp_t = toc(tmp_t);
    if (flag_verbose>0): disp(sprintf(' %% d0Y_qy__: %0.6fs',tmp_t)); #end;

    tmp_t = tic();
    a_k_Y_yk__ = local_yk__from_yk_(n_k_p_r,l_max_,a_k_Y_yk_);
    a_k_p_qk_ = (mmmm( d0Y_qy__.to(dtype=torch.complex64) , a_k_Y_yk__.to(dtype=torch.complex64) )).ravel(); assert(numel(a_k_p_qk_)==n_qk);
    tmp_t = toc(tmp_t);
    if (flag_verbose>0): disp(sprintf(' %% a_k_p_qk_: %0.6fs',tmp_t)); #end;

    if (flag_verbose>0): disp(sprintf(' %% [finished %s] n_qk %d, n_y_sum %d',str_thisfunction,n_qk,sum(n_y_))); #end;
    return(
        a_k_p_qk_,
        sqrt_2lp1_,
        sqrt_2mp1_,
        sqrt_rat0_m_,
        sqrt_rat3_lm__,
        sqrt_rat4_lm__,
    );

