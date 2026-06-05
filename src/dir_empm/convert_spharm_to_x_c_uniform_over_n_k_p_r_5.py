from dir_empm.dir_matlab_macros import * ;
from dir_empm.sample_sphere_7 import sample_sphere_7 ;
from dir_empm.convert_spharm_to_k_p_uniform_over_n_k_p_r_5 import convert_spharm_to_k_p_uniform_over_n_k_p_r_5 ;
from dir_empm.xxnufft3d3 import xxnufft3d3 ;

def convert_spharm_to_x_c_uniform_over_n_k_p_r_5(
        flag_verbose=None,
        k_eq_d=None,
        n_k_p_r=None,
        k_p_r_=None,
        k_p_r_max=None,
        weight_3d_k_p_r_=None,
        l_max_=None,
        a_k_Y_yk_=None,
        half_diameter_x_c=None,
        n_x_g_pack=None,
        sqrt_2lp1_=None,
        sqrt_2mp1_=None,
        sqrt_rat0_m_=None,
        sqrt_rat3_lm__=None,
        sqrt_rat4_lm__=None,
        flag_u_vs_c=None,
):
    str_thisfunction = 'convert_spharm_to_x_c_uniform_over_n_k_p_r_5';
    if (flag_verbose>0): disp(sprintf(' %% [entering %s]',str_thisfunction)); #end;

    if isempty(half_diameter_x_c): half_diameter_x_c = 1.0; #end;
    if isempty(n_x_g_pack): n_x_g_pack = 64; #end;
    if isempty(flag_u_vs_c): flag_u_vs_c = 1; #end;

    diameter_x_c = 2.0*half_diameter_x_c;
    x_p_r_max = half_diameter_x_c;
    if flag_u_vs_c==0:
        n_x_c_pack = n_x_g_pack;
        x_c_0_ = torch.linspace(-x_p_r_max,+x_p_r_max,n_x_c_pack).to(dtype=torch.float32);
        x_c_1_ = torch.linspace(-x_p_r_max,+x_p_r_max,n_x_c_pack).to(dtype=torch.float32);
        x_c_2_ = torch.linspace(-x_p_r_max,+x_p_r_max,n_x_c_pack).to(dtype=torch.float32);
        x_c_2___, x_c_1___, x_c_0___ = torch.meshgrid(x_c_2_, x_c_1_, x_c_0_, indexing='ij') ; n_xxx_c = n_x_c_pack ** 3 ; #<-- reversed to match matlab. ;
        n_xxx_g = n_xxx_c;
        x_g_0___ = x_c_0___;
        x_g_1___ = x_c_1___;
        x_g_2___ = x_c_2___;
    #end;%if flag_u_vs_c==0:
    if flag_u_vs_c==1:
        n_x_u_pack = n_x_g_pack;
        x_u_0_ = torch.linspace(-x_p_r_max,+x_p_r_max,n_x_u_pack+1).to(dtype=torch.float32)[:-1];
        x_u_1_ = torch.linspace(-x_p_r_max,+x_p_r_max,n_x_u_pack+1).to(dtype=torch.float32)[:-1];
        x_u_2_ = torch.linspace(-x_p_r_max,+x_p_r_max,n_x_u_pack+1).to(dtype=torch.float32)[:-1];
        x_u_2___, x_u_1___, x_u_0___ = torch.meshgrid(x_u_2_, x_u_1_, x_u_0_, indexing='ij') ; n_xxx_u = n_x_u_pack ** 3 ; #<-- reversed to match matlab. ;
        n_xxx_g = n_xxx_u;
        x_g_0___ = x_u_0___;
        x_g_1___ = x_u_1___;
        x_g_2___ = x_u_2___;
    #end;%if flag_u_vs_c==1:
    a_x_g_xxx_ = torch.zeros(n_xxx_g).to(dtype=torch.complex64);
    
    tmp_t=tic();
    (
        n_qk,
        n_qk_csum_,
        k_p_r_qk_,
        k_p_azimu_b_qk_,
        k_p_polar_a_qk_,
        weight_3d_k_p_qk_,
        weight_shell_qk_,
        tmp_n_k_p_r,
        tmp_k_p_r_,
        tmp_weight_3d_k_p_r_,
        k_c_0_qk_,
        k_c_1_qk_,
        k_c_2_qk_,
    ) = sample_sphere_7(
        0*flag_verbose,
        k_p_r_max,
        k_eq_d,
        'L',
        1,
    )[:13] ; #%<-- sum(weight_3d_k_p_r_)*(4*pi) = (4/3)*pi*k_p_r_max^3 --> sum(weight_3d_k_p_r_) = (1/3)*k_p_r_max^3 ;
    tmp_t = toc(tmp_t);
    if (flag_verbose>0): disp(sprintf(' %% sample_sphere_7: %0.6fs',tmp_t)); #end;

    flag_continue = ( ( abs(tmp_n_k_p_r - n_k_p_r) < 1e-6 ) and ( fnorm(tmp_k_p_r_ - k_p_r_) < 1e-6 ) and ( fnorm(tmp_weight_3d_k_p_r_ - weight_3d_k_p_r_) < 1e-6 ) );
    if flag_continue==0:
        disp(sprintf(' %% Warning, incompatible grids generated in %s',str_thisfunction));
    #end;%if flag_continue==0;

    #%%%%%%%%%%%%%%%%;
    if flag_continue==1:
    #%%%%%%%%%%%%%%%%;
        if (flag_verbose>0): disp(sprintf(' %% compatible grids generated in %s',str_thisfunction)); #end;
        #%%%%%%%%;
        tmp_t=tic();
        (
            a_k_p_qk_,
            sqrt_2lp1_,
            sqrt_2mp1_,
            sqrt_rat0_m_,
            sqrt_rat3_lm__,
            sqrt_rat4_lm__,
        ) = convert_spharm_to_k_p_uniform_over_n_k_p_r_5(
            flag_verbose,
            n_qk,
            n_qk_csum_,
            k_p_r_qk_,
            k_p_azimu_b_qk_,
            k_p_polar_a_qk_,
            weight_3d_k_p_qk_,
            weight_shell_qk_,
            n_k_p_r,
            k_p_r_,
            weight_3d_k_p_r_,
            l_max_,
            a_k_Y_yk_,
            sqrt_2lp1_,
            sqrt_2mp1_,
            sqrt_rat0_m_,
            sqrt_rat3_lm__,
            sqrt_rat4_lm__,
        )[:6];
        tmp_t = toc(tmp_t);
        if (flag_verbose>0): disp(sprintf(' %% a_k_Y_yk_ --> a_k_p_qk_ time %0.2fs',tmp_t)); #end;
        #%%%%%%%%;
        tmp_t=tic();
        eta = pi/k_p_r_max;
        a_x_g_xxx_ = xxnufft3d3(
            n_qk,
            2*pi*k_c_0_qk_*eta,
            2*pi*k_c_1_qk_*eta,
            2*pi*k_c_2_qk_*eta,
            a_k_p_qk_*weight_3d_k_p_qk_,
            +1,
            1e-12,
            n_xxx_g,
            x_g_0___.ravel()/eta,
            x_g_1___.ravel()/eta,
            x_g_2___.ravel()/eta,
        );
        tmp_t = toc(tmp_t);
        if (flag_verbose>0): disp(sprintf(' %% a_k_p_qk_ --> a_x_g_xxx_ time %0.2fs',tmp_t)); #end;
        #%%%%%%%%%%%%%%%%;
        # end;%if flag_continue==1;
        #%%%%%%%%%%%%%%%%;

    if (flag_verbose>0): disp(sprintf(' %% [finished %s]',str_thisfunction)); #end;
    
    return(
        a_x_g_xxx_,
        a_k_p_qk_,
        sqrt_2lp1_,
        sqrt_2mp1_,
        sqrt_rat0_m_,
        sqrt_rat3_lm__,
        sqrt_rat4_lm__,
    );

