from dir_empm.dir_matlab_macros import * ;

def cg_rhs_2(
        n_M = None,
        n_w = None,
        viewing_polar_a_M_ = None,
        viewing_azimu_b_M_ = None,
        viewing_gamma_z_M_ = None,
        flag_cleanup = None,
):
    str_thisfunction = 'cg_rhs_2';
    flag_verbose=0;
    ################################################################;
    if (flag_verbose>0): print(f" %% [entering {str_thisfunction}]");
    ################################################################;

    if viewing_gamma_z_M_ is None or numel(viewing_gamma_z_M_) == 0: viewing_gamma_z_M_ = torch.zeros(n_M).to(dtype=torch.float32);
    if flag_cleanup is None: flag_cleanup = 1;
    flag_1 = 1;
    flag_d = 0;
    flag_dd = 0;
    tolerance_cg_rhs = 1e-16;
    tolerance_pole = np.sqrt(tolerance_cg_rhs);
    tolerance_cg_rhs_upb = 1.0/np.sqrt(1e-16);

    if flag_1:
        sa_wM__ = torch.tile(torch.reshape(torch.sin(viewing_polar_a_M_),mtr((1,n_M))), mtr((n_w,1))) ;
        ca_wM__ = torch.tile(torch.reshape(torch.cos(viewing_polar_a_M_),mtr((1,n_M))), mtr((n_w,1))) ;
        sb_wM__ = torch.tile(torch.reshape(torch.sin(viewing_azimu_b_M_),mtr((1,n_M))), mtr((n_w,1))) ;
        cb_wM__ = torch.tile(torch.reshape(torch.cos(viewing_azimu_b_M_),mtr((1,n_M))), mtr((n_w,1))) ;
        viewing_gamma_z_M_ = torch.reshape(viewing_gamma_z_M_,mtr((1,n_M))) ;
        inplane_gamma_z_w_ = torch.reshape((2*pi*torch.arange(n_w) / np.maximum(1,n_w)),mtr((n_w,1)));
        combine_gamma_z_wM__ = torch.tile(inplane_gamma_z_w_, mtr((1,n_M))) - torch.tile(viewing_gamma_z_M_, mtr((n_w,1))) ;
        sc_wM__ = torch.sin(combine_gamma_z_wM__) ;
        cc_wM__ = torch.cos(combine_gamma_z_wM__) ;
        k_c_0_wM__ = cb_wM__*ca_wM__*cc_wM__ - sb_wM__*sc_wM__ ;
        k_c_1_wM__ = sb_wM__*ca_wM__*cc_wM__ + cb_wM__*sc_wM__ ;
        k_c_2_wM__ = -sa_wM__*cc_wM__ ;
        k_p_r01_bkp_wM__ = torch.sqrt(k_c_0_wM__**2 + k_c_1_wM__**2) ;
        k_p_r01_wM__ = torch.sqrt((ca_wM__*cc_wM__)**2 + sc_wM__**2) ;
        fnorm_disp(flag_verbose,'k_p_r01_wM__',k_p_r01_wM__,'k_p_r01_bkp_wM__',k_p_r01_bkp_wM__,' %<-- should be <1e-12');
        k_p_polar_a_wM__ = torch.atan2(k_p_r01_wM__, k_c_2_wM__) ;
        k_p_azimu_b_wM__ = torch.atan2(k_c_1_wM__, k_c_0_wM__) ;
        for nM in range(n_M):
            if np.abs(periodize(viewing_polar_a_M_[nM] - pi / 2, -pi / 2, pi / 2)) < tolerance_pole:
                tmp_index_0_pos_ = efind(k_c_0_wM__[nM,:] > +tolerance_pole) ;
                tmp_index_0_neg_ = efind(k_c_0_wM__[nM,:] < -tolerance_pole) ;
                flag_0_use = fnorm(k_c_0_wM__[nM,:]) >= 0.25 ;
                tmp_index_1_pos_ = efind(k_c_1_wM__[nM,:] > +tolerance_pole) ;
                tmp_index_1_neg_ = efind(k_c_1_wM__[nM,:] < -tolerance_pole) ;
                flag_1_use = fnorm(k_c_1_wM__[nM,:]) >= 0.25 ;
                tmp_index_2_pos_ = efind(torch.abs(k_c_2_wM__[nM,:] - 1) < tolerance_pole) ;
                tmp_index_2_neg_ = efind(torch.abs(k_c_2_wM__[nM,:] + 1) < tolerance_pole) ;
                assert flag_0_use or flag_1_use ;
                if flag_0_use:
                    tmp_azimu_b_pos = torch.mean(k_p_azimu_b_wM__[nM,tmp_index_0_pos_]).item() ;
                    tmp_azimu_b_neg = torch.mean(k_p_azimu_b_wM__[nM,tmp_index_0_neg_]).item() ;
                else:
                    tmp_azimu_b_pos = torch.mean(k_p_azimu_b_wM__[nM,tmp_index_1_pos_]).item() ;
                    tmp_azimu_b_neg = torch.mean(k_p_azimu_b_wM__[nM,tmp_index_1_neg_]).item() ;
                    tmp_azimu_b_mid = periodize(0.5*(tmp_azimu_b_pos + tmp_azimu_b_neg), 0, 2*pi) ;
                    k_p_azimu_b_wM__[nM,tmp_index_2_pos_] = periodize(tmp_azimu_b_mid + 0*pi, 0, 2*pi) ;
                    k_p_azimu_b_wM__[nM,tmp_index_2_neg_] = periodize(tmp_azimu_b_mid + 1*pi, 0, 2*pi) ;
                #end;%ifelse;
            #end;%if;
        #end;%for;
    #end;%if flag_1;

    if flag_1:
        if not np.isfinite(k_p_polar_a_wM__.ravel().numpy()).all():
            print(f"Warning, k_p_polar_a_wM__ not finite in {str_thisfunction}")
        if not np.isfinite(k_p_azimu_b_wM__.ravel().numpy()).all():
            print(f"Warning, k_p_azimu_b_wM__ not finite in {str_thisfunction}")
        if not np.isfinite(k_c_0_wM__.ravel().numpy()).all():
            print(f"Warning, k_c_0_wM__ not finite in {str_thisfunction}")
        if not np.isfinite(k_c_1_wM__.ravel().numpy()).all():
            print(f"Warning, k_c_1_wM__ not finite in {str_thisfunction}")
        if not np.isfinite(k_c_2_wM__.ravel().numpy()).all():
            print(f"Warning, k_c_2_wM__ not finite in {str_thisfunction}")
        if not np.isfinite(k_p_r01_wM__.ravel().numpy()).all():
            print(f"Warning, k_p_r01_wM__ not finite in {str_thisfunction}")
    #end;%if flag_1;

    if flag_1:
        return(
            k_p_polar_a_wM__,
            k_p_azimu_b_wM__,
            k_c_0_wM__,
            k_c_1_wM__,
            k_c_2_wM__,
            k_p_r01_wM__,
        );
    #end;%if flag_1;
    
    ################################################################;
    if (flag_verbose>0): print(f" %% [finished {str_thisfunction}]");
    ################################################################;
