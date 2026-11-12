from dir_empm.dir_matlab_macros import * ;
from dir_empm.local_yk__from_yk_ import local_yk__from_yk_ ;

def principled_marching_cost_matrix_3(
        n_k_p_r=None,
        weight_k_p_r_=None,
        l_max=None,
        a_k_Y_yk__=None,
        CTF_k_p_r_kk__=None,
):
    #%%%%%%%%;
    #% more efficient calculation of volumetric cost. ;
    #% scales by inverse-standard-deviations. ;
    #% scales by cross-correlation of CTF_k_p_r_. ;
    #%%%%%%%%;
    flag_verbose = 0;
    str_thisfunction = 'principled_marching_cost_matrix_3';
    
    if (flag_verbose>0): disp(sprintf(' %% [entering %s]',str_thisfunction)); #end;

    if isempty(CTF_k_p_r_xcor_kk__): CTF_k_p_r_xcor_kk__ = torch.ones(mtr((n_k_p_r,n_k_p_r))).to(dtype=torch.float32); #end;
    if isempty(delta_sigma): delta_sigma = 0; #end;
    if isempty(pm_delta_integral_tolerance): pm_delta_integral_tolerance = 1e-3; #end;
    if (pm_delta_integral_tolerance<=0): pm_delta_integral_tolerance = 1e-3; #end;

    if (isempty(n_molecule)): n_molecule = 1; molecule_density_ = torch.ones(n_molecule).to(dtype=torch.float32); #end;
    if (n_molecule==0): n_molecule = 1; molecule_density_ = torch.ones(n_molecule).to(dtype=torch.float32); #end;
    #%%%%%%%%;
    tmp_t = tic();
    n_y_max = (1+l_max)**2;
    n_m_max = 1+2*l_max;
    n_polar_a = n_m_max; polar_a_ = matlab_linspace(-pi,pi,n_polar_a+1,dtype=torch.float32); polar_a_ = polar_a_[0:n_polar_a].ravel();
    weight_so3 = (2*pi)*(2*pi)*4; #%<-- total volume of so3. ;
    #%%%%%%%%;
    X_weight_r_ = torch.zeros(mtr((n_k_p_r,1))).to(dtype=float32);
    for nk_p_r in range(n_k_p_r):
        X_weight_r_[nk_p_r] = np.sqrt(weight_k_p_r_[nk_p_r].item());
    #end;%for nk_p_r=0:n_k_p_r-1;
    #%%%%%%%%;
    X_ori_kk__ = torch.zeros(mtr((n_k_p_r,n_k_p_r))).to(dtype=torch.float32);
    for nk_p_r_0 in range(n_k_p_r):
        X_weight_r_0 = X_weight_r_[nk_p_r_0].item(); #%<-- volume element = inverse-variance associated with nk_p_r_0. ;
        for nk_p_r_1 in range(nk_p_r_0,n_k_p_r):
            X_weight_r_1 = X_weight_r_[nk_p_r_1].item(); #%<-- volume element = inverse-variance associated with nk_p_r_1. ;
            X_weight_r_01 = X_weight_r_0*X_weight_r_1; #%<-- product of inverse-standard-deviations associated with nk_p_r_0 and nk_p_r_1. ;
            X_ori_kk__[nk_p_r_1,nk_p_r_0] = register_spharm_to_spharm_2(flag_verbose,1,1,1,l_max,a_k_Y_yk__[nk_p_r_0,:],a_k_Y_yk__[nk_p_r_1,:]) * X_weight_r_01 * CTF_k_p_r_kk__[nk_p_r_1,nk_p_r_0].item();
            X_ori_kk__[nk_p_r_0,nk_p_r_1] = np.conj(X_ori_kk__[nk_p_r_1,nk_p_r_0].item());
            X_tau_kk__[nk_p_r_1,nk_p_r_0] = (4*pi)**2 * np.conj(a_k_Y_yk__[nk_p_r_0,0].item())*a_k_Y_yk__[nk_p_r_1,0].item() * X_weight_r_01 * CTF_k_p_r_kk__[nk_p_r_1,nk_p_r_0].item();
            X_tau_kk__[nk_p_r_0,nk_p_r_1] = np.conj(X_tau_kk__[nk_p_r_1,nk_p_r_0].item());
        #end;%for nk_p_r_1=nk_p_r_0:n_k_p_r-1;
    #end;%for nk_p_r_0=0:n_k_p_r-1;
    #%%%%%%%%;
    X_kk__ = torch.real(X_ori_kk__)*weight_so3 - torch.real(X_tau_kk__);
    #%%%%%%%%;

    n_y_sum = int(torch.sum(n_y_).item());
    n_y_csum_ = cumsum_0(n_y_);
    l_max_max = int(torch.max(l_max_).item());
    m_max_ = torch.arange(-l_max_max,+l_max_max+1).to(dtype=torch.int32);
    n_m_max = 1+2*l_max_max;
    n_polar_a = n_m_max; polar_a_ = torch.linspace(-pi,pi,n_polar_a+1).to(dtype=torch.float32); polar_a_ = polar_a_[0:n_polar_a].ravel();
    weight_so3 = (2*pi)*(2*pi)*4; #%<-- total volume of so3. ;
    a_k_Y_ykv__ = torch.reshape(a_k_Y_ykv__,mtr((n_y_sum,n_molecule))).to(dtype=torch.complex64); #<-- if n_molecule==1. ;
    a_k_Y_ykv___ = torch.zeros(mtr((n_y_max,n_k_p_r,n_molecule))).to(dtype=torch.complex64);
    for nmolecule in range(n_molecule):
        a_k_Y_ykv___[nmolecule,:,:] = torch.reshape(local_yk__from_yk_(n_k_p_r,l_max_,a_k_Y_ykv__[nmolecule,:]),mtr((n_y_max,n_k_p_r)));
    #end;%for nmolecule=0:n_molecule-1;
    tmp_t = toc(tmp_t);
    if (flag_verbose>1): disp(sprintf(' %% a_k_Y_ykv___: %0.3fs',tmp_t)); #end;
    if (isempty(molecule_density_)): molecule_density_ = torch.ones(n_molecule).to(dtype=torch.float32); #end;
    molecule_density_ = molecule_density_/np.maximum(1e-12,torch.sum(molecule_density_).item());
    #%%%%%%%%;

    if (flag_verbose>0): disp(sprintf(' %% [finished %s]',str_thisfunction)); #end;
    return(
        X_kk__,
        X_weight_r_,
        X_ori_kk__,
        X_tau_kk__,
        weight_so3,
        n_m_max,
        polar_a_,
    );
