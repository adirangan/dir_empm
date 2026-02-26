from dir_empm.dir_matlab_macros import * ;
from dir_empm.local_yk__from_yk_ import local_yk__from_yk_ ;
from dir_empm.pm_delta_integral_3 import pm_delta_integral_3 ;

def principled_marching_cost_matrix_7(
        n_k_p_r=None,
        k_p_r_=None,
        weight_k_p_r_=None,
        l_max_=None,
        n_molecule=None,
        molecule_density_=None,
        a_k_Y_ykv__=None,
        CTF_k_p_r_xcor_kk__=None,
        delta_sigma=None,
        pm_delta_integral_tolerance=None,
):
    #%%%%%%%%;
    #% updated version of principled_marching_cost_matrix_6.m. ;
    #% more efficient calculation of volumetric cost. ;
    #% scales by inverse-standard-deviations. ;
    #% scales by cross-correlation of CTF_k_p_r_. ;
    #% integrates over multiple molecules. ;
    #% accounts for (isotropic gaussian) distribution of translations with standard-deviation delta_sigma. ;
    #% allows for delta_sigma==0. ;
    #%%%%%%%%;
    flag_verbose = 0;
    str_thisfunction = 'principled_marching_cost_matrix_7';
    
    if (flag_verbose>0): disp(sprintf(' %% [entering %s]',str_thisfunction)); #end;

    if isempty(CTF_k_p_r_xcor_kk__): CTF_k_p_r_xcor_kk__ = torch.ones(mtr((n_k_p_r,n_k_p_r))).to(dtype=torch.float32); #end;
    if isempty(delta_sigma): delta_sigma = 0; #end;
    if isempty(pm_delta_integral_tolerance): pm_delta_integral_tolerance = 1e-3; #end;
    if (pm_delta_integral_tolerance<=0): pm_delta_integral_tolerance = 1e-3; #end;

    if (isempty(n_molecule)): n_molecule = 1; molecule_density_ = torch.ones(n_molecule).to(dtype=torch.float32); #end;
    if (n_molecule==0): n_molecule = 1; molecule_density_ = torch.ones(n_molecule).to(dtype=torch.float32); #end;
    #%%%%%%%%;
    tmp_t = tic();
    n_y_ = (1+l_max_)**2;
    n_y_max = int(torch.max(n_y_).item());
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

    #%%%%%%%%;
    tmp_t = tic();
    I_pos__,I_neg_ = pm_delta_integral_3(n_k_p_r,k_p_r_,delta_sigma,l_max_max,pm_delta_integral_tolerance);
    I_pos__ = I_pos__/(2*pi);
    I_neg__ = (torch.reshape(I_neg_,mtr((n_k_p_r,1)))*torch.reshape(I_neg_,mtr((1,n_k_p_r))))/(2*pi)**2;
    tmp_t = toc(tmp_t);
    if (flag_verbose>1): disp(sprintf(' %% I_pos__ and I_neg_: %0.3fs',tmp_t)); #end;
    #%%%%%%%%;
    tmp_t = tic();
    X_weight_r_ = torch.sqrt(weight_k_p_r_);
    tmp_t = toc(tmp_t); 
    if (flag_verbose>1): disp(sprintf(' %% X_weight_r_: %0.3fs',tmp_t)); #end;
    #%%%%%%%%;
    tmp_t = tic();
    mu_vv__ = torch.reshape(molecule_density_,mtr((n_molecule,1)))*torch.reshape(molecule_density_,mtr((1,n_molecule)));
    w_kk__ = torch.reshape(X_weight_r_,mtr((n_k_p_r,1)))*torch.reshape(X_weight_r_,mtr((1,n_k_p_r)));
    X_ori_kk__=torch.reshape(torch.sum(torch.reshape(mu_vv__,mtr((n_1,n_1,n_1,n_molecule,n_molecule)))*torch.reshape(w_kk__,mtr((n_1,n_k_p_r,n_k_p_r,n_1,n_1)))*torch.reshape(CTF_k_p_r_xcor_kk__,mtr((n_1,n_k_p_r,n_k_p_r,n_1,n_1)))*torch.sum(torch.conj(torch.reshape(a_k_Y_ykv___,mtr((n_y_max,n_k_p_r,n_1,n_molecule,n_1))))*torch.reshape(a_k_Y_ykv___,mtr((n_y_max,n_1,n_k_p_r,n_1,n_molecule))),dim=4-0,keepdim=True),dim=(4-3,4-4),keepdim=True),mtr((n_k_p_r,n_k_p_r)));
    X_tau_kk__=(4*pi)**2*torch.reshape(torch.sum(torch.reshape(mu_vv__,mtr((n_1,n_1,n_1,n_molecule,n_molecule)))*torch.reshape(w_kk__,mtr((n_1,n_k_p_r,n_k_p_r,n_1,n_1)))*torch.reshape(CTF_k_p_r_xcor_kk__,mtr((n_1,n_k_p_r,n_k_p_r,n_1,n_1)))*torch.sum(torch.conj(torch.reshape(a_k_Y_ykv___[:,:,0],mtr((n_1,n_k_p_r,n_1,n_molecule,n_1))))*torch.reshape(a_k_Y_ykv___[:,:,0],mtr((n_1,n_1,n_k_p_r,n_1,n_molecule))),dim=4-0,keepdim=True),dim=(4-3,4-4),keepdim=True),mtr((n_k_p_r,n_k_p_r)));
    #%%%%%%%%;
    X_kk__ = torch.real(X_ori_kk__)*weight_so3*I_pos__ - torch.real(X_tau_kk__)*I_neg__;
    #%%%%%%%%;
    tmp_t = toc(tmp_t); 
    if (flag_verbose>1): disp(sprintf(' %% X_kk__: %0.3fs',tmp_t)); #end;

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
