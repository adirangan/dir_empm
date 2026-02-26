from dir_empm.dir_matlab_macros import * ;
from scipy.special import iv ;

def pm_delta_integral_3(
        n_k_p_r=None,
        k_p_r_=None,
        delta_sigma=None,
        l_max=None,
        pm_delta_integral_tolerance=None,
):
    #% calculates integrals associated with delta: ;
    #%
    #% Input: ;
    #% n_k_p_r: integer number of k-values. ;
    #% k_p_r_: real array of size n_k_p_r. k_values. ;
    #% delta_sigma: real standard-deviation for isotropic gaussian distribution fo delta-values. ;
    #% l_max: maximum l_val to use in expansion. ;
    #% pm_delta_integral_tolerance: real tolerance for warning (default 1e-2). ;
    #%
    #% Output: ;
    #% I_pos__ = real array of size (n_k_p_r,n_k_p_r). ;
    #% I_pos__(nk_p_r_0,nk_p_r_1) = \int dphi * 1/twopi/delta_sigma^2 * exp(-delta^2/2/delta_sigma^2) * delta * d_delta * d_omega * exp(+i*twopi*k_0*delta*cos(phi - omega)) * exp(-i*twopi*k_1*delta*cos(phi - omega)). ;
    #% where k_0 = k_p_r_(nk_p_r_0) and k_1 = k_p_r_(nk_p_r_1). ;
    #% I_neg_ = real array of size n_k_p_r. ;
    #% I_neg_(nk_p_r) = \int dphi * 1/twopi/delta_sigma^2 * exp(-delta^2/2/delta_sigma^2) * delta * d_delta * d_omega * exp(\pm i*twopi*k*delta*cos(phi - omega)) ;
    #% where k = k_p_r_(nk_p_r). ;

    flag_verbose=0;
    str_thisfunction = 'pm_delta_integral_3';
    if (flag_verbose>0): disp(sprintf(' %% [entering %s]',str_thisfunction)); #end;

    if isempty(pm_delta_integral_tolerance): pm_delta_integral_tolerance = 1e-3; #end;
    if (pm_delta_integral_tolerance<=0): pm_delta_integral_tolerance = 1e-3; #end;
    #%%%%%%%%;
    #% Note that, defining: ;
    #% I_neg = @(ks) 2*pi*(0.5*ks*sqrt(pi/2)*exp(-ks*ks/4)*(besseli(-0.5,ks*ks/4) - besseli(+0.5,ks*ks/4)));
    #% we see that I_neg is within ~1e-6 of 2*pi for ks<=1e-3. ;
    #%%%%%%%%;

    twopi = 2*pi;

    ks_ = twopi*k_p_r_*delta_sigma;
    ksks4_ = ks_**2/4;
    I_neg_ = torch.zeros(n_k_p_r).to(dtype=torch.float32);
    tmp_index_ = efind(ks_> pm_delta_integral_tolerance);
    I_neg_[tmp_index_] = twopi*(0.5*ks_[tmp_index_].ravel()*np.sqrt(pi/2)*torch.exp(-ksks4_[tmp_index_])*(iv(-0.5,ksks4_[tmp_index_]) - iv(+0.5,ksks4_[tmp_index_])));
    tmp_index_ = efind(ks_<=pm_delta_integral_tolerance);
    I_neg_[tmp_index_] = twopi;

    I_pos__ = torch.zeros(mtr((n_k_p_r,n_k_p_r))).to(dtype=torch.float32);
    k_p_r_1__,k_p_r_0__ = torch.meshgrid(k_p_r_,k_p_r_,indexing='ij'); #<-- reversed to match matlab. ;
    tmp_z__ = twopi**2 * k_p_r_0__*k_p_r_1__ * delta_sigma**2;
    I_pos__ = twopi * torch.exp(-twopi**2*0.5*(k_p_r_0__**2 + k_p_r_1__**2)*delta_sigma**2) * torch.exp(tmp_z__);

    if (flag_verbose>0): disp(sprintf(' %% [finished %s]',str_thisfunction)); #end;
    return(
        I_pos__,
        I_neg_,
    );
