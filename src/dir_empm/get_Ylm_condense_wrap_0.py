from dir_empm.dir_matlab_macros import * ;
from dir_empm.ylgndr_2 import ylgndr_2 ;
from dir_empm.get_Ylm__2 import get_Ylm__2 ;

def get_Ylm_condense_wrap_0(
  flag_verbose=None,
  n_k_all=None,
  n_k_all_csum_=None,
  k_p_azimu_b_all_=None,
  k_p_polar_a_all_=None,
  n_k_p_r=None,
  l_max_=None,
  ):
  str_thisfunction = 'get_Ylm_condense_wrap_0' ;

  n_lm_k_ = (l_max_+1)**2 ;
  n_k_per_shell_k_ = torch.zeros(n_k_p_r).to(dtype=torch.int32) ;
  index_sub_ka__ = cell(n_k_p_r); #<-- list of n_k_p_r empty lists. ;
  k_p_azimu_b_sub_ka__ = cell(n_k_p_r); #<-- list of n_k_p_r empty lists. ;
  k_p_polar_a_sub_ka__ = cell(n_k_p_r); #<-- list of n_k_p_r empty lists. ;

  for nk_p_r in range(n_k_p_r):
    n_k_all_csum = int(n_k_all_csum_[nk_p_r].item());
    if flag_verbose > 1: print(f' %% nk_p_r {nk_p_r}/{n_k_p_r}, n_k_all_csum {n_k_all_csum} --> {n_k_all_csum/n_k_all:.2f}%') ;
    if (nk_p_r<n_k_p_r-1):
      n_k_per_shell = int(n_k_all_csum_[nk_p_r+1].item()) - int(n_k_all_csum_[nk_p_r].item()); 
    else:
      n_k_per_shell = n_k_all - n_k_all_csum ;
    #end;%if;
    index_sub_ = torch.arange(n_k_all_csum,n_k_all_csum+n_k_per_shell).to(dtype=torch.int32) ;
    k_p_azimu_b_sub_ = k_p_azimu_b_all_[index_sub_];
    k_p_polar_a_sub_ = k_p_polar_a_all_[index_sub_];
    n_k_per_shell_k_[nk_p_r] = n_k_per_shell;
    index_sub_ka__[nk_p_r] = index_sub_; #<-- enter into cell-array. ;
    k_p_azimu_b_sub_ka__[nk_p_r] = k_p_azimu_b_sub_; #<-- enter into cell-array. ;
    k_p_polar_a_sub_ka__[nk_p_r] = k_p_polar_a_sub_; #<-- enter into cell-array. ;
  #end;%for nk_p_r=0:n_k_p_r-1;

  np_u_n_k_per_shell_,_,np_index_nu_n_k_per_shell_from_nk_p_r_ = np.unique(n_k_per_shell_k_.numpy().ravel(),return_index=True,return_inverse=True);
  u_n_k_per_shell_ = torch.tensor(np_u_n_k_per_shell_).to(dtype=torch.int32);
  n_u_n_k_per_shell = numel(u_n_k_per_shell_);
  index_nu_n_k_per_shell_from_nk_p_r_ = torch.tensor(np_index_nu_n_k_per_shell_from_nk_p_r_).to(dtype=torch.int32);
  index_k_per_shell_uka__ = cell(n_u_n_k_per_shell); #<-- list of n_u_n_k_per_shell empty lists. ;
  l_max_uk_ = torch.zeros(n_u_n_k_per_shell).to(dtype=torch.int32);
  for nu_n_k_per_shell in range(n_u_n_k_per_shell):
    u_n_k_per_shell = int(u_n_k_per_shell_[nu_n_k_per_shell].item());
    index_k_per_shell_uka__[nu_n_k_per_shell] = efind(n_k_per_shell_k_==u_n_k_per_shell);
    l_max_uk_[nu_n_k_per_shell] = int(torch.max(l_max_[index_k_per_shell_uka__[nu_n_k_per_shell]]).item());
  #end;%for nu_n_k_per_shell=0:n_u_n_k_per_shell-1;

  n_lm_uk_ = (l_max_uk_+1)**2;
  n_k_per_shell_uk_ = torch.zeros(n_u_n_k_per_shell).to(dtype=torch.int32);
  k_p_azimu_b_sub_uka__ = cell(n_u_n_k_per_shell); #<-- list of n_u_n_k_per_shell empty lists. ;
  k_p_polar_a_sub_uka__ = cell(n_u_n_k_per_shell); #<-- list of n_u_n_k_per_shell empty lists. ;
  for nu_n_k_per_shell in range(n_u_n_k_per_shell):
    u_n_k_per_shell = int(u_n_k_per_shell_[nu_n_k_per_shell].item());
    index_k_per_shell_uka__[nu_n_k_per_shell] = efind(n_k_per_shell_k_==u_n_k_per_shell);
    nk_p_r_0 = int(index_k_per_shell_uka__[nu_n_k_per_shell][0].item());
    k_p_azimu_b_sub_uka__[nu_n_k_per_shell] = k_p_azimu_b_sub_ka__[nk_p_r_0];
    k_p_polar_a_sub_uka__[nu_n_k_per_shell] = k_p_polar_a_sub_ka__[nk_p_r_0];
  #end;%for nu_n_k_per_shell=0:n_u_n_k_per_shell-1;

  if flag_verbose > 1: print(f' %% [entering {str_thisfunction}] n_k_all {n_k_all}, n_lm_sum {torch.sum(n_lm_k_).item()}, n_u_n_k_per_shell {n_u_n_k_per_shell}') ;

  Ylm_uklma___ = cell(n_u_n_k_per_shell); #<-- list of n_u_n_k_per_shell empty lists. ;
  for nu_n_k_per_shell in range(n_u_n_k_per_shell):
    k_p_azimu_b_sub_ = k_p_azimu_b_sub_uka__[nu_n_k_per_shell];
    k_p_polar_a_sub_ = k_p_polar_a_sub_uka__[nu_n_k_per_shell];
    n_k_per_shell = numel(k_p_polar_a_sub_);
    l_max = int(l_max_uk_[nu_n_k_per_shell].item());
    n_l_max = l_max+1; flag_flip=0;
    n_lm = (l_max+1)**2;
    if flag_verbose: print(f' %% nu_n_k_per_shell {nu_n_k_per_shell}/{n_u_n_k_per_shell}: Ylm_lma__ {n_lm*n_k_per_shell*8/1e9:.2f}GB') ;
    tmp_t = timeit.default_timer();
    (
      _,
      Ylm_sub__,
    ) = get_Ylm__2(
      [],
      n_l_max,
      torch.arange(l_max+1).to(dtype=torch.int32),
      n_k_per_shell,
      k_p_azimu_b_sub_,
      k_p_polar_a_sub_,
      flag_flip,
    )[:2];
    Ylm_lma__ = torch.zeros(mtr((n_lm,n_k_per_shell)),dtype=torch.complex64);
    ix1=0;
    for l_val in range(l_max+1):
      n_m = 2*l_val+1;
      for nm in range(n_m):
        m_val = nm - l_val;
        tmp_i8_index_lhs_ = matlab_index_2d_0(n_lm,ix1,n_k_per_shell,':');
        tmp_i8_index_rhs_ = matlab_index_2d_0(1+2*l_val,nm,n_k_per_shell,':');
        Ylm_lma__.flatten()[tmp_i8_index_lhs_] = Ylm_sub__[l_val].flatten()[tmp_i8_index_rhs_];
        ix1 = ix1+1;
      #end;%for nm=0:n_m-1;
    #end;%for l_val=0:l_max;
    tmp_t = timeit.default_timer()-tmp_t;
    if flag_verbose: print(f' %% nu_n_k_per_shell {nu_n_k_per_shell}/{n_u_n_k_per_shell} Ylm_lma__: {tmp_t:.6f}s') ;
    Ylm_uklma___[nu_n_k_per_shell] = Ylm_lma__;
    del Ylm_sub__;
  #end;%for nu_n_k_per_shell=0:n_u_n_k_per_shell-1;

  if flag_verbose: print(f' %% [finished {str_thisfunction}] n_k_all {n_k_all}, n_lm_sum {torch.sum(n_lm_k_).item()}, n_u_n_k_per_shell {n_u_n_k_per_shell}') ;

  return(
    Ylm_uklma___,
    k_p_azimu_b_sub_uka__,
    k_p_polar_a_sub_uka__,
    l_max_uk_,
    index_nu_n_k_per_shell_from_nk_p_r_,
    index_k_per_shell_uka__,
  );
