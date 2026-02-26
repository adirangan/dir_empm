from dir_empm.dir_matlab_macros import * ;
from dir_empm.ylgndr_2 import ylgndr_2 ;

def wignerd_c(
    parameter=None,
    n_l=None,
    beta=None,
    V_lmm___=None,
    L_lm__=None,
):
  #% Generates wigner-d matrices up to n_l ;
  #% See Gumerov and Duraiswami 2014. ; %<-- I could not get this to work. ;
  #% Try Feng, Wang, Yang, Jin, 2015. ; %<-- This works. ;
  str_thisfunction = 'wignerd_c' ;

  if not isinstance(parameter, dict): parameter = {'type': 'parameter'} ;
  if 'flag_verbose' not in parameter: parameter['flag_verbose'] = 0 ;
  flag_verbose = parameter['flag_verbose'] ;
  if 'flag_d' not in parameter: parameter['flag_d'] = 1 ;
  flag_d = parameter['flag_d'] ;
  if 'flag_dd' not in parameter: parameter['flag_dd'] = 1 ;
  flag_dd = parameter['flag_dd'] ;

  if (flag_verbose>0): print(f" %% [entering {str_thisfunction}]") ;

  flag_precomputation = (
    V_lmm___ is None or
    L_lm__ is None or
    0 ) ;

  flag_d0 = 1;
  flag_d1 = flag_d;
  flag_d2 = flag_dd;

  l_max = n_l;
  l_val_ = torch.arange(l_max+1).to(dtype=torch.int32);
  m_val_ = torch.arange(l_max+1).to(dtype=torch.int32);
  m_max_ = torch.arange(-l_max,+l_max+1).to(dtype=torch.int32);

  if flag_precomputation:
    if (flag_verbose>0): print(f' %% flag_precomputation: {flag_precomputation} l_max {l_max}');
    V_lmm___ = cell(1+l_max); #<-- cell array. ;
    L_lm__ = cell(1+l_max); #<-- cell array. ;
    for l_val in range(l_max+1):
      m_val_ = torch.arange(-l_val,+l_val+1).to(dtype=torch.int32);
      n_m_val = numel(m_val_);
      X_ = torch.sqrt( (l_val + m_val_) * (1 + l_val - m_val_) ); 
      X_frwd_ = -X_ ;
      X_back_ = +torch.flip(X_,(0,));
      S__ = torch.sparse.spdiags(torch.reshape(torch.concatenate((X_frwd_,X_back_),0),mtr((n_m_val,2))),torch.tensor([+1,-1]),mtr((n_m_val,n_m_val))).to_dense().T; #<-- I believe that, as of 20251021, the torch.sparse.spdiags output is indexed as the transpose of the matlab-compatible indexing (hence the final transposition). ;
      L_m_,V_mm__ = torch.linalg.eig(S__.T) ; V_mm__ = V_mm__.T; #<-- I believe that, as of 20251021, the torch.linalg.eig input and output are indexed as the transpose of the matlab-compatible indexing (hence the pre- and pos-transposition). Note also that, while the eigenstructure computed by torch.linalg.eig matches that of matlab eig, the ordering of the eigenvalues and the phasing of the eigenvectors (i.e., the entries of L_m_ and columns of V_mm__) will not necessarily match. ;
      V_lmm___[l_val] = V_mm__;
      L_lm__[l_val] = L_m_;
      del V_mm__ ; del L_m_ ; del S__ ; del X_ ;
    #end;%for l_val=1:l_max;
  #end;%if flag_precomputation;

  d0W_ = None; d1W_ = None;  d2W_ = None;
  if (flag_d0): d0W_ = cell(1+l_max); d0W_[0] = torch.ones(mtr((1,1))).to(dtype=torch.float32);
  if (flag_d1): d1W_ = cell(1+l_max); d1W_[0] = torch.zeros(mtr((1,1))).to(dtype=torch.float32);
  if (flag_d2): d2W_ = cell(1+l_max); d2W_[0] = torch.zeros(mtr((1,1))).to(dtype=torch.float32);
  #%%%%%%%%;
  for l_val in range(1,l_max+1):
    m_val_ = torch.arange(-l_val,+l_val+1).to(dtype=torch.int32);
    n_m_val = numel(m_val_);
    sgn_ = (-1)**m_val_ * (m_val_>=0) + (+1)*(m_val_< 0);
    sgn__ = torch.reshape(sgn_,mtr((n_m_val,1))) * torch.reshape(sgn_,mtr((1,n_m_val))) ;
    V_mm__ = V_lmm___[l_val];
    L_m_ = L_lm__[l_val];
    #%%%%;
    if (flag_d0):
      tmp_d0W__ = mmmm( V_mm__ , torch.reshape((+L_m_/2)**0 * torch.exp(+L_m_*beta/2),mtr((n_m_val,1))) * torch.conj(V_mm__.T) ) ;
      tmp_d0W__ = tmp_d0W__ * sgn__;
      d0W_[l_val] = tmp_d0W__;
      del tmp_d0W__;
    #end;%if (flag_d0);
    #%%%%;
    if (flag_d1):
      tmp_d1W__ = mmmm( V_mm__ , torch.reshape((+L_m_/2)**1 * torch.exp(+L_m_*beta/2),mtr((n_m_val,1))) * torch.conj(V_mm__.T) );
      tmp_d1W__ = tmp_d1W__ * sgn__;
      d1W_[l_val] = tmp_d1W__;
      del tmp_d1W__;
    #end;%if (flag_d1);
    #%%%%;
    if (flag_d2):
      tmp_d2W__ = mmmm( V_mm__ , torch.reshape((+L_m_/2)**2 * torch.exp(+L_m_*beta/2),mtr((n_m_val,1))) * torch.conj(V_mm__.T) );
      tmp_d2W__ = tmp_d2W__ * sgn__;
      d2W_[l_val] = tmp_d2W__;
      del tmp_d2W__;
    #end;%if (flag_d2);
    #%%%%;
    del m_val_; del n_m_val; del sgn_; del sgn__; del V_mm__; del L_m_;
  #end;%for l_val=1:l_max;

  if (flag_verbose>0): print(f" %% [finished {str_thisfunction}]") ; 

  return (
    parameter,
    d0W_,
    d1W_,
    d2W_,
    V_lmm___,
    L_lm__,
  ) ;
