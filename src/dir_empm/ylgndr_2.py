from dir_empm.dir_matlab_macros import * ;

def ylgndr_2(
  parameter=None,
  l_max=None,
  x_=None,
  sqrt_2lp1_=None,
  sqrt_2mp1_=None,
  sqrt_rat0_m_=None,
  sqrt_rat3_lm__=None,
  sqrt_rat4_lm__=None,
  ):
  str_thisfunction = 'ylgndr_2' ;

  if not isinstance(parameter, dict): parameter = {'type': 'parameter'} ;
  if 'flag_verbose' not in parameter: parameter['flag_verbose'] = 0 ;
  flag_verbose = parameter['flag_verbose'] ;
  if 'flag_d' not in parameter: parameter['flag_d'] = 1 ;
  flag_d = parameter['flag_d'] ;
  if 'flag_dd' not in parameter: parameter['flag_dd'] = 1 ;
  flag_dd = parameter['flag_dd'] ;

  if flag_verbose > 0: print(f" %% [entering {str_thisfunction}]") ;

  flag_precomputation = (
    sqrt_2lp1_ is None or
    sqrt_2mp1_ is None or
    sqrt_rat0_m_ is None or
    sqrt_rat3_lm__ is None or
    sqrt_rat4_lm__ is None
  ) ;

  if flag_precomputation:
    l_val_ = torch.arange(0,l_max+1).to(dtype=torch.int32).flatten() ;
    m_val_ = torch.arange(0,l_max+1).to(dtype=torch.int32).flatten() ;
    base_2lm1_ = 2.0 * l_val_ - 1.0 ;
    sqrt_2lp1_ = torch.sqrt(2.0 * l_val_ + 1.0) ; #<-- note torch.sqrt applied to torch.int32 automatically produces torch.float32. ;
    sqrt_2mp1_ = torch.sqrt(2.0 * m_val_ + 1.0) ; #<-- note torch.sqrt applied to torch.int32 automatically produces torch.float32. ;
    sqrt_rat0_m_ = torch.sqrt((2.0 * m_val_ - 1.0) / torch.maximum(torch.tensor(1),2.0 * m_val_)) ;
    l_mat_l1__ = l_val_.reshape(mtr((1+l_max,1))) ;
    m_mat_1m__ = m_val_.reshape(mtr((1,1+l_max))) ;
    assert(l_mat_l1__.shape==mtr((1+l_max,1)));
    assert(m_mat_1m__.shape==mtr((1,1+l_max)));
    sqrt_rat1_lm__ = torch.sqrt( (l_mat_l1__ + m_mat_1m__ - 1.0) * (l_mat_l1__ - m_mat_1m__ - 1.0) ) ;
    sqrt_rat2_lm__ = torch.sqrt( (l_mat_l1__ - m_mat_1m__ + 0.0) * (l_mat_l1__ + m_mat_1m__ + 0.0) ) ;
    sqrt_rat3_lm__ = sqrt_rat1_lm__ / sqrt_rat2_lm__ ;
    sqrt_rat4_lm__ = base_2lm1_.reshape(mtr((l_max+1,1))) / sqrt_rat2_lm__ ;
    for m_val in range(0,l_max+1):
      if m_val==0: sqrt_rat0_m_[m_val]=0 ;
      for l_val in range(0,l_max+1):
        if (m_val>=l_val):
          sqrt_rat3_lm__[m_val,l_val]=0; #<-- matlab_arranged, python_addressed. ;
          sqrt_rat4_lm__[m_val,l_val]=0; #<-- matlab_arranged, python_addressed. ;
        #end;%if;
      #end;%for l_val;
    #end;%for m_val;
  #end;%if flag_precomputation ;
  
  flag_d0 = 1;
  flag_d1 = flag_d;
  flag_d2 = flag_dd;

  n_x = numel(x_);
  x_ = x_.flatten();
  d0u_ = -torch.sqrt((1 - x_) * (1 + x_));
  d1u_ = x_ / torch.sqrt((1 - x_) * (1 + x_));
  d2u_ = -1.0 / d0u_ - x_**2 / d0u_**3;

  d0y_jlm___ = None; d1y_jlm___ = None; d2y_jlm___ = None;
  if flag_d0: d0y_jlm___ = torch.zeros(mtr((n_x,l_max+1,l_max+1))).to(dtype=torch.float32);
  if flag_d1: d1y_jlm___ = torch.zeros(mtr((n_x,l_max+1,l_max+1))).to(dtype=torch.float32);
  if flag_d2: d2y_jlm___ = torch.zeros(mtr((n_x,l_max+1,l_max+1))).to(dtype=torch.float32);
  if flag_d0: d0y_jlm___[0,0,:] = 1.0; #<-- matlab_arranged, python_addressed. ;
  if flag_d1: d1y_jlm___[0,0,:] = 0.0; #<-- matlab_arranged, python_addressed. ;
  if flag_d2: d2y_jlm___[0,0,:] = 0.0; #<-- matlab_arranged, python_addressed. ;

  for m_val in range(0, l_max + 1):
    if m_val > 0:
      if flag_d0: d0y_jlm___[ m_val, m_val,:] = d0y_jlm___[ m_val - 1, m_val - 1,:] * d0u_ * sqrt_rat0_m_[m_val] ; #<-- matlab_arranged, python_addressed. ;
      if flag_d1: d1y_jlm___[ m_val, m_val,:] = d0y_jlm___[ m_val, m_val,:] * (-m_val) * x_ / d0u_**2 ;  #<-- matlab_arranged, python_addressed. ;
      if flag_d2:
        d2y_jlm___[ m_val, m_val,:] = (
          d1y_jlm___[ m_val, m_val,:] * (-m_val) * x_ / d0u_**2 +
          d0y_jlm___[ m_val, m_val,:] * (-m_val) * (1 + x_**2) / d0u_**4
        ) ;  #<-- matlab_arranged, python_addressed. ;
    if m_val < l_max:
      if flag_d0:
        d0y_jlm___[ m_val, m_val + 1,:] = x_ * d0y_jlm___[ m_val, m_val,:] * sqrt_2mp1_[m_val] ;  #<-- matlab_arranged, python_addressed. ;
      if flag_d1:
        d1y_jlm___[ m_val, m_val + 1,:] = (
          d0y_jlm___[ m_val, m_val,:] + x_ * d1y_jlm___[ m_val, m_val,:]
        ) * sqrt_2mp1_[m_val] ;  #<-- matlab_arranged, python_addressed. ;
      if flag_d2:
        d2y_jlm___[ m_val, m_val + 1,:] = (
          2 * d1y_jlm___[ m_val, m_val,:] + x_ * d2y_jlm___[ m_val, m_val,:]
        ) * sqrt_2mp1_[m_val] ; #<-- matlab_arranged, python_addressed. ;
    for l_val in range(m_val + 2, l_max + 1):
      if flag_d0:
        d0y_jlm___[ m_val, l_val,:] = (
          sqrt_rat4_lm__[ m_val,l_val] * x_ * d0y_jlm___[ m_val, l_val - 1,:] -
          sqrt_rat3_lm__[ m_val,l_val] * d0y_jlm___[ m_val, l_val - 2,:]
        ) ; #<-- matlab_arranged, python_addressed. ;
      if flag_d1:
        d1y_jlm___[ m_val, l_val,:] = (
          sqrt_rat4_lm__[ m_val,l_val] * (d0y_jlm___[ m_val, l_val - 1,:] + x_ * d1y_jlm___[ m_val, l_val - 1,:]) -
          sqrt_rat3_lm__[ m_val,l_val] * d1y_jlm___[ m_val, l_val - 2,:]
        ) ; #<-- matlab_arranged, python_addressed. ;
      if flag_d2:
        d2y_jlm___[ m_val, l_val,:] = (
          sqrt_rat4_lm__[ m_val,l_val] * (2 * d1y_jlm___[ m_val, l_val - 1,:] + x_ * d2y_jlm___[ m_val, l_val - 1,:]) -
          sqrt_rat3_lm__[ m_val,l_val] * d2y_jlm___[ m_val, l_val - 2,:]
        ) ; #<-- matlab_arranged, python_addressed. ;
  #end;%for m_val in range(0, l_max + 1):

  if flag_d0: d0y_jlm___ = d0y_jlm___ * sqrt_2lp1_.reshape(mtr((1,1+l_max,1))) ; 
  if flag_d1: d1y_jlm___ = d1y_jlm___ * sqrt_2lp1_.reshape(mtr((1,1+l_max,1))) ; 
  if flag_d2: d2y_jlm___ = d2y_jlm___ * sqrt_2lp1_.reshape(mtr((1,1+l_max,1))) ;

  if flag_verbose > 0: print(f" %% [finished {str_thisfunction}]") ; 

  return (
    parameter,
    d0y_jlm___,
    d1y_jlm___,
    d2y_jlm___,
    sqrt_2lp1_,
    sqrt_2mp1_,
    sqrt_rat0_m_,
    sqrt_rat3_lm__,
    sqrt_rat4_lm__,
  ) ;
