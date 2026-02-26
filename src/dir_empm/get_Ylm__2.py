from dir_empm.dir_matlab_macros import * ;
from dir_empm.ylgndr_2 import ylgndr_2 ;

def get_Ylm__2(
  parameter=None,
  n_l=None,
  l_val_=None,
  n_all=None,
  azimu_b_all_=None,
  polar_a_all_=None,
  flag_flip=None,
  sqrt_2lp1_=None,
  sqrt_2mp1_=None,
  sqrt_rat0_m_=None,
  sqrt_rat3_lm__=None,
  sqrt_rat4_lm__=None,
  ):
  str_thisfunction = 'get_Ylm__2' ;

  if not isinstance(parameter, dict): parameter = {'type': 'parameter'};
  if 'flag_verbose' not in parameter: parameter['flag_verbose'] = 0;
  flag_verbose = parameter['flag_verbose']
  if 'flag_d' not in parameter: parameter['flag_d'] = 1;
  flag_d = parameter['flag_d']
  if 'flag_dd' not in parameter: parameter['flag_dd'] = 1;
  flag_dd = parameter['flag_dd']

  flag_precomputation = (
    sqrt_2lp1_ is None or
    sqrt_2mp1_ is None or
    sqrt_rat0_m_ is None or
    sqrt_rat3_lm__ is None or
    sqrt_rat4_lm__ is None
  );
  if flag_flip is None: flag_flip=0 ;
  if flag_flip: polar_a_all_ = 1*pi - polar_a_all_; #<-- switch hemispheres. ;
  if flag_flip: azimu_b_all_ = 1*pi + azimu_b_all_; #<-- add twelve hours. ;

  flag_d0 = 1;
  flag_d1 = flag_d;
  flag_d2 = flag_dd;

  if flag_verbose > 0: print(f" %% [entering {str_thisfunction}]") ;

  if flag_d0: d0Y_lmj___ = cell(n_l); #<-- list of n_l empty lists. ;
  if flag_d1: d1Y_lmj___ = cell(n_l); #<-- list of n_l empty lists. ;
  if flag_d2: d2Y_lmj___ = cell(n_l); #<-- list of n_l empty lists. ;
  np_polar_a_unique_,np_index_unique_,np_index_return_ = np.unique(polar_a_all_.numpy().ravel(),return_index=True,return_inverse=True);
  l_max = max(1,torch.max(l_val_).item());
  polar_a_unique_ = torch.tensor(np_polar_a_unique_).to(dtype=torch.float32);
  n_sub = numel(polar_a_unique_);
  cos_polar_a_unique_ = torch.cos(polar_a_unique_);
  sin_polar_a_unique_ = torch.sin(polar_a_unique_);
  tmp_t = timeit.default_timer();
  (
    parameter,
    d0y_jlm___,
    d1y_jlm___,
    d2y_jlm___,
    sqrt_2lp1_,
    sqrt_2mp1_,
    sqrt_rat0_m_,
    sqrt_rat3_lm__,
    sqrt_rat4_lm__,
  ) = ylgndr_2(
            parameter,
            l_max,
            cos_polar_a_unique_,
            sqrt_2lp1_,
            sqrt_2mp1_,
            sqrt_rat0_m_,
            sqrt_rat3_lm__,
            sqrt_rat4_lm__,
  );
  tmp_t = timeit.default_timer()-tmp_t; 
  if flag_verbose: print(f" %% ylgndr_2 (flag_precomputation {flag_precomputation}): {tmp_t:0.6f}s");

  tmp_t = timeit.default_timer();
  expimb_mj__ = torch.exp(+i*torch.reshape(torch.arange(-l_max,1+l_max),mtr((1+2*l_max,1)))*torch.reshape(azimu_b_all_[0:n_all],mtr((1,n_all)))).to(dtype=torch.complex64);
  tmp_t = timeit.default_timer()-tmp_t; 
  if flag_verbose: print(f" %% expimb_mj__: {tmp_t:0.6f}s");

  tmp_t = timeit.default_timer();
  if flag_d0:
    tmp_d0Y_lmj___ = torch.permute(d0y_jlm___,mtr(mts((1,2,0)))); #<-- permutation before inflation is faster. ;
    assert(tmp_d0Y_lmj___.size()==mtr((1+l_max,1+l_max,n_sub)));
    tmp_lval_ = torch.tensor(np.arange(0,1+l_max)).to(dtype=torch.int32);
    tmp_mval_ = torch.abs(torch.arange(-l_max,1+l_max)).to(dtype=torch.int32);
    tmp_jval_ = torch.tensor(np_index_return_).to(dtype=torch.int32);
    #tmp_d0Y_lmj___ = tmp_d0Y_lmj___.index_select(0,tmp_jval_).index_select(1,tmp_mval_) * torch.reshape(expimb_mj__,mtr((1,1+2*l_max,n_all))) / torch.sqrt(torch.tensor(4*pi)) ;
    tmp_i8_index_ = matlab_index_3d_0(1+l_max,tmp_lval_,1+l_max,tmp_mval_,n_sub,tmp_jval_);
    tmp_d0Y_lmj___ = torch.reshape(tmp_d0Y_lmj___.flatten()[tmp_i8_index_],mtr((1+l_max,1+2*l_max,n_all))) * torch.reshape(expimb_mj__,mtr((1,1+2*l_max,n_all))) / torch.sqrt(torch.tensor(4*pi)) ;
    for nl in range(n_l):
      tmp_lval = l_val_[nl].item(); tmp_lval_ = torch.tensor(tmp_lval).to(dtype=torch.int32);
      tmp_mval_ = torch.tensor(l_max+np.arange(-tmp_lval,1+tmp_lval)).to(dtype=torch.int32);
      tmp_jval_ = torch.tensor(np.arange(0,n_all)).to(dtype=torch.int32);
      #d0Y_lmj___[nl] = torch.reshape(tmp_d0Y_lmj___.index_select(0,tmp_jval_).index_select(1,tmp_mval_).index_select(2,tmp_lval_),(n_all,1+2*tmp_lval));
      tmp_i8_index_ = matlab_index_3d_0(1+l_max,tmp_lval_,1+2*l_max,tmp_mval_,n_all,tmp_jval_);
      d0Y_lmj___[nl] = torch.reshape(tmp_d0Y_lmj___.flatten()[tmp_i8_index_],mtr((1+2*tmp_lval,n_all))).to(dtype=torch.complex64);
    #end;%for nl=0:n_l-1;
  #end;%if flag_d0;
  if flag_d1:
    tmp_d1Y_lmj___ = ((-1)**flag_flip)**1 * torch.permute(d1y_jlm___*-sin_polar_a_unique_,mtr(mts((1,2,0))));
    assert(tmp_d1Y_lmj___.size()==mtr((1+l_max,1+l_max,n_sub)));
    tmp_lval_ = torch.tensor(np.arange(0,1+l_max)).to(dtype=torch.int32);
    tmp_mval_ = torch.abs(torch.arange(-l_max,1+l_max)).to(dtype=torch.int32);
    tmp_jval_ = torch.tensor(np_index_return_).to(dtype=torch.int32);
    #tmp_d1Y_lmj___ = tmp_d1Y_lmj___.index_select(0,tmp_jval_).index_select(1,tmp_mval_) * torch.reshape(expimb_mj__,mtr((1,1+2*l_max,n_all))) / torch.sqrt(torch.tensor(4*pi)) ;
    tmp_i8_index_ = matlab_index_3d_0(1+l_max,tmp_lval_,1+l_max,tmp_mval_,n_sub,tmp_jval_);
    tmp_d1Y_lmj___ = torch.reshape(tmp_d1Y_lmj___.flatten()[tmp_i8_index_],mtr((1+l_max,1+2*l_max,n_all))) * torch.reshape(expimb_mj__,mtr((1,1+2*l_max,n_all))) / torch.sqrt(torch.tensor(4*pi)) ;
    for nl in range(n_l):
      tmp_lval = l_val_[nl].item(); tmp_lval_ = torch.tensor(tmp_lval).to(dtype=torch.int32);
      tmp_mval_ = torch.tensor(l_max+np.arange(-tmp_lval,1+tmp_lval)).to(dtype=torch.int32);
      tmp_jval_ = torch.tensor(np.arange(0,n_all)).to(dtype=torch.int32);
      #d1Y_lmj___[nl] = torch.reshape(tmp_d1Y_lmj___.index_select(0,tmp_jval_).index_select(1,tmp_mval_).index_select(2,tmp_lval_),(n_all,1+2*tmp_lval));
      tmp_i8_index_ = matlab_index_3d_0(1+l_max,tmp_lval_,1+2*l_max,tmp_mval_,n_all,tmp_jval_);
      d1Y_lmj___[nl] = torch.reshape(tmp_d1Y_lmj___.flatten()[tmp_i8_index_],mtr((1+2*tmp_lval,n_all))).to(dtype=torch.complex64);
    #end;%for nl=0:n_l-1;
  #end;%if flag_d1;
  if flag_d2:
    tmp_d2Y_lmj___ = ((-1)**flag_flip)**2 * torch.permute(d2y_jlm___*(-sin_polar_a_unique_)**2 + d1y_jlm___*(-cos_polar_a_unique_),mtr(mts((1,2,0))));
    assert(tmp_d2Y_lmj___.size()==mtr((1+l_max,1+l_max,n_sub)));
    tmp_lval_ = torch.tensor(np.arange(0,1+l_max)).to(dtype=torch.int32);
    tmp_mval_ = torch.abs(torch.arange(-l_max,1+l_max)).to(dtype=torch.int32);
    tmp_jval_ = torch.tensor(np_index_return_).to(dtype=torch.int32);
    #tmp_d2Y_lmj___ = tmp_d2Y_lmj___.index_select(0,tmp_jval_).index_select(1,tmp_mval_) * torch.reshape(expimb_mj__,mtr((1,1+2*l_max,n_all))) / torch.sqrt(torch.tensor(4*pi)) ;
    tmp_i8_index_ = matlab_index_3d_0(1+l_max,tmp_lval_,1+l_max,tmp_mval_,n_sub,tmp_jval_);
    tmp_d2Y_lmj___ = torch.reshape(tmp_d2Y_lmj___.flatten()[tmp_i8_index_],mtr((1+l_max,1+2*l_max,n_all))) * torch.reshape(expimb_mj__,mtr((1,1+2*l_max,n_all))) / torch.sqrt(torch.tensor(4*pi)) ;
    for nl in range(n_l):
      tmp_lval = l_val_[nl].item(); tmp_lval_ = torch.tensor(tmp_lval).to(dtype=torch.int32);
      tmp_mval_ = torch.tensor(l_max+np.arange(-tmp_lval,1+tmp_lval)).to(dtype=torch.int32);
      tmp_jval_ = torch.tensor(np.arange(0,n_all)).to(dtype=torch.int32);
      #d2Y_lmj___[nl] = torch.reshape(tmp_d2Y_lmj___.index_select(0,tmp_jval_).index_select(1,tmp_mval_).index_select(2,tmp_lval_),(n_all,1+2*tmp_lval));
      tmp_i8_index_ = matlab_index_3d_0(1+l_max,tmp_lval_,1+2*l_max,tmp_mval_,n_all,tmp_jval_);
      d2Y_lmj___[nl] = torch.reshape(tmp_d2Y_lmj___.flatten()[tmp_i8_index_],mtr((1+2*tmp_lval,n_all))).to(dtype=torch.complex64);
    #end;%for nl=0:n_l-1;
  #end;%if flag_d2;
  tmp_t = timeit.default_timer()-tmp_t;
  if flag_verbose: print(f" %% permutation before inflation: {tmp_t:0.6f}s");

  index_pole_ = efind((1-torch.abs(torch.cos(polar_a_all_)))<1e-16);
  for nl in range(n_l):
    l_val = l_val_[nl].item();
    for m_val in range(-l_val,1+l_val):
      m_abs = np.abs(m_val);
      if flag_d1 and m_abs>0: 
        d1Y_lmj___[nl][index_pole_,l_val+m_val]=0;
      if flag_d2 and m_abs>0: d2Y_lmj___[nl][index_pole_,l_val+m_val]=0;
    #end;%for m_val=-l_val:+l_val;
  #end;%for nl=0:n_l-1;

  if flag_verbose > 0: print(f" %% [finished {str_thisfunction}]") ;

  return (
    parameter,
    d0Y_lmj___,
    d1Y_lmj___,
    d2Y_lmj___,
    sqrt_2lp1_,
    sqrt_2mp1_,
    sqrt_rat0_m_,
    sqrt_rat3_lm__,
    sqrt_rat4_lm__,
  )
