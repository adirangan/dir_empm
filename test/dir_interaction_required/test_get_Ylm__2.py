# M-x reb-change-syntax --> string ;
# M-x query-replace-regexp ;
# (\([^(),]*\),\([^(),]*\),\([^(),]*\)) ;
# [\3,\2,\1] ;

from dir_empm.dir_matlab_macros import * ;
from dir_empm.get_Ylm__2 import get_Ylm__2 ;

flag_verbose=2; flag_disp=0; nf=0;
print(' %% testing get_Ylm__2');
flag_verbose=1;
nf=0;
n_l = 48;
l_val_ = torch.tensor(3)+torch.arange(0,n_l); l_val_ = l_val_.to(dtype=torch.int32) ;
l_max = torch.max(l_val_).item();
#n_all = 1024*(1/8);
n_all = 8; #<-- reduce n_all if plotting first-derivative and second-derivative errors. ;
azimu_b_all_ = 2*pi*torch.linspace(0.0,1.0,n_all).to(dtype=torch.float32); #<-- throw in a few meridian entries just to check. ;
polar_a_all_ = 1*pi*torch.linspace(0.0,1.0,n_all).to(dtype=torch.float32); #<-- throw in a few polar entries just to check. ;
polar_a_all_[0:int(n_all/2)] = polar_a_all_[int(n_all/2):n_all]; #<-- test nonunique polar_a_all_. ;
#%%%%;
flag_flip = 0;
tmp_t = timeit.default_timer();
(
    _,
    Ylm_2__,
    _,
    _,
    sqrt_2lp1_,
    sqrt_2mp1_,
    sqrt_rat0_m_,
    sqrt_rat3_lm__,
    sqrt_rat4_lm__,
) = get_Ylm__2(
    [],
    n_l,
    l_val_,
    n_all,
    azimu_b_all_,
    polar_a_all_,
);
tmp_t = timeit.default_timer()-tmp_t;
if flag_verbose: print(f" %% get_Ylm__2 (not precomputation): {tmp_t:0.6f}s");
tmp_t = timeit.default_timer();
(
    _,
    Ylm_2__,
    _,
    _,
    _,
    _,
    _,
    _,
    _,
) = get_Ylm__2(
    [],
    n_l,
    l_val_,
    n_all,
    azimu_b_all_,
    polar_a_all_,
    flag_flip,
    sqrt_2lp1_,
    sqrt_2mp1_,
    sqrt_rat0_m_,
    sqrt_rat3_lm__,
    sqrt_rat4_lm__,
);
tmp_t = timeit.default_timer()-tmp_t;
if flag_verbose: print(f" %% get_Ylm__2 (yes precomputation): {tmp_t:0.6f}s");
#%%%%;
flag_flip = 1;
tmp_t = timeit.default_timer();
(
    _,
    Ylm_2__,
    _,
    _,
    _,
    _,
    _,
    _,
    _,
) = get_Ylm__2(
    [],
    n_l,
    l_val_,
    n_all,
    azimu_b_all_,
    polar_a_all_,
    flag_flip,
    sqrt_2lp1_,
    sqrt_2mp1_,
    sqrt_rat0_m_,
    sqrt_rat3_lm__,
    sqrt_rat4_lm__,
);
tmp_t = timeit.default_timer()-tmp_t;
if flag_verbose: print(f" %% get_Ylm__2 (yes precomputation): {tmp_t:0.6f}s");
#%%%%;
flag_flip = 1;
tmp_t = timeit.default_timer();
(
    _,
    d0Y_mid_lmj___,
    d1Y_mid_lmj___,
    d2Y_mid_lmj___,
    _,
    _,
    _,
    _,
    _,
) = get_Ylm__2(
    [],
    n_l,
    l_val_,
    n_all,
    azimu_b_all_,
    polar_a_all_,
    flag_flip,
    sqrt_2lp1_,
    sqrt_2mp1_,
    sqrt_rat0_m_,
    sqrt_rat3_lm__,
    sqrt_rat4_lm__,
);
tmp_t = timeit.default_timer()-tmp_t;
if flag_verbose: print(f" %% get_Ylm__2 (yes precomputation): {tmp_t:0.6f}s");

dir_base = '/data/rangan' ;
dir_ascii = dir_base + '/dir_cryoem/dir_rangan_python/dir_ascii' ;
for nl in range(n_l):
    l_val = l_val_[nl].item();
    fname_ascii = dir_ascii + '/d0Y_mid_' + str(l_val) + '_mj__.ascii' ;
    print(f" %% writing {fname_ascii}");
    np.savetxt(fname_ascii,d0Y_mid_lmj___[nl].numpy().ravel());
for nl in range(n_l):
    l_val = l_val_[nl].item();
    fname_ascii = dir_ascii + '/d1Y_mid_' + str(l_val) + '_mj__.ascii' ;
    print(f" %% writing {fname_ascii}");
    np.savetxt(fname_ascii,d1Y_mid_lmj___[nl].numpy().ravel());
for nl in range(n_l):
    l_val = l_val_[nl].item();
    fname_ascii = dir_ascii + '/d2Y_mid_' + str(l_val) + '_mj__.ascii' ;
    print(f" %% writing {fname_ascii}");
    np.savetxt(fname_ascii,d2Y_mid_lmj___[nl].numpy().ravel());

