#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;

__all__ = [
    "matlab_index_2d_0",
    "matlab_index_3d_0",
    "matlab_index_4d_0",
    "matlab_index_2d_gpu_0",
    "matlab_index_3d_gpu_0",
    "matlab_index_4d_gpu_0",
    "matlab_scalar_round ",
    "periodize",
    "disp",
    "sprintf",
    "fnorm_disp",
    "unique_0",
    "intersect_0",
    "union_0",
    "setdiff_0",
    "parameter_timing_update",
    "parameter_timing_printf",
    "matlab_svds",
    "matlab_save",
    "matlab_load",
];

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;

import io ;
import numpy as np ; pi = np.pi ; i = 1j ; import torch ; import timeit ;
from scipy.sparse import csr_matrix ;
from scipy.io import savemat ;
from scipy.io import loadmat ;

rng = lambda a : torch.manual_seed(a) ;
cell = lambda n: [ [] for _ in range(n) ] ; #%<-- cell array. ;
isempty = lambda t: (t is None) or (isinstance(t, torch.Tensor) and t.numel() == 0) ;
numel_unique = lambda a : np.unique(a.numpy().ravel()).size ;
numel = lambda a : int(a.numel()) ;
cumsum_0 = lambda a : torch.cumsum(torch.concatenate((torch.tensor([0]),a)) , 0).to(torch.int32) ;
fnorm = lambda a : torch.linalg.norm(a).item() ;
ndims = lambda a : a.ndim ;
size = lambda a , d : a.shape[ndims(a)-1-d] ;
mtr = lambda a : tuple(reversed(a)) ; #<-- matlab-arranged size (i.e., tuple(reversed(...))). ;
msr = lambda str : str[::-1] ; #<-- for einsum (i.e., string reversed (...)). ;
mts = lambda a : tuple(len(a) - x - 1 for x in a) ; #<-- for permute (i.e., tuple subtract (...)). ;
etumrep = lambda a : torch.permute(a,tuple(torch.arange(ndims(a)-1,-1,-1).tolist())) ; #<-- matlab-arranged permutation (for savemat). ;
tic = lambda : timeit.default_timer() ;
toc = lambda a : tic() - a ;
mmmm = lambda A , B : torch.einsum( msr('ab') + ',' + msr('bc') + '->' + msr('ac') , A , B ) ; #<-- matlab matrix matrix multiplication. ;
mmvm = lambda A , B : torch.einsum( msr('ab') + ',' +  msr('b') + '->' +  msr('a') , A , B ) ; #<-- matlab matrix vector multiplication. ;
mvmm = lambda A , B : torch.einsum(  msr('b') + ',' + msr('bc') + '->' +  msr('c') , A , B ) ; #<-- matlab vector matrix multiplication. ;
mvvm = lambda A , B : torch.einsum(  msr('b') + ',' +  msr('b') + '->' +   msr('') , A , B ) ; #<-- matlab vector vector multiplication. ;
efind = lambda a : torch.where(a)[0] ;
n_1 = int(1); n_2 = int(2); n_3 = int(3);
n_byte_per_float32 = 4; n_byte_per_float64 = 8;
n_byte_per_complex64 = 8; n_byte_per_complex128 = 16;
np_sparse = lambda nr_,nc_,v_,n_r,n_c : csr_matrix( ( v_.numpy(),(nc_.numpy(),nr_.numpy()) ) , shape=(n_c,n_r) ) ; #<-- note matlab-arranged dimensions. ;
m_npcsr_mm = lambda np_csr_A,B : torch.reshape(torch.tensor(np_csr_A.T.dot(B.numpy().T).T),mtr((size(np_csr_A,0),size(B,1)))) ; #<-- note extra transposes to match matlab. ;
m_npcsr_vm = lambda np_csr_A,B : torch.reshape(torch.tensor(np_csr_A.T.dot(torch.reshape(B.ravel(),mtr((numel(B.ravel()),1))).numpy().T).T),mtr((size(np_csr_A,0),n_1))).ravel() ; #<-- note extra transposes to match matlab. ;
import torch;
numel = lambda a : int(a.numel()) ;
efind = lambda a : torch.where(a)[0] ;

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;

def sprintf(format_str, *format_args):
    tmp_str = io.StringIO();
    tmp_str.write(format_str % format_args);
    return(tmp_str.getvalue());

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;

def disp(input_str):
    print(input_str);

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;

def fnorm_disp(flag_verbose, str_v0, v0, str_v1, v1, str_postfix=""):
    if np.isscalar(v0): v0 = torch.tensor([v0]) ;
    if np.isscalar(v1): v1 = torch.tensor([v1]) ;
    if v0.ndim != v1.ndim: print(f"Warning, {str_v0} ndims(v0) {v0.ndim} != {str_v1} ndims(v1) {v1.ndim}");
    if v0.size() != v1.size(): print(f"Warning, {str_v0} numel(v0) {v0.size} != {str_v1} numel(v1) {v1.size}");
    
    d0_ = v0.shape;
    d1_ = v1.shape;
    n_d0 = len(d0_);
    n_d1 = len(d1_);
    
    for nd0 in range(n_d0):
        nd1 = nd0;
        d0 = d0_[nd0];
        d1 = d1_[nd1];
        if d0 != d1: print(f"Warning, size({str_v0},1+{nd0}) {d0} != size({str_v1},1+{nd1}) {d1}") ;
    
    errrel = torch.linalg.norm(v0 - v1) / max(1e-12, torch.linalg.norm(v0)) ;
    if flag_verbose > 0:
        print(f"{str_v0:>16} {torch.linalg.norm(v0):+16.6f} vs {str_v1:>16} {torch.linalg.norm(v1):+16.6f}: r {errrel:0.16f}{str_postfix}");
    
    return errrel ;

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;

def matlab_index_2d_0(
        n_0=None,
        index_n0_=None,
        n_1=None,
        index_n1_=None,
):
      str_thisfunction = 'matlab_index_2d_0';
      if index_n0_ is None: index_n0_ = torch.arange(0,n_0);
      if index_n1_ is None: index_n1_ = torch.arange(0,n_1);
      if index_n0_==':': index_n0_ = torch.arange(0,n_0);
      if index_n1_==':': index_n1_ = torch.arange(0,n_1);
      if np.isscalar(index_n0_): index_n0_ = torch.tensor([index_n0_]) ;
      if np.isscalar(index_n1_): index_n1_ = torch.tensor([index_n1_]) ;
      index_n1__,index_n0__ = torch.meshgrid(index_n1_.to(dtype=torch.int64),index_n0_.to(dtype=torch.int64),indexing='ij');
      index_n01__ = index_n0__.to(dtype=torch.int64) + index_n1__.to(dtype=torch.int64)*n_0; #<-- this should encourage index_n01__ to be torch.int64. ;
      index_n01_ = torch.flatten(index_n01__);
      return(index_n01_);

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;

def matlab_index_2d_gpu_0(
            device_use=None,
            n_0=None,
            index_n0_=None,
            n_1=None,
            index_n1_=None,
):
      str_thisfunction = 'matlab_index_2d_gpu_0';
      if index_n0_ is None: index_n0_ = torch.arange(0,n_0).to(device=device_use);
      if index_n1_ is None: index_n1_ = torch.arange(0,n_1).to(device=device_use);
      if index_n0_==':': index_n0_ = torch.arange(0,n_0).to(device=device_use);
      if index_n1_==':': index_n1_ = torch.arange(0,n_1).to(device=device_use);
      if np.isscalar(index_n0_): index_n0_ = torch.tensor([index_n0_]).to(device=device_use) ;
      if np.isscalar(index_n1_): index_n1_ = torch.tensor([index_n1_]).to(device=device_use) ;
      index_n1__,index_n0__ = torch.meshgrid(index_n1_.to(dtype=torch.int64,device=device_use),index_n0_.to(dtype=torch.int64,device=device_use),indexing='ij');
      index_n01__ = index_n0__.to(dtype=torch.int64,device=device_use) + index_n1__.to(dtype=torch.int64,device=device_use)*n_0; #<-- this should encourage index_n01__ to be torch.int64. ;
      index_n01_ = torch.flatten(index_n01__).to(device=device_use);
      return(index_n01_);

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;

def matlab_index_3d_0(
        n_0=None,
        index_n0_=None,
        n_1=None,
        index_n1_=None,
        n_2=None,
        index_n2_=None,
):
      str_thisfunction = 'matlab_index_3d_0';
      if index_n0_ is None: index_n0_ = torch.arange(0,n_0);
      if index_n1_ is None: index_n1_ = torch.arange(0,n_1);
      if index_n2_ is None: index_n2_ = torch.arange(0,n_2);
      if index_n0_==':': index_n0_ = torch.arange(0,n_0);
      if index_n1_==':': index_n1_ = torch.arange(0,n_1);
      if index_n2_==':': index_n2_ = torch.arange(0,n_2);
      if np.isscalar(index_n0_): index_n0_ = torch.tensor([index_n0_]) ;
      if np.isscalar(index_n1_): index_n1_ = torch.tensor([index_n1_]) ;
      if np.isscalar(index_n2_): index_n2_ = torch.tensor([index_n2_]) ;
      index_n2___,index_n1___,index_n0___ = torch.meshgrid(index_n2_.to(dtype=torch.int64),index_n1_.to(dtype=torch.int64),index_n0_.to(dtype=torch.int64),indexing='ij');
      index_n012___ = index_n0___.to(dtype=torch.int64) + (index_n1___.to(dtype=torch.int64) + index_n2___.to(dtype=torch.int64)*n_1)*n_0; #<-- this should encourage index_n012___ to be torch.int64. ;
      index_n012_ = torch.flatten(index_n012___);
      return(index_n012_);

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;

def matlab_index_3d_gpu_0(
            device_use=None,
            n_0=None,
            index_n0_=None,
            n_1=None,
            index_n1_=None,
            n_2=None,
            index_n2_=None,
):
      str_thisfunction = 'matlab_index_3d_gpu_0';
      if index_n0_ is None: index_n0_ = torch.arange(0,n_0).to(device=device_use);
      if index_n1_ is None: index_n1_ = torch.arange(0,n_1).to(device=device_use);
      if index_n2_ is None: index_n2_ = torch.arange(0,n_2).to(device=device_use);
      if index_n0_==':': index_n0_ = torch.arange(0,n_0).to(device=device_use);
      if index_n1_==':': index_n1_ = torch.arange(0,n_1).to(device=device_use);
      if index_n2_==':': index_n2_ = torch.arange(0,n_2).to(device=device_use);
      if np.isscalar(index_n0_): index_n0_ = torch.tensor([index_n0_]).to(device=device_use) ;
      if np.isscalar(index_n1_): index_n1_ = torch.tensor([index_n1_]).to(device=device_use) ;
      if np.isscalar(index_n2_): index_n2_ = torch.tensor([index_n2_]).to(device=device_use) ;
      index_n2___,index_n1___,index_n0___ = torch.meshgrid(index_n2_.to(dtype=torch.int64,device=device_use),index_n1_.to(dtype=torch.int64,device=device_use),index_n0_.to(dtype=torch.int64,device=device_use),indexing='ij');
      index_n012___ = index_n0___.to(dtype=torch.int64,device=device_use) + (index_n1___.to(dtype=torch.int64,device=device_use) + index_n2___.to(dtype=torch.int64,device=device_use)*n_1)*n_0; #<-- this should encourage index_n012___ to be torch.int64. ;
      index_n012_ = torch.flatten(index_n012___).to(device=device_use);
      return(index_n012_);

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;

def matlab_index_4d_0(
        n_0=None,
        index_n0_=None,
        n_1=None,
        index_n1_=None,
        n_2=None,
        index_n2_=None,
        n_3=None,
        index_n3_=None,
):
      str_thisfunction = 'matlab_index_4d_0';
      if index_n0_ is None: index_n0_ = torch.arange(0,n_0);
      if index_n1_ is None: index_n1_ = torch.arange(0,n_1);
      if index_n2_ is None: index_n2_ = torch.arange(0,n_2);
      if index_n3_ is None: index_n3_ = torch.arange(0,n_3);
      if index_n0_==':': index_n0_ = torch.arange(0,n_0);
      if index_n1_==':': index_n1_ = torch.arange(0,n_1);
      if index_n2_==':': index_n2_ = torch.arange(0,n_2);
      if index_n3_==':': index_n3_ = torch.arange(0,n_3);
      if np.isscalar(index_n0_): index_n0_ = torch.tensor([index_n0_]) ;
      if np.isscalar(index_n1_): index_n1_ = torch.tensor([index_n1_]) ;
      if np.isscalar(index_n2_): index_n2_ = torch.tensor([index_n2_]) ;
      if np.isscalar(index_n3_): index_n3_ = torch.tensor([index_n3_]) ;
      index_n3____,index_n2____,index_n1____,index_n0____ = torch.meshgrid(index_n3_.to(dtype=torch.int64),index_n2_.to(dtype=torch.int64),index_n1_.to(dtype=torch.int64),index_n0_.to(dtype=torch.int64),indexing='ij');
      index_n0123____ = index_n0____.to(dtype=torch.int64) + (index_n1____.to(dtype=torch.int64) + (index_n2____.to(dtype=torch.int64) + index_n3____.to(dtype=torch.int64)*n_2)*n_1)*n_0; #<-- this should encourage index_n0123____ to be torch.int64. ;
      index_n0123_ = torch.flatten(index_n0123____);
      return(index_n0123_);

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;

def matlab_index_4d_gpu_0(
            device_use=None,
            n_0=None,
            index_n0_=None,
            n_1=None,
            index_n1_=None,
            n_2=None,
            index_n2_=None,
            n_3=None,
            index_n3_=None,
):
      str_thisfunction = 'matlab_index_4d_gpu_0';
      if index_n0_ is None: index_n0_ = torch.arange(0,n_0).to(device=device_use);
      if index_n1_ is None: index_n1_ = torch.arange(0,n_1).to(device=device_use);
      if index_n2_ is None: index_n2_ = torch.arange(0,n_2).to(device=device_use);
      if index_n3_ is None: index_n3_ = torch.arange(0,n_3).to(device=device_use);
      if index_n0_==':': index_n0_ = torch.arange(0,n_0).to(device=device_use);
      if index_n1_==':': index_n1_ = torch.arange(0,n_1).to(device=device_use);
      if index_n2_==':': index_n2_ = torch.arange(0,n_2).to(device=device_use);
      if index_n3_==':': index_n3_ = torch.arange(0,n_3).to(device=device_use);
      if np.isscalar(index_n0_): index_n0_ = torch.tensor([index_n0_]).to(device=device_use) ;
      if np.isscalar(index_n1_): index_n1_ = torch.tensor([index_n1_]).to(device=device_use) ;
      if np.isscalar(index_n2_): index_n2_ = torch.tensor([index_n2_]).to(device=device_use) ;
      if np.isscalar(index_n3_): index_n3_ = torch.tensor([index_n3_]).to(device=device_use) ;
      index_n3____,index_n2____,index_n1____,index_n0____ = torch.meshgrid(index_n3_.to(dtype=torch.int64,device=device_use),index_n2_.to(dtype=torch.int64,device=device_use),index_n1_.to(dtype=torch.int64,device=device_use),index_n0_.to(dtype=torch.int64,device=device_use),indexing='ij');
      index_n0123____ = index_n0____.to(dtype=torch.int64,device=device_use) + (index_n1____.to(dtype=torch.int64,device=device_use) + (index_n2____.to(dtype=torch.int64,device=device_use) + index_n3____.to(dtype=torch.int64,device=device_use)*n_2)*n_1)*n_0; #<-- this should encourage index_n0123____ to be torch.int64. ;
      index_n0123_ = torch.flatten(index_n0123____).to(device=device_use);
      return(index_n0123_);

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;

def matlab_index_5d_0(
        n_0=None,
        index_n0_=None,
        n_1=None,
        index_n1_=None,
        n_2=None,
        index_n2_=None,
        n_3=None,
        index_n3_=None,
        n_4=None,
        index_n4_=None,
):
      str_thisfunction = 'matlab_index_5d_0';
      if index_n0_ is None: index_n0_ = torch.arange(0,n_0);
      if index_n1_ is None: index_n1_ = torch.arange(0,n_1);
      if index_n2_ is None: index_n2_ = torch.arange(0,n_2);
      if index_n3_ is None: index_n3_ = torch.arange(0,n_3);
      if index_n4_ is None: index_n4_ = torch.arange(0,n_4);
      if index_n0_==':': index_n0_ = torch.arange(0,n_0);
      if index_n1_==':': index_n1_ = torch.arange(0,n_1);
      if index_n2_==':': index_n2_ = torch.arange(0,n_2);
      if index_n3_==':': index_n3_ = torch.arange(0,n_3);
      if index_n4_==':': index_n4_ = torch.arange(0,n_4);
      index_n4_____,index_n3_____,index_n2_____,index_n1_____,index_n0_____ = torch.meshgrid(index_n4_,index_n3_,index_n2_,index_n1_,index_n0_,indexing='ij');
      index_n01234_____ = index_n0_____ + (index_n1_____ + (index_n2_____ + (index_n3_____ + index_n4_____*n_3)*n_2)*n_1)*n_0;
      index_n01234_ = torch.flatten(index_n01234_____);
      return(index_n01234_);

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;

#%%%%%%%%;
#% This is designed under the assumption that all matlab vectors are column-vectors. ;
#% This will produce errors when 2-dimensional arrays have sizes of 1. ;
#%%%%%%%%;
def matlab_load(
        fname_mat=None,
):
    dictionary_reversed = loadmat(file_name=fname_mat);
    dictionary_original = dictionary_reversed;
    for key in dictionary_reversed:
        if isinstance(dictionary_reversed[key],np.ndarray):
            try: 
                dictionary_original[key] = etumrep(torch.tensor(dictionary_reversed[key]));
            except:
                pass ;
        #end;%if isinstance(dictionary_reversed[key],torch.Tensor): ;
        if isinstance(dictionary_original[key],torch.Tensor):
            if dictionary_original[key].ndim==2:
                try:
                    dictionary_original[key] = torch.squeeze(dictionary_original[key]); #%<-- Here we squeeze all vectors. ;
                except:
                    pass ;
            #end;%if ;
        #end;%if isinstance(dictionary_original[key],torch.Tensor): ;
    #end;%for key in dictionary_reversed: ;
    return(dictionary_original);

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;

def matlab_save(
        fname_mat=None,
        dictionary_original=None,
):
    dictionary_reversed = dictionary_original;
    for key in dictionary_original:
        if isinstance(dictionary_reversed[key],torch.Tensor):
            dictionary_reversed[key] = etumrep(dictionary_original[key]);
        #end;%if isinstance(dictionary_reversed[key],torch.Tensor): ;
    #end;%for key in dictionary_original: ;
    savemat(
        file_name=fname_mat,
        mdict= dictionary_reversed,
        oned_as='column',
    );

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;

def matlab_scalar_round(scalar):
    if scalar % 1 == 0.5: return int(scalar)+1;
    return round(scalar)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;

def matlab_svd(
        X_rc__=None,
):
    n_r = size(X_rc__,0); n_c = size(X_rc__,1);
    tmp_UX_rn__ , tmp_SX_n_ , tmp_VX_cn__ = torch.linalg.svd(X_rc__.T,full_matrices=False); 
    tmp_UX_rn__ = tmp_UX_rn__.T; #%<-- extra transposes to match matlab. ;
    return(
        tmp_UX_rn__,
        tmp_SX_n_,
        tmp_VX_cn__,
    );

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;

def matlab_svds(
        X_rc__=None,
        n_svd=None,
):
    n_r = size(X_rc__,0); n_c = size(X_rc__,1);
    if n_svd is None: n_svd = int(np.minimum(n_r,n_c)); #end;
    tmp_UX_rn__ , tmp_SX_n_ , tmp_VX_cn__ = matlab_svd(X_rc__);
    tmp_UX_rn__ = tmp_UX_rn__[0:n_svd,:];
    tmp_SX_n_ = tmp_SX_n_[0:n_svd];
    tmp_VX_cn__ = tmp_VX_cn__[0:n_svd,:];
    return(
        tmp_UX_rn__,
        tmp_SX_n_,
        tmp_VX_cn__,
    );

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;

def parameter_timing_printf(
        parameter=None,
):
    if parameter is None: parameter = {'type': 'parameter'};
    if 'timing' not in parameter or parameter['timing'] is None:
        parameter['timing'] = [
            ['label', 'total time', 'number of calls', 'number of ops', 'MHz', 'GHz']
        ];
    #end;%if ;
    timing = parameter['timing'];
    n_label = len(timing)-1;
    t_tot_ = np.zeros(n_label);
    for nlabel in range(n_label):
        nt=1; t_tot = timing[1+nlabel][nt];
        t_tot_[nlabel] = t_tot;
    #end;%for nlabel in range(n_label): ;
    index_srt_ = np.flip(np.argsort(t_tot_),axis=0);
    disp(sprintf(' %% %16.16ss %16.16s %16.16s %s','t_tot','n_cal','n_ops','label'));
    for nlabel in range(n_label):
        nt=0;
        label = timing[1+index_srt_[nlabel]][nt]; nt=nt+1;
        t_tot = timing[1+index_srt_[nlabel]][nt]; nt=nt+1;
        n_cal = timing[1+index_srt_[nlabel]][nt]; nt=nt+1;
        n_ops = timing[1+index_srt_[nlabel]][nt]; nt=nt+1;
        MHz_ = timing[1+index_srt_[nlabel]][nt]; nt=nt+1;
        GHz_ = timing[1+index_srt_[nlabel]][nt]; nt=nt+1;
        disp(sprintf(' %% %+16.6fs %+16.4d %+16.4d %s',t_tot,n_cal,n_ops,label));
    #end;%for nlabel in range(n_label): ;

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;

def parameter_timing_update(
        parameter=None,
        str_field=None,
        dt=None,
        ncall=None,
        nop=None,
):
    if dt is None: dt=0.0;
    if ncall is None: ncall=int(0);
    if nop is None: nop=int(0);
    if parameter is None: parameter = {'type': 'parameter'};
    if 'timing' not in parameter or parameter['timing'] is None:
        parameter['timing'] = [
            ['label', 'total time', 'number of calls', 'number of ops', 'MHz', 'GHz']
        ];
    #end;%if ;
    timing = parameter['timing'];
    tmp_index = None;
    for idx in range(1, len(timing)):
        if timing[idx][0] == str_field:
            tmp_index = idx ;
            break ;
        #end;%if;
    #end;%for;

    if tmp_index is not None:
        timing[tmp_index][1] = timing[tmp_index][1] + dt ;
        timing[tmp_index][2] = timing[tmp_index][2] + ncall ;
        timing[tmp_index][3] = timing[tmp_index][3] + nop ;
    else:
        timing.append([str_field, dt, ncall, nop, 0.0, 0.0]) ;
        tmp_index = len(timing) - 1 ;
    #end;%if;

    n_op = timing[tmp_index][3] ;
    d_t = timing[tmp_index][1] ;
    if (n_op > 0) and (d_t > 0):
        timing[tmp_index][4] = (n_op / np.maximum(1e-12,d_t)) / 1e6 ; # MHz
        timing[tmp_index][5] = (n_op / np.maximum(1e-12,d_t)) / 1e9 ; # GHz
    #end;%if;

    parameter['timing'] = timing;
    return(parameter);

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;

def periodize(input, a, b):
    return a + (input - a) % (b - a)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;

#%%%%%%%%;
#% Warning, This is a brute-force O(n^2) intersection function. ;
#% The only goal is for this to mimic the matlab intersection. ;
#%%%%%%%%;
def intersect_0(a_, b_):
    u_a_,index_nu_a_from_na_,n_u_a_ = torch.unique(a_.ravel(),return_inverse=True,return_counts=True,dim=0);
    u_b_,index_nu_b_from_nb_,n_u_b_ = torch.unique(b_.ravel(),return_inverse=True,return_counts=True,dim=0);
    u_a_u_b_ = torch.concatenate([u_a_.ravel(), u_b_.ravel()],0);
    u_u_a_u_b_ , index_nu_u_a_u_b_from_nu_a_u_b_ , n_u_u_a_u_b_ = torch.unique(u_a_u_b_.ravel(),return_inverse=True,return_counts=True,dim=0);
    index_cap_ = efind(n_u_u_a_u_b_> 1);
    cap_ = u_u_a_u_b_.ravel()[index_cap_];
    n_cap = numel(cap_);
    index_na_from_ncap_ = torch.zeros(n_cap).to(dtype=torch.int32);
    index_nb_from_ncap_ = torch.zeros(n_cap).to(dtype=torch.int32);
    for ncap in range(n_cap):
        cap = cap_[ncap].item();
        na = int(torch.min(efind(a_==cap)).item());
        index_na_from_ncap_[ncap] = na;
        nb = int(torch.min(efind(b_==cap)).item());
        index_nb_from_ncap_[ncap] = nb;
    #end;%for ncap in range(n_cap):
    return(cap_,index_na_from_ncap_,index_nb_from_ncap_);

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;

#%%%%%%%%;
#% Warning, This is a brute-force O(n^2) union function. ;
#% The only goal is for this to mimic the matlab union. ;
#%%%%%%%%;
def union_0(a_, b_):
    u_a_,index_nu_a_from_na_,n_u_a_ = torch.unique(a_.ravel(),return_inverse=True,return_counts=True,dim=0);
    u_b_,index_nu_b_from_nb_,n_u_b_ = torch.unique(b_.ravel(),return_inverse=True,return_counts=True,dim=0);
    u_a_u_b_ = torch.concatenate([u_a_.ravel(), u_b_.ravel()],0);
    u_u_a_u_b_ , index_nu_u_a_u_b_from_nu_a_u_b_ , n_u_u_a_u_b_ = torch.unique(u_a_u_b_.ravel(),return_inverse=True,return_counts=True,dim=0);
    cup_ = u_u_a_u_b_.ravel();
    n_cup = numel(cup_);
    index_na_from_ncup_ = -torch.ones(n_cup).to(dtype=torch.int32);
    index_nb_from_ncup_ = -torch.ones(n_cup).to(dtype=torch.int32);
    for ncup in range(n_cup):
        cup = cup_[ncup].item();
        tmp_index_na_ = efind(a_==cup);
        if numel(tmp_index_na_)> 0:
            na = int(torch.min(tmp_index_na_).item());
            index_na_from_ncup_[ncup] = na;
        #end;%if (numel(tmp_index_na_)> 0;
        if numel(tmp_index_na_)==0:
            tmp_index_nb_ = efind(b_==cup);
            if numel(tmp_index_nb_)> 0:
                nb = int(torch.min(tmp_index_nb_).item());
                index_nb_from_ncup_[ncup] = nb;
            #end;%if (numel(tmp_index_nb_)> 0;
        #end;%if (numel(tmp_index_na_)==0;
    #end;%for ncup in range(n_cup):
    index_na_from_ncup_ = index_na_from_ncup_[efind(index_na_from_ncup_>=0)];
    index_nb_from_ncup_ = index_nb_from_ncup_[efind(index_nb_from_ncup_>=0)];
    return(cup_,index_na_from_ncup_,index_nb_from_ncup_);

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;

#%%%%%%%%;
#% Warning, This is a brute-force O(n^2) unique function. ;
#% The only goal is for this to mimic the matlab unique. ;
#%%%%%%%%;
def unique_0(a_):
    u_a_,index_nu_a_from_na_,n_u_a_ = torch.unique(a_.ravel(),return_inverse=True,return_counts=True,dim=0);
    n_u_a = numel(u_a_);
    index_nu_a_from_na_ = index_nu_a_from_na_.to(dtype=torch.int32);
    index_na_from_nu_a_ = torch.zeros(n_u_a).to(dtype=torch.int32);
    for nu_a in range(n_u_a):
        u_a = u_a_[nu_a].item();
        na = int(torch.min(efind(a_==u_a)).item());
        index_na_from_nu_a_[nu_a] = na;
    #end;%for nu_a=0:n_u_a-1;
    return(u_a_,index_na_from_nu_a_,index_nu_a_from_na_);

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;

#%%%%%%%%;
#% Warning, This is a brute-force O(n^2) setdiff function. ;
#% The only goal is for this to mimic the matlab setdiff. ;
#%%%%%%%%;
def setdiff_0(a_, b_):
    cup_,index_nb_from_ncup_,index_na_from_ncup_ = union_0(b_,a_);
    setdiff_ = a_.ravel()[index_na_from_ncup_];
    return(setdiff_);

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;


