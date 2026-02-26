import numpy as np ; pi = np.pi ; import torch ; import timeit ;

'''
function errrel = fnorm_disp(flag_verbose,str_v0,v0,str_v1,v1,str_postfix);
na=0;
if (nargin<1+na); flag_verbose=[]; end; na=na+1;
if (nargin<1+na); str_v0=[]; end; na=na+1;
if (nargin<1+na); v0=[]; end; na=na+1;
if (nargin<1+na); str_v1=[]; end; na=na+1;
if (nargin<1+na); v1=[]; end; na=na+1;
if (nargin<1+na); str_postfix=[]; end; na=na+1;
if isempty(str_postfix); str_postfix=''; end;

if ndims(v0)~=ndims(v1);
disp(sprintf(' %% Warning, %s ndims(v0) %d ~= %s ndims(v1) %d',str_v0,ndims(v0),str_v1,ndims(v1)));
end;%if ndims(v0)~=ndims(v1);
if numel(v0)~=numel(v1);
disp(sprintf(' %% Warning, %s numel(v0) %d ~= %s numel(v1) %d',str_v0,numel(v0),str_v1,numel(v1)));
end;%if numel(v0)~=numel(v1);
d0_ = size(v0); n_d0 = numel(d0_);
d1_ = size(v1); n_d1 = numel(d1_);
for nd0=0:n_d0-1;
nd1 = nd0;
d0 = d0_(1+nd0); d1 = d1_(1+nd1);
if (d0~=d1);
disp(sprintf(' %% Warning, size(%s,1+%d) %d ~= size(%s,1+%d) %d',str_v0,nd0,d0,str_v1,nd1,d1));
end;%if (d0~=d1);
end;%for nd0=0:n_d0-1;
errrel = fnorm(v0-v1)/max(1e-12,fnorm(v0));
if (flag_verbose>0);
disp(sprintf(' %% %16s %+16.6f vs %16s %+16.6f: r %0.16f%s',str_v0,fnorm(v0),str_v1,fnorm(v1),errrel,str_postfix));
end;%if (flag_verbose>0);
'''
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
