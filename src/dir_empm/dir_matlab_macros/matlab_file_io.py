import os ; import sys ; import numpy as np ; import torch ;

def matlab_which_file_(fname):
    if os.path.isabs(fname) or os.path.dirname(fname): return(fname if os.path.isfile(fname) else None); #end;
    for d in [os.getcwd()] + [p for p in sys.path if p]:
        f = os.path.join(d,fname);
        if os.path.isfile(f): return(f); #end;
    #end;%for d;
    return(None);

def matlab_exist_file(fname): #<-- matlab: exist(fname,'file'): 2 if found. ;
    return(2 if matlab_which_file_(fname) is not None else 0);

def matlab_exist_dir(dirname): #<-- matlab: exist(dirname,'dir'): 7 if found. ;
    return(7 if os.path.isdir(dirname) else 0);

def mkdir(dirname): #<-- matlab: mkdir. ;
    os.makedirs(dirname,exist_ok=True);
    return;

def fopen(fname,permission='r'): #<-- returns a python file object, or -1 if it cannot be opened (as matlab returns fid=-1). ;
    if permission.startswith('r'):
        f = matlab_which_file_(fname);
        if f is None: return(-1); #end;
        fname = f;
    #end;%if permission.startswith('r');
    mode = permission if permission.endswith('b') or 't' in permission else permission.replace('+','')+('+' if '+' in permission else '');
    try: return(open(fname,mode));
    except OSError: return(-1);
    return;

def fscanf(fid,format_str='%f'): #<-- '%s': all words joined without spaces, as matlab does; '%d','%f','%g': a column of numbers. ;
    text = fid.read();
    if format_str.strip()=='%s': return(''.join(text.split())); #end;
    if format_str.strip() in ('%d','%f','%g','%e','%i'):
        return(torch.tensor([float(x) for x in text.split()]).to(dtype=torch.float64));
    #end;% if format_str;
    raise ValueError(f'fscanf: format {format_str!r} not supported');

def fclose(fid):
    if fid in (-1,None): return(-1); #end;
    fid.close(); return(0);

#%%%%%%%%;
#% dummy functions in place of matlab setup. ;
#%%%%%%%%;
def setup_platform_(name):
      print(f' % [entering startup_{name}.m]');
      print(f' % dummy function (empty)');
      print(f' % [finished startup_{name}.m]');
      return;

def setup_eval1(): setup_platform_('eval1');
def setup_access1(): setup_platform_('access1');
def setup_OptiPlex(): setup_platform_('OptiPlex');
def setup_rusty(): setup_platform_('rusty');
def setup_ceph(): setup_platform_('ceph');
