import numpy as np ; pi = np.pi ; import torch ;
from scipy.io import savemat ;
from scipy.io import loadmat ;
etumrep = lambda a : torch.permute(a,tuple(torch.arange(a.ndim-1,-1,-1).tolist())) ; #<-- matlab-arranged permutation (for savemat). ;

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
