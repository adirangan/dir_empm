import numpy as np ; pi = np.pi ; import torch ;
from scipy.io import savemat ;
from scipy.io import loadmat ;
etumrep = lambda a : torch.permute(a,tuple(torch.arange(a.ndim-1,-1,-1).tolist())) ; #<-- matlab-arranged permutation (for savemat). ;

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


