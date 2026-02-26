import torch ;
def i4_torch_arange(
        start=0,
        end=0,
        step=1,
):
    flag_verbose=0;
    str_thisfunction = 'i4_torch_arange';
    if flag_verbose > 0: print(f' %% [entering {str_thisfunction}]');
    i4_arange_ = torch.arange(0).to(dtype=torch.int32) ;
    if step> 0:
        if end> start:
            i4_arange_ = torch.arange(start,end,step);
        #end;%if (end> start);
    #end;%if (step> 0);
    if step< 0:
        if end< start:
            i4_arange_ = torch.arange(start,end,step);
        #end;%if (end< start);
    #end;%if (step< 0);
    if flag_verbose > 0: print(f' %% [finished {str_thisfunction}]');
    return(i4_arange_);
