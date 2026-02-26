from dir_empm.dir_matlab_macros import * ;

def knn_cluster_CTF_k_p_r_kC__1(
        parameter=None,
        n_k_p_r=None,
        k_p_r_=None,
        weight_2d_k_p_r_=None,
        n_CTF=None,
        CTF_k_p_r_kC__=None,
):
    #%%%%%%%%;
    #% Groups isotropic CTF_k_p_r_kC__ into clusters. ;
    #% The clustering proceeds iteratively, ;
    #% at each step choosing the CTF_k_p_r_k_ with the lowest norm, ;
    #% and then assigning to it the other CTFs with relative l2-difference less than tolerance_cluster. ;
    #%%%%%%%%;

    str_thisfunction = 'knn_cluster_CTF_k_p_r_kC__1';
    tolerance_machine = 1e-6;

    if isempty(parameter): parameter = {'type':'parameter'}; #end;%if isempty(parameter);
    #%%%%%%%%;
    if 'flag_verbose' not in parameter: parameter['flag_verbose'] = 0; #end; %<-- parameter_bookmark. ;
    flag_verbose = parameter['flag_verbose'];
    if 'tolerance_master' not in parameter: parameter['tolerance_master'] = 1e-2; #end; %<-- parameter_bookmark. ;
    tolerance_master = parameter['tolerance_master'];
    if 'tolerance_cluster' not in parameter: parameter['tolerance_cluster'] = tolerance_master; #end; %<-- parameter_bookmark. ;
    tolerance_cluster = parameter['tolerance_cluster'];

    if (flag_verbose>0): disp(sprintf(' %% [entering %s]',str_thisfunction)); #end;

    CTF_k_p_l2_C_ = torch.zeros(n_CTF).to(dtype=torch.float32);
    CTF_k_p_l2_C_ = torch.sqrt(torch.sum(torch.abs(CTF_k_p_r_kC__)**2 * torch.reshape(weight_2d_k_p_r_,mtr((n_k_p_r,1))),1-0)).ravel(); assert(numel(CTF_k_p_l2_C_)==n_CTF);
    
    index_ncluster_from_nCTF_ = torch.zeros(n_CTF).to(dtype=torch.int32);
    niteration=0;ncluster=0;
    index_unassigned_ = torch.arange(n_CTF).to(dtype=torch.int32);
    flag_continue = numel(index_unassigned_)>0;
    while flag_continue:
        n_unassigned = numel(index_unassigned_);
        if (flag_verbose>1): disp(sprintf(' %% niteration %d: n_unassigned %d',niteration,n_unassigned)); #end;
        tmp_l2_min,tmp_index_min = torch.min(CTF_k_p_l2_C_[index_unassigned_],dim=0-0);
        if (flag_verbose>1): disp(sprintf(' %% min at index %d (%d) --> %0.6f',tmp_index_min,index_unassigned_(1+tmp_index_min),tmp_l2_min)); #end;
        tmp_i8_index_rhs_ = matlab_index_2d_0(n_k_p_r,':',n_CTF,index_unassigned_[tmp_index_min]);
        CTF_cen_k_p_r_k_ = CTF_k_p_r_kC__.ravel()[tmp_i8_index_rhs_];
        l2_cen = CTF_k_p_l2_C_[index_unassigned_[tmp_index_min]];
        assert(np.abs(l2_cen-tmp_l2_min)<tolerance_machine);
        tmp_i8_index_rhs_ = matlab_index_2d_0(n_k_p_r,':',n_CTF,index_unassigned_);
        tmp_l2_ = torch.sqrt(torch.sum(torch.abs(torch.reshape(CTF_k_p_r_kC__.ravel()[tmp_i8_index_rhs_],mtr((n_k_p_r,n_unassigned))) - torch.reshape(CTF_cen_k_p_r_k_,mtr((n_k_p_r,1))))**2 * torch.reshape(weight_2d_k_p_r_,mtr((n_k_p_r,1))),1-0)).ravel(); assert(numel(tmp_l2_)==n_unassigned);
        tmp_index_nearby_ = efind(tmp_l2_/np.maximum(1e-12,tmp_l2_min)<= tolerance_cluster); #%<-- relative difference. ;
        n_nearby = numel(tmp_index_nearby_); #%<-- should be at least 1, including CTF_cen_k_p_r_k_ itself. ;
        if (n_nearby<=0): disp(sprintf(' %% Warning, n_nearby<=0 in knn_cluster_CTF_k_p_r_kC__0')); #end;
        tmp_index_nearby_ = union_0(tmp_index_nearby_,tmp_index_min)[0];
        if (flag_verbose>1): disp(sprintf(' %% niteration %d: n_nearby = %d',niteration,n_nearby)); #end;
        index_ncluster_from_nCTF_[index_unassigned_[tmp_index_nearby_]] = ncluster;
        ncluster = ncluster+1;
        index_unassigned_ = index_unassigned_[setdiff_0(torch.arange(n_unassigned),tmp_index_nearby_)];
        flag_continue = numel(index_unassigned_)> 0;
    #end;%while flag_continue;

    if (flag_verbose>0): disp(sprintf(' %% [finished %s]',str_thisfunction)); #end;
    
    return(
        parameter,
        index_ncluster_from_nCTF_,
    );

