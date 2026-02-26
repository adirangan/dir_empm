from dir_empm.dir_matlab_macros import * ;
from dir_empm.knn_cluster_CTF_k_p_r_kC__1 import knn_cluster_CTF_k_p_r_kC__1 ;

str_thisfunction = 'knn_cluster_CTF_k_p_r_kC__1';
flag_verbose=1;
if (flag_verbose>0): disp(sprintf(' %% testing %s',str_thisfunction)); #end;
parameter = {'type':'parameter'};
parameter['flag_verbose'] = flag_verbose;
parameter['tolerance_cluster'] = 1e-1;
n_k_p_r = 49;
k_p_r_ = torch.linspace(1,7,n_k_p_r).to(dtype=torch.float32);
weight_2d_k_p_r_ = torch.cos(k_p_r_)**2;
n_CTF = 128;
CTF_k_p_r_kC__ = torch.zeros(mtr((n_k_p_r,n_CTF))).to(dtype=torch.float32);
CTF_phi_C_ = torch.linspace(0,pi,n_CTF).to(dtype=torch.float32);
for nCTF in range(n_CTF):
    CTF_k_p_r_kC__[nCTF,:] = torch.sin(k_p_r_ - CTF_phi_C_[nCTF].item());
#end;%for nCTF=0:n_CTF-1;
tmp_t=tic();
(
    parameter,
    index_ncluster_from_nCTF_,
) = knn_cluster_CTF_k_p_r_kC__1(
    parameter,
    n_k_p_r,
    k_p_r_,
    weight_2d_k_p_r_,
    n_CTF,
    CTF_k_p_r_kC__,
);
tmp_t=toc(tmp_t);
if (flag_verbose>0): disp(sprintf(' %% knn_cluster_CTF_k_p_r_kC__1: %0.6fs',tmp_t)); #end;
#%%%%%%%%;
dir_base = '/data/rangan' ;
dir_pymat = dir_base + '/dir_cryoem/dir_rangan_python/dir_pymat' ;
fname_pymat = dir_pymat + '/test_knn_cluster_CTF_k_p_r_kC__1.mat' ;
disp(sprintf(' %% writing fname_pymat: %s',fname_pymat));
matlab_save(
    fname_mat=fname_pymat,
    dictionary_original= {
        "index_ncluster_from_nCTF_":index_ncluster_from_nCTF_,
    },
);
#%%%%%%%%;




