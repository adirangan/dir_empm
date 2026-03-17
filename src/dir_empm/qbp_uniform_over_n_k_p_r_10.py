from dir_empm.dir_matlab_macros import * ;
from dir_empm.transf_p_to_p import transf_p_to_p ;
from dir_empm.sample_shell_6 import sample_shell_6 ;
from dir_empm.ylgndr_2 import ylgndr_2 ;
from dir_empm.cg_rhs_2 import cg_rhs_2 ;
from scipy.spatial import cKDTree ;
from scipy.sparse import csr_matrix ;
from dir_empm.local_yk_from_yk__ import local_yk_from_yk__ ;

def qbp_uniform_over_n_k_p_r_10(
        qbp_eps=None,
        n_k_p_r=None,
        k_p_r_=None,
        l_max_=None,
        n_w_=None,
        n_M=None,
        M_k_p_wkM__=None,
        index_nCTF_from_nM_=None,
        CTF_k_p_wkC__=None,
        euler_polar_a_M_=None,
        euler_azimu_b_M_=None,
        euler_gamma_z_M_=None,
        image_delta_x_M_=None,
        image_delta_y_M_=None,
        image_I_value_M_=None,
):
    #%%%%%%%%;
    #% Applies quadrature-back-propagation to solve for a_k_Y_yk_. ;
    #% Associates CTF_k_p_wkC__(:,1+index_nCTF_from_nM_(1+nM)) with image M_k_p_wkM__(:,1+nM);
    #% ;
    #% Input: ;
    #% qbp_eps: pseudoinverse threshold used for solving local least-squares for ctf. ; %<-- Bug: this is actually ignored internally, and a value of 1e-12 is hard-coded. Not fixing because we are putting together a draft of the paper. ;
    #% n_k_p_r: integer number of shells. ;
    #% k_p_r_: real array of size n_k_p_r. k-values for each shell. ;
    #% l_max_: integer array of size n_k_p_r. l_max_(1+nk_p_r) is the order used for a_k_Y_yk_ on shell nk_p_r. ;
    #% n_w_: integer array of size n_k_p_r. n_w_(1+nk_p_r) is the number of inplane_gamma_z values recorded at that ring. ;
    #% n_M: integer number of images. ;
    #% M_k_p_wkM__: complex array of size (n_w_sum,n_M). stack of images in k_p_ format. ;
    #% index_nCTF_from_nM_: integer array of size n_M. index_nCTF_from_nM_(1+nM) is the (base 0) CTF_index used for image M_k_p_wkM__(:,1+nM). ;
    #%             This can be empty or set to 1, in which case the same CTF_k_p_ will be used for each image. ;
    #% CTF_k_p_wkC__: complex array of size(n_w_sum,n_CTF). stack of ctf-functions in k_p_ format. ;
    #%            If index_nCTF_from_nM_ is empty or set to 1, then we assume this contains only a single CTF_k_p_, ;
    #%            which will then be used for all images. ;
    #% euler_polar_a_M_: real array of size n_M. polar_a used for each image ;
    #% euler_azimu_b_M_: real array of size n_M. azimu_b used for each image ;
    #% euler_gamma_z_M_: real array of size n_M. gamma_z used for each image ;
    #% image_delta_x_M_: real array of size n_M. delta_x used for each image ;
    #% image_delta_y_M_: real array of size n_M. delta_y used for each image ;
    #% image_I_value_M_: real array of size n_M. I_value used for each image ;
    #% ;
    #% Output: ;
    #% a_k_Y_yk_: complex array of size n_y_sum. output functions in k_Y_ format. ;
    #%%%%%%%%;
    
    tolerance_machine = 1e-6;
    str_thisfunction = 'qbp_uniform_over_n_k_p_r_10';

    flag_verbose=0;
    if (flag_verbose>0): disp(sprintf(' %% [entering %s]',str_thisfunction)); #end;

    n_w_ = n_w_.ravel(); n_w_max = int(torch.max(n_w_).item()); n_w_sum = int(torch.sum(n_w_).item()); n_w_csum_ = cumsum_0(n_w_);
    if (n_w_sum!=n_w_max*n_k_p_r): disp(sprintf(' %% Warning, n_w_sum %d ~= n_w_max*n_k_p_r %d*%d in %s',n_w_sum,n_w_max,n_k_p_r,str_thisfunction)); #end;
    assert(n_w_sum==n_w_max*n_k_p_r);

    if isempty(qbp_eps): qbp_eps = 1e-3; #end;
    if isempty(image_delta_x_M_): image_delta_x_M_ = torch.zeros(n_M).to(dtype=torch.float32) #end;
    if isempty(image_delta_y_M_): image_delta_y_M_ = torch.zeros(n_M).to(dtype=torch.float32) #end;
    if isempty(image_I_value_M_): image_I_value_M_ = torch.ones(n_M).to(dtype=torch.float32); #end;
    if isempty(index_nCTF_from_nM_): index_nCTF_from_nM_ = torch.zeros(n_M).to(dtype=torch.int32) #end;
    if isempty(CTF_k_p_wkC__): CTF_k_p_wkC__ = torch.ones(n_w_sum).to(dtype=torch.float32); #end;
    if (qbp_eps>1): qbp_eps = np.maximum(1e-12,0.1**(qbp_eps)); #end; %<-- convert qbp_eps from nl10 scale to explicit value. ;

    #%%%%%%%%;
    #% construct CTF of same size as images. ;
    #%%%%%%%%
    n_CTF = size(CTF_k_p_wkC__,1);
    if (size(CTF_k_p_wkC__,0)==n_k_p_r):
        CTF_k_p_r_kC__ = CTF_k_p_wkC__;
        CTF_k_p_wkC__ = torch.reshape(torch.tile(torch.reshape(CTF_k_p_r_kC__,mtr((1,n_k_p_r,n_CTF))),mtr((n_w_max,1,1))),mtr((n_w_sum,n_CTF))).to(dtype=torch.float32);
    #end;%if (size(CTF_k_p_wkC__,1)==n_k_p_r);
    #%%%%%%%%;
    tmp_i8_index_rhs_ = matlab_index_2d_0(n_w_sum,':',n_CTF,index_nCTF_from_nM_);
    CTF_k_p_wkM__ = torch.reshape(CTF_k_p_wkC__.ravel()[tmp_i8_index_rhs_],mtr((n_w_sum,n_M))).to(dtype=torch.float32);
    #%%%%%%%%;

    T_M_k_p_wkM__ = M_k_p_wkM__;
    if ( (fnorm(image_delta_x_M_)>0) or (fnorm(image_delta_y_M_)>0) or (fnorm(image_I_value_M_-torch.ones(n_M).to(dtype=torch.float32))>tolerance_machine) ):
        T_M_k_p_wkM__ = (torch.reshape(transf_p_to_p(n_k_p_r,k_p_r_,n_w_,n_w_sum,M_k_p_wkM__,+image_delta_x_M_,+image_delta_y_M_).to(dtype=torch.complex64),mtr((n_w_sum,n_M))) * torch.reshape(image_I_value_M_.to(dtype=torch.float32),mtr((1,n_M))) ).to(dtype=torch.complex64) ;
    #end;%if ( (fnorm(image_delta_x_M_)>0) | (fnorm(image_delta_y_M_)>0) | (fnorm(image_I_value_M_-ones(n_M,1))>0) );

    l_max_ = l_max_[:n_k_p_r].to(dtype=torch.int32); l_max_max = int(torch.max(l_max_).item());
    n_y_ = (1+l_max_)**2; n_y_max = int(torch.max(n_y_).item()); n_y_sum = int(torch.sum(n_y_).item()); n_y_csum_ = cumsum_0(n_y_);
    quad_k_eq_d = np.sqrt(4*pi/np.maximum(1,n_y_max));
    if (flag_verbose>0): disp(sprintf(' %% quad_k_eq_d %0.6f',quad_k_eq_d)); #end;

    #%%%%%%%%;
    tmp_t = tic();
    str_L = 'L'; flag_uniform_over_polar_a = 0;
    (
        n_q,
        k_p_azimu_b_q_,
        k_p_polar_a_q_,
        weight_shell_q_,
        k_c_0_q_,
        k_c_1_q_,
        k_c_2_q_,
        n_polar_a,
        polar_a_,
        n_azimu_b_,
    ) = sample_shell_6(
        1.0,
        quad_k_eq_d,
        str_L,
        flag_uniform_over_polar_a,
    ) ;
    tmp_t = toc(tmp_t);
    if (flag_verbose>0): disp(sprintf(' %% sample_shell_6: %0.6fs',tmp_t)); #end;
    k_c_q3__ = torch.reshape(torch.concatenate( ( k_c_0_q_.ravel() , k_c_1_q_.ravel() , k_c_2_q_.ravel() ) , 0 ),mtr((n_q,n_3)));
    k_p_polar_a_unique_,index_unique_,index_return_ = unique_0(k_p_polar_a_q_);
    n_polar_a_unique = numel(k_p_polar_a_unique_);
    #%%%%%%%%;

    #%%%%%%%%;
    if 'sqrt_2lp1_' not in locals(): sqrt_2lp1_=None; #end;
    if 'sqrt_2mp1_' not in locals(): sqrt_2mp1_=None; #end;
    if 'sqrt_rat0_m_' not in locals(): sqrt_rat0_m_=None; #end;
    if 'sqrt_rat3_lm__' not in locals(): sqrt_rat3_lm__=None; #end;
    if 'sqrt_rat4_lm__' not in locals(): sqrt_rat4_lm__=None; #end;
    tmp_t = tic();
    parameter_ylgndr = {'type':'ylgndr'};
    parameter_ylgndr['flag_verbose'] = 0;
    parameter_ylgndr['flag_d'] = 0;
    parameter_ylgndr['flag_dd'] = 0;
    (
        parameter_ylgndr,
        d0y_jlm___,
        _,
        _,
        sqrt_2lp1_,
        sqrt_2mp1_,
        sqrt_rat0_m_,
        sqrt_rat3_lm__,
        sqrt_rat4_lm__,
    ) = ylgndr_2(
        parameter_ylgndr,
        l_max_max,
        torch.cos(k_p_polar_a_unique_),
        sqrt_2lp1_,
        sqrt_2mp1_,
        sqrt_rat0_m_,
        sqrt_rat3_lm__,
        sqrt_rat4_lm__,
    );
    tmp_t = toc(tmp_t);
    if (flag_verbose>0): disp(sprintf(' %% ylgndr_2: %0.6fs',tmp_t)); #end;
    #%%%%%%%%;

    #%%%%%%%%;
    #% This version involves preliminary inflation of m_val_, ;
    #% as well as extraneous exp evaluations, ;
    #% all to avoid memory movement. ;
    #%%%%%%%%;

    tmp_t = tic();
    d0y_lmj___ = torch.permute(d0y_jlm___,mtr(mts((1,2,0)))); #%<-- permutation before inflation is faster. ;
    tmp_i8_index_rhs_ = matlab_index_3d_0(1+l_max_max,':',1+l_max_max,torch.abs(torch.arange(-l_max_max,+l_max_max+1).to(dtype=torch.int32)),n_polar_a_unique,':');
    d0y_jml___ = torch.permute(torch.reshape(d0y_lmj___.ravel()[tmp_i8_index_rhs_],mtr((1+l_max_max,1+2*l_max_max,n_polar_a_unique))),mtr(mts((2,1,0)))); #%<-- retain unique cos(polar_a_{j}) for now.; 
    d0y_jy__ = torch.zeros(mtr((n_polar_a_unique,n_y_max)));
    for nl in range(numel(l_max_)):
        l_max = int(l_max_[nl].item());
        for l_val in range(l_max+1):
            m_val_ = torch.arange(-l_val,+l_val+1).to(dtype=torch.int32);
            tmp_i8_index_lhs_ = matlab_index_2d_0(n_polar_a_unique,':',n_y_max,l_val**2+l_val+m_val_);
            tmp_i8_index_rhs_ = matlab_index_3d_0(n_polar_a_unique,':',1+2*l_max_max,l_max_max+m_val_,1+l_max_max,l_val);
            #d0y_jy__.ravel()[tmp_i8_index_lhs_] = d0y_jml___.ravel()[tmp_i8_index_rhs_];
            d0y_jy__ = matlab_assign_lhs_from_rhs_(d0y_jy__,tmp_i8_index_lhs_, d0y_jml___.ravel()[tmp_i8_index_rhs_]);
        #end;%for l_val=0:l_max;
    #end;%for nl=0:numel(l_max_)-1;
    tmp_t = toc(tmp_t);
    if (flag_verbose>0): disp(sprintf(' %% d0y_jy__: %0.6fs',tmp_t)); #end;

    tmp_t = tic();
    tmp_i8_index_rhs_ = matlab_index_2d_0(n_polar_a_unique,index_return_,n_y_max,':');
    d0y_qy__ = torch.reshape(d0y_jy__.ravel()[tmp_i8_index_rhs_],mtr((n_q,n_y_max)));
    tmp_t = toc(tmp_t);
    if (flag_verbose>0): disp(sprintf(' %% d0y_jy__: %0.6fs',tmp_t)); #end;

    tmp_t = tic();
    m_val_y_ = torch.zeros(n_y_max).to(dtype=torch.int32);
    for nl in range(numel(l_max_)):
        l_max = int(l_max_[nl].item());
        for l_val in range(l_max+1):
            m_val_ = torch.arange(-l_val,+l_val+1).to(dtype=torch.int32);
            m_val_y_[l_val**2+l_val+m_val_] = m_val_.ravel();
        #end;%for l_val=0:l_max;
    #end;%for nl=0:numel(l_max_)-1;
    expimb_qy__ = torch.permute(torch.reshape(torch.exp(+i* (torch.reshape(m_val_y_,mtr((n_y_max,1))) * torch.reshape(k_p_azimu_b_q_,mtr((1,n_q))))),mtr((n_y_max,n_q))),mtr(mts((1,0)))).to(dtype=torch.complex64);
    tmp_t = toc(tmp_t);
    if (flag_verbose>0): disp(sprintf(' %% expimb_qy__: %0.6fs',tmp_t)); #end;

    tmp_t = tic();
    d0Y_qy__ = d0y_qy__ * expimb_qy__ / np.sqrt(4*pi);
    tmp_t = toc(tmp_t);
    if (flag_verbose>0): disp(sprintf(' %% d0Y_qy__: %0.6fs',tmp_t)); #end;

    #%%%%;
    tmp_t = tic();
    (
        data_k_p_polar_a_wM__,
        data_k_p_azimu_b_wM__,
        data_k_c_0_wM__,
        data_k_c_1_wM__,
        data_k_c_2_wM__,
    ) = cg_rhs_2(
        n_M,
        n_w_max,
        euler_polar_a_M_,
        euler_azimu_b_M_,
        +euler_gamma_z_M_,
    )[:5];
    n_wM = n_w_max*n_M;
    data_k_c_wM3__ = torch.reshape(torch.concatenate( ( data_k_c_0_wM__.ravel() , data_k_c_1_wM__.ravel() , data_k_c_2_wM__.ravel() ) , 0 ),mtr((n_wM,n_3)));
    tmp_t = toc(tmp_t);
    if (flag_verbose>0): disp(sprintf(' %% cg_rhs_2: %0.6fs',tmp_t)); #end;
    #%%%%;
    tmp_t = tic();
    tmp_tree = cKDTree(k_c_q3__.T.numpy()); tmp_D_, index_nq_from_nwM_ = tmp_tree.query(data_k_c_wM3__.T,k=1); 
    index_nq_from_nwM_ = torch.tensor(index_nq_from_nwM_).to(dtype=torch.int32);
    tmp_t = toc(tmp_t); 
    if (flag_verbose>0): disp(sprintf(' %% knnsearch: %0.6fs',tmp_t)); #end;
    tmp_t = tic();
    np_quad_from_data_qwM__ = np_sparse(index_nq_from_nwM_,torch.arange(n_wM).to(dtype=torch.int32),torch.ones(n_wM).to(dtype=torch.float32),n_q,n_wM);
    n_quad_from_data_q_ = m_npcsr_vm( np_quad_from_data_qwM__ , torch.ones(n_wM).to(dtype=torch.float32) ).to(dtype=torch.int32); assert(numel(n_quad_from_data_q_)==n_q); #%<-- number of data-points per quadrature-point. ;
    tmp_t = toc(tmp_t); 
    if (flag_verbose>0): disp(sprintf(' %% quad_from_data_qwM__: %0.6fs',tmp_t)); #end;
    #data_from_quad_wMq__ = bsxfun(@rdivide,transpose(quad_from_data_qwM__),max(1,transpose(n_quad_from_data_q_))); #%<-- unneeded. ;
    tmp_t = tic();
    CTF_k_p_wMk__ = torch.reshape(torch.permute(torch.reshape(CTF_k_p_wkM__,mtr((n_w_max,n_k_p_r,n_M))),mtr(mts((0,2,1)))),mtr((n_wM,n_k_p_r)));
    CTF2_k_p_qk__ = m_npcsr_mm( np_quad_from_data_qwM__ , torch.abs(CTF_k_p_wMk__)**2 ).to(dtype=torch.float32);
    tmp_t = toc(tmp_t); 
    if (flag_verbose>0): disp(sprintf(' %% CTF2_k_p_qk__: %0.6fs',tmp_t)); #end;
    tmp_t = tic();
    T_M_k_p_wMk__ = torch.reshape(torch.permute(torch.reshape(T_M_k_p_wkM__,mtr((n_w_max,n_k_p_r,n_M))),mtr(mts((0,2,1)))),mtr((n_wM,n_k_p_r)));
    CTF_T_M_k_p_wMk__ = CTF_k_p_wMk__*T_M_k_p_wMk__;
    tmp_t = toc(tmp_t); 
    if (flag_verbose>0): disp(sprintf(' %% CTF_T_M_k_p_wMk__: %0.6fs',tmp_t)); #end;
    #%%%%;
    tmp_t = tic();
    a_k_p_qk__ = m_npcsr_mm( np_quad_from_data_qwM__ , CTF_T_M_k_p_wMk__ ).to(dtype=torch.complex64) / torch.maximum(torch.tensor([tolerance_machine]),CTF2_k_p_qk__);
    tmp_t = toc(tmp_t); 
    if (flag_verbose>0): disp(sprintf(' %% a_k_p_qk__: %0.6fs',tmp_t)); #end;
    tmp_t = tic();
    a_k_Y_yk__ = mmmm( torch.conj((torch.reshape(weight_shell_q_,mtr((n_q,1))) * d0Y_qy__).T).to(dtype=torch.complex64) , a_k_p_qk__.to(dtype=torch.complex64) )
    a_k_Y_yk_ = local_yk_from_yk__(n_k_p_r,l_max_,a_k_Y_yk__);
    tmp_t = toc(tmp_t); 
    if (flag_verbose>0): disp(sprintf(' %% a_k_Y_yk_: %0.6fs',tmp_t)); #end;
    #%%%%;

    #%%%%%%%%;
    if (flag_verbose>0): disp(sprintf(' %% [finished %s]',str_thisfunction)); #end;
    #%%%%%%%%;
    return(
        a_k_Y_yk_,
        n_quad_from_data_q_,
        a_k_p_qk__.ravel(),
    );
