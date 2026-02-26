from dir_empm.dir_matlab_macros import * ;

'''
function M_p_ = rotate_p_to_p_fftw(n_r,n_w_,n_A,S_p_,gamma);
ic = 0;
for nr=0:n_r-1;
ic = ic + n_w_(1+nr);
end;%for nr=0:n_r-1;
assert(ic==n_A);
M_p_ = zeros(size(S_p_));
n_w_max = n_w_(1+n_r-1);
C_ = zeros(n_w_max,1);
for nw=0:n_w_max-1;
q = nw - n_w_max/2;
C =  +cos(q*gamma) + i* -sin(q*gamma) ;
C_(1+nw) = C;
end;%for nw=0:n_w_max-1;
ic=0;
for nr=0:n_r-1;
if (n_w_(1+nr)>0);
fftw_out_ = fft(S_p_(1+ic + (0:n_w_(1+nr)-1)));
for nq=0:n_w_(1+nr)-1;
q = nq;
if (q>n_w_(1+nr)/2-1);
q = q - n_w_(1+nr);
end;%!if (q.ge.n_w_(1+nr)/2-1);
C = C_(1+n_w_max/2 + q);
fftw_out_(1+nq) = fftw_out_(1+nq) * C;
end;%for nq=0:n_w_(1+nr)-1;
fftw_0in_ = ifft(fftw_out_);
M_p_(1+ic + (0:n_w_(1+nr)-1)) = fftw_0in_;
ic = ic + n_w_(1+nr);
end;%if (n_w_(1+nr)>0);
end;%for nr=0:n_r-1;
'''
def rotate_p_to_p_fftw_using_numpy(n_r, n_w_, n_A, S_p_, gamma):
    ic = sum(n_w_[:n_r])
    assert ic == n_A

    M_p_ = np.zeros_like(S_p_)
    n_w_max = n_w_[n_r - 1]
    C_ = np.zeros(n_w_max, dtype=complex)

    for nw in range(n_w_max):
        q = nw - n_w_max / 2
        C = np.cos(q * gamma) + 1j * -np.sin(q * gamma)
        C_[nw] = C

    ic = 0
    for nr in range(n_r):
        if n_w_[nr] > 0:
            fftw_out_ = np.fft.fft(S_p_[ic:ic + n_w_[nr]])
            for nq in range(n_w_[nr]):
                q = nq
                if q > n_w_[nr] / 2 - 1:
                    q -= n_w_[nr]
                C = C_[int(n_w_max / 2 + q)]
                fftw_out_[nq] *= C
            fftw_0in_ = np.fft.ifft(fftw_out_)
            M_p_[ic:ic + n_w_[nr]] = fftw_0in_
            ic += n_w_[nr]
            
    return M_p_
