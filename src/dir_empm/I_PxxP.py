from dir_empm.dir_matlab_macros import * ;
from scipy.special import jv

r'''
function output = I_PxxP(K,phi_M,delta_M,omega_M,phi_S,delta_S,omega_S);
% calculates the integral: ;
% \int_{k=0}^{K} \int_{psi=0}^{2*pi} conj[CTF * S] * M dpsi * kdk ;
% where: ;
% CTF = 2*k*cos(psi - phi_S) ;
% S   = exp( +i*2*pi*k*delta_S * cos(psi - omega_S) ) ;
% M   = 2*k*cos(psi - phi_M) * exp( +i*2*pi*k*delta_M * cos(psi - omega_M) ) ;
delta_M_ = delta_M * [cos(omega_M) ; sin(omega_M)] ;
delta_S_ = delta_S * [cos(omega_S) ; sin(omega_S)] ;
delta_ = delta_M_ - delta_S_ ;
delta = sqrt(delta_(1).^2 + delta_(2).^2) ;
omega = atan2(delta_(2),delta_(1)) ;
phi_p = phi_S + phi_M ;
phi_n = phi_S - phi_M ;
t = 2*pi*delta ;
tK = t*K ;
b0 = besselj(0,tK);
b2 = besselj(2,tK);
b4 = besselj(4,tK);
%mode_2 = ( -4*pi*cos(phi_p - 2*omega) * K^3 * besselj(3,tK) ) / t ;
%mode_2 = -4*pi*cos(phi_p - 2*omega) * K^4 * ( besselj(2,tK) + besselj(4,tK) ) / 6;
mode_2 = -4*pi*cos(phi_p - 2*omega) * K^4 * ( b2 + b4 )/6;
%mode_0 = 4*pi*cos(phi_n) * ( tK^3 * besselj(1,tK) - 2*tK^2 *besselj(2,tK) ) / t^4 ;
%mode_0 = 4*pi*cos(phi_n) * K^4 * ( besselj(0,tK)/4 + besselj(2,tK)/6 - besselj(4,tK)/12 ) ;
mode_0 = 4*pi*cos(phi_n) * K^4 * ( b0/4 + b2/6 - b4/12 );
output = mode_0 + mode_2 ;
'''

def I_PxxP(K, phi_M, delta_M, omega_M, phi_S, delta_S, omega_S):
    delta_M_ = delta_M * torch.tensor([np.cos(omega_M), np.sin(omega_M)]).to(dtype=torch.float32) ;
    delta_S_ = delta_S * torch.tensor([np.cos(omega_S), np.sin(omega_S)]).to(dtype=torch.float32) ;
    delta_ = delta_M_ - delta_S_ ;
    delta = np.sqrt(delta_[0].item()**2 + delta_[1].item()**2) ;
    omega = np.arctan2(delta_[1].item(), delta_[0].item()) ;
    phi_p = phi_S + phi_M ;
    phi_n = phi_S - phi_M ;
    t = 2 * pi * delta ;
    tK = t * K ;
    b0 = jv(0, tK) ;
    b2 = jv(2, tK) ;
    b4 = jv(4, tK) ;
    mode_2 = -4 * pi * np.cos(phi_p - 2 * omega) * K**4 * (b2 + b4) / 6 ;
    mode_0 = 4 * pi * np.cos(phi_n) * K**4 * (b0 / 4 + b2 / 6 - b4 / 12) ;
    output = mode_0 + mode_2 ;
    return output ;
