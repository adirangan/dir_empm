from dir_empm.dir_matlab_macros import * ;
from dir_empm.niko_ctf import niko_ctf ;

flag_verbose=1;
str_thisfunction = 'niko_ctf';
if (flag_verbose>0): disp(sprintf(' %% testing %s',str_thisfunction)); #end;
if (flag_verbose>0): disp(sprintf(' %% testing %s using typical values from, say, EMPIAR-10005.',str_thisfunction)); #end;
CTF_Voltage_kV = 300.000000 ;
CTF_Defocus_U_Angstrom = 22174.200000 ;
CTF_Defocus_V_Angstrom = 21393.000000 ;
CTF_Defocus_Angle_degree = 1.600000 ;
CTF_Spherical_Aberration_mm = 2.000000 ;
CTF_Amplitude_Contrast_cosangle = 0.100000 ;
#%%%%%%%%;
CTF_Spherical_Aberration_Angstrom = CTF_Spherical_Aberration_mm*(10.0**7.0); #%<-- spherical aberration in Angstroms. ;
CTF_Voltage_1V = CTF_Voltage_kV*1000.0 ; #%<-- voltage in Volts. ;
CTF_lambda_Angstrom = 12.2643247/np.sqrt(CTF_Voltage_1V+CTF_Voltage_1V**2*0.978466*0.000001); #%<-- electron wavelength in Angstroms, close to 0.02 Angstrom (see Kirkland p17). ;
CTF_Defocus_Angle_radian = CTF_Defocus_Angle_degree*pi/180.0; #%<-- convert into radians. ;
w1 = np.sqrt(1.0-CTF_Amplitude_Contrast_cosangle**2); #%<-- phase-contrast: corresponding sin(angle). ;
w2 = CTF_Amplitude_Contrast_cosangle; #%<-- amplitude-contrast: corresponding cos(angle). ;
Pixel_Spacing_Angstrom = 1.20;
CTF_Object_Pixel_Size_Angstrom = Pixel_Spacing_Angstrom; #%<-- pixel size of the scanner in physical space in Angstroms ;
n_Pixels_across = 256;
Box_Size_Angstrom = n_Pixels_across*CTF_Object_Pixel_Size_Angstrom; #%<-- box size in Angstroms. ;
CTF_lambda_BoxSide = CTF_lambda_Angstrom/Box_Size_Angstrom; #%<-- wavelengths per box. ;
#%%%%%%%%;
Cs = CTF_Spherical_Aberration_Angstrom ;
var_lambda = CTF_lambda_Angstrom ;
df1_Angstrom = CTF_Defocus_U_Angstrom ;
df2_Angstrom = CTF_Defocus_V_Angstrom ;
angast = CTF_Defocus_Angle_radian ;
thetatr = CTF_lambda_BoxSide/pi ;
#%%%%%%%%;
n_k = 128;
k_p_r_ = torch.linspace(1,48.0/(2*pi),n_k).to(dtype=torch.float64);
rad_ = 2*pi*k_p_r_ ; #%<-- 2*pi*(wavenumber/2 in sidelength-2 box). ;
wavenumber_per_Angstrom_ = 2*pi*k_p_r_/pi/Box_Size_Angstrom ; #%<-- wavenumber_per_Angstrom ;
electron_wavelength_Angstrom = CTF_lambda_Angstrom ; #%<-- electron_wavelength_Angstrom ;
angle_ = rad_*thetatr ; #%<-- 2*pi*k_p_r * CTF_lambda_Angstrom/Box_Size_Angstrom / pi ; #%<-- wavenumber_per_Angstrom * electron_wavelength_Angstrom. ;
k_p_w_ = 2*pi*torch.arange(n_k).to(dtype=torch.float64)/np.maximum(1,n_k);
angspt_ = k_p_w_ ; #%<-- polar-angle in frequency-space. ;
l_ = 2*pi*k_p_r_*torch.cos(k_p_w_) ; #%<-- 2*pi*(wavenumber/2 in sidelength-2 box). ;
m_ = 2*pi*k_p_r_*torch.sin(k_p_w_) ; #%<-- 2*pi*(wavenumber/2 in sidelength-2 box). ;
c1_ = 2*pi*angle_**2/np.maximum(1e-32,2*var_lambda) ; #%<-- pi * wavenumber_per_Angstrom**2 * electron_wavelength_Angstrom. ;
c2_ = -c1_*Cs*angle_**2/2 ; #%<-- - pi/2 * wavenumber_per_Angstrom**4 * electron_wavelength_Angstrom**3 * CTF_Spherical_Aberration_Angstrom ;
angdif_ = k_p_w_ - CTF_Defocus_Angle_radian ; #%<-- in radians. ;
df_Angstrom_ = 0.5*(df1_Angstrom+df2_Angstrom + torch.cos(2*angdif_)*(df1_Angstrom-df2_Angstrom)) ; #%<-- matches CTFFIND4 and cryojax. ;
chi_tmp_ = (pi*wavenumber_per_Angstrom_**2*electron_wavelength_Angstrom)*df_Angstrom_ - CTF_Spherical_Aberration_Angstrom*(pi/2)*wavenumber_per_Angstrom_**4*electron_wavelength_Angstrom**3 ; #%<-- this matches CTFFIND4 and cryojax. ;
#%%%%%%%%;
(
    ctfv_ctf_,
    chi_ctf_,
) = niko_ctf(
    Cs,
    var_lambda,
    w1,
    w2,
    df1_Angstrom,
    df2_Angstrom,
    angast,
    thetatr,
    l_,
    m_,
);
fnorm_disp(flag_verbose,'chi_ctf_',chi_ctf_,'chi_tmp_',chi_tmp_,'%<-- should be zero (up to single-digit precision)');
#%%%%%%%%;
if (flag_verbose>0): disp(sprintf(' %% testing %s using synthetic values.',str_thisfunction)); #end;
Cs = 1.0;
var_lambda = 2;
w1 = +3*np.sqrt(2); w2 = -5*np.sqrt(2);
df1 = -1.0; df2 = +1.0;
angast = 3*pi/4;
thetatr = 1e-3;
l = 1e+3/np.sqrt(2); m = 1e+3/np.sqrt(2);
torch_l = torch.zeros(1).to(dtype=torch.float64); torch_l[0] = l;
torch_m = torch.zeros(1).to(dtype=torch.float64); torch_m[0] = m;
#%%%%;
rad = np.sqrt(l**2 + m**2); #%<-- rad = l**2+m**2; rad = dsqrt(rad);
angle = rad*thetatr;
angspt = np.atan2(m,l); #%<-- angspt=datan2(m,l);
c1 = 2.0*pi*angle*angle/np.maximum(1e-32,2.0*var_lambda);
c2 = -c1*Cs*angle*angle/2.0;
angdif = angspt-angast;
ccos = np.cos(2.0*angdif); #%<-- ccos=dcos(2.0*angdif);
df = 0.5*(df1+df2+ccos*(df1-df2));
chi=c1*df+c2;
ctfv = -w1*np.sin(chi)-w2*np.cos(chi); #%<-- ctfv=-w1*dsin(chi)-w2*dcos(chi);
output = ctfv;
#%%%%;
(
    ctfv_ctf,
    chi_ctf,
) = niko_ctf(
    Cs,
    var_lambda,
    w1,
    w2,
    df1,
    df2,
    angast,
    thetatr,
    torch_l,
    torch_m,
);
torch_tmp = torch.zeros(1);
torch_tmp[0] = 2.0;
fnorm_disp(flag_verbose,'ctfv_ctf',ctfv_ctf,'2.0',torch_tmp,'%<-- should be zero (up to single-digit precision)');
torch_tmp[0] = pi/4;
fnorm_disp(flag_verbose,'chi_ctf',chi_ctf,'pi/4',torch_tmp,'%<-- should be zero (up to single-digit precision)');
#%%%%%%%%;
