from dir_empm.dir_matlab_macros import * ;
from dir_empm.niko_ctf import niko_ctf ;

def CTF_k_p_wkC__from_params_0(
        n_k_p_r,
        k_p_r_,
        n_w_,
        n_CTF,
        Voltage_C_,
        DefocusU_C_,
        DefocusV_C_,
        DefocusAngle_C_,
        SphericalAberration_C_,
        AmplitudeContrast_C_,
        n_Pixels_across,
        Pixel_Spacing,
):

    n_w_max = int(torch.max(n_w_).item()); n_w_sum = int(torch.sum(n_w_).item());
    k_c_0_ = torch.zeros(n_w_sum).to(dtype=torch.float64);
    k_c_1_ = torch.zeros(n_w_sum).to(dtype=torch.float64);
    if numel_unique(n_w_)==1:
        k_c_0_wk__ = torch.zeros(mtr((n_w_sum,n_k_p_r))).to(dtype=torch.float64);
        k_c_1_wk__ = torch.zeros(mtr((n_w_sum,n_k_p_r))).to(dtype=torch.float64);
        gamma_z_ = 2*pi*torch.arange(n_w_max).to(dtype=torch.float64)/np.maximum(1,n_w_max);
        k_c_0_wk__ = mmmm(torch.reshape(torch.cos(gamma_z_),mtr((n_w_max,1))),torch.reshape(2*pi*k_p_r_,mtr((1,n_k_p_r))));
        k_c_0_wk_ = k_c_0_wk__.ravel();
        k_c_1_wk__ = mmmm(torch.reshape(torch.sin(gamma_z_),mtr((n_w_max,1))),torch.reshape(2*pi*k_p_r_,mtr((1,n_k_p_r))));
        k_c_1_wk_ = k_c_1_wk__.ravel();
        k_c_0_ = k_c_0_wk_;
        k_c_1_ = k_c_1_wk_;
    #end;%if numel_unique(n_w_)==1:
    if numel_unique(n_w_)>1:
        na = 0 ;
        for nk_p_r in range(n_k_p_r):
            r = 2 * pi * k_p_r_[nk_p_r].item() ;
            for nw in range(int(n_w_[nk_p_r].item())):
                gamma = (2 * pi * nw) / max(1, int(n_w_[nk_p_r].item())) ;
                k_c_0_[na] = r * np.cos(gamma) ;
                k_c_1_[na] = r * np.sin(gamma) ;
                na += 1 ;
            #end;%for nw in range(int(n_w_[nk_p_r].item()));
        #end;%for nk_p_r in range(n_k_p_r);
        if na != n_w_sum: print('Warning, na!=n_w_sum in CTF_k_p_wkC__from_params_0.py') ;
    #end;%if numel_unique(n_w_)>1:

    CTF_k_p_wkC__ = torch.zeros(mtr((n_w_sum,n_CTF))).to(dtype=torch.float64);
    for nCTF in range(n_CTF):
        CTF_Spherical_Aberration = SphericalAberration_C_[nCTF];#% spherical aberration of the lens in mm ;
        CTF_Spherical_Aberration=CTF_Spherical_Aberration*(1.0e7);#% convert into Angstroms ;
        CTF_Voltage_kV = Voltage_C_[nCTF];#% voltage in kVolts ;
        CTF_Voltage_1V=CTF_Voltage_kV*1000.0 ;#% convert into Volts ;
        CTF_lambda = 12.2643247/np.maximum(1e-24,np.sqrt(CTF_Voltage_1V+CTF_Voltage_1V**2 * 0.978466e-6));#% electron wavelength in Angstroms ;
        CTF_Defocus_U = DefocusU_C_[nCTF];#% defocus values (in Angstroms) ;
        CTF_Defocus_V = DefocusV_C_[nCTF];#% defocus values (in Angstroms) ;
        CTF_Defocus_Angle = DefocusAngle_C_[nCTF];#% angle of astigmatism ;
        CTF_Defocus_Angle = CTF_Defocus_Angle*pi/180.0;#% convert into radians ; #%<-- may already be in radians! make sure not to convert twice!;
        CTF_Amplitude_Contrast = AmplitudeContrast_C_[nCTF];#% CTF_Amplitude Contrast ;
        tmp_w1=np.sqrt(1.0-CTF_Amplitude_Contrast**2);#% weights for the amplitude and phase contrasts in CTF ;
        tmp_w2=CTF_Amplitude_Contrast;#% weights for the amplitude and phase contrasts in CTF ;
        #%  CTF_Object_Pixel_Size = CTF_Detector_Pixel_Size/np.maximum(1e-24,CTF_Magnification);
        CTF_Object_Pixel_Size = Pixel_Spacing;#% pixel size of the scanner in physical space in Angstroms ;
        CTF_lambda_per_box = CTF_lambda/np.maximum(1e-24,n_Pixels_across*CTF_Object_Pixel_Size);#% n_Pixels_across*CTF_Object_Pixel_Size is the box size in Angstroms ;
        CTF_k_p_wkC__[nCTF,:] = -niko_ctf(CTF_Spherical_Aberration,CTF_lambda,tmp_w1,tmp_w2,CTF_Defocus_U,CTF_Defocus_V,CTF_Defocus_Angle,CTF_lambda_per_box/pi,k_c_0_,k_c_1_);
    #end;%for nCTF=0:n_CTF-1;

    return CTF_k_p_wkC__;
