from dir_empm.dir_matlab_macros import * ;

r'''
function output = niko_ctf(Cs,var_lambda,w1,w2,df1,df2,angast,thetatr,l,m);
rad = l.^2+m.^2; rad = dsqrt(rad);
angle = rad.*thetatr;
angspt=datan2(m,l);
c1=2.0d0.*pi.*angle.*angle./(2.0d0.*var_lambda);
c2=-c1.*Cs.*angle.*angle./2.0d0;
angdif=angspt-angast;
ccos=dcos(2.0d0.*angdif);
df = 0.5d0.*(df1+df2+ccos.*(df1-df2));
chi=c1.*df+c2;
ctfv=-w1.*dsin(chi)-w2.*dcos(chi);
output = ctfv;

function output = dsqrt(input); 
output = sqrt(input);
function output = datan2(input1,input2); 
output = atan2(input1,input2);
function output = dcos(input); 
output = cos(input);
function output = dsin(input); 
output = sin(input);
'''

def niko_ctf(
        Cs,
        var_lambda,
        w1,
        w2,
        df1,
        df2,
        angast,
        thetatr,
        l_,
        m_,
):
    rad_ = l_**2 + m_**2; rad_ = torch.sqrt(rad_);
    angle_ = rad_*thetatr;
    angspt_ = torch.atan2(m_,l_);
    c1_ = 2.0*pi * angle_*angle_ / (2.0*var_lambda);
    c2_ = -c1_ * Cs * angle_*angle_ / 2.0;
    angdif_ = angspt_ - angast;
    ccos_ = torch.cos(2.0*angdif_);
    df_ = 0.5*(df1 + df2 + ccos_*(df1-df2));
    chi_ = c1_*df_ + c2_;
    ctfv_ = -w1*torch.sin(chi_) - w2*torch.cos(chi_);
    output_ = ctfv_;
    return(output_,chi_);

