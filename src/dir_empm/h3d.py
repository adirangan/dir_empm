from dir_empm.dir_matlab_macros import * ;
from scipy.special import jv

def h3d(kd):
    r'''
    This function provides a formula for the bandlimited integral of a plane-wave in 3d.
    It can also be used to calculate the bandlimited-inner-product of two plane-waves in 3d.

    More specifically, this function calculates the following integral:
    \[ \int_{k=0}^{k=K} \int_{\epolara=0}^{\epolara=\pi} \int_{\eazimub=0}^{\eazimub=2\pi} \fT(\vec{\kappa}) d\eazimub \sin(\epolara)d\epolara k^{2}\dk \]
    normalized so that the domain has size $4\pi/3$.
    In this integrand $\fT(\vec{\kappa})$ is a plane-wave:
    \[ \fT(\vec{\kappa}) = \exp(+i2\pi \vec{\kappa}\cdot\vec{\delta}_{T}), \]
    defined by the displacement-vector $\vec{\delta}_{T} \in \Real^{3}$.

    The output is:
    \[ (4\pi) \frac{ \sin(\kd) - \kd\cos(\kd) }{ \kd^{3} }, \]
    where the intermediate quantity $\kd$ is:
    \[ \kd = 2pi K \delta_{T}, \]
    where $\delta_{T}$ is the magnitude of the displacement $\vec{\delta}_{T}$.

    This function can also be used to calculate the following bandlimited inner-product:
    \[ \int_{k=0}^{k=K} \int_{\epolara=0}^{\epolara=\pi} \int_{\eazimub=0}^{\eazimub=2\pi} \fS(\vec{\kappa})^{\dagger} \fM(\vec{\kappa}) d\eazimub \sin(\epolara)d\epolara k^{2}\dk \]
    where $\fS(\vec{\kappa})$ and $\fM(\vec{\kappa})$ are plane-waves defined via:
    \[ \fS(\vec{\kappa}) = \exp(+i2\pi \vec{\kappa}\cdot\vec{\delta}_{S}), \]
    \[ \fM(\vec{\kappa}) = \exp(+i2\pi \vec{\kappa}\cdot\vec{\delta}_{M}), \]
    each defined by their own displacement-vector.
    Here we simply multiply the two plane-waves together to get $\fT$ defined via:
    \[ \vec{\delta}_{T} & = & \vec{\delta}_{S} - \vec{\delta}_{M}. \]
    '''   
    if np.isscalar(kd): kd = torch.tensor([kd]).to(dtype=torch.float32) ;
    output = (4*pi) * (torch.sin(kd) - kd*torch.cos(kd))/torch.maximum(torch.tensor([1e-12]),kd**3) ;
    return output ;
