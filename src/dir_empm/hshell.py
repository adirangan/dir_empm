from dir_empm.dir_matlab_macros import * ;
from scipy.special import jv

def hshell(kd):
    r'''
    This function provides a formula for the integral of a plane-wave on a spherical shell in 3d.
    It can also be used to calculate the inner-product of two plane-waves on that same shell.

    More specifically, this function calculates the following integral:
    \[ \int_{\epolara=0}^{\epolara=\pi} \int_{\eazimub=0}^{\eazimub=2\pi} \fT(\vec{\kappa}) d\eazimub \sin(\epolara)d\epolara \]
    for a fixed value of $|\vec{\kappa}|$, all normalized so that the domain has area $4\pi$.
    In this integrand $\fT(\vec{\kappa})$ is a plane-wave:
    \[ \fT(\vec{\kappa}) = \exp(+i2\pi \vec{\kappa}\cdot\vec{\delta}_{T}), \]
    defined by the displacement-vector $\vec{\delta}_{T} \in \Real^{3}$.

    The output is:
    \[ (4\pi) \frac{ \sin(\kd) }{ \kd^{1} }, \]
    where the intermediate quantity $\kd$ is:
    \[ \kd = 2pi K \delta_{T}, \]
    where $\delta_{T}$ is the magnitude of the displacement $\vec{\delta}_{T}$.

    This function can also be used to calculate the following shell-limited inner-product (on the same shell specified by $|\vec{\kappa}|$ as above):
    \[ \int_{\epolara=0}^{\epolara=\pi} \int_{\eazimub=0}^{\eazimub=2\pi} \fS(\vec{\kappa})^{\dagger} \fM(\vec{\kappa}) d\eazimub \sin(\epolara)d\epolara \]
    where $\fS(\vec{\kappa})$ and $\fM(\vec{\kappa})$ are plane-waves defined via:
    \[ \fS(\vec{\kappa}) = \exp(+i2\pi \vec{\kappa}\cdot\vec{\delta}_{S}), \]
    \[ \fM(\vec{\kappa}) = \exp(+i2\pi \vec{\kappa}\cdot\vec{\delta}_{M}), \]
    each defined by their own displacement-vector.
    Here we simply multiply the two plane-waves together to get $\fT$ defined via:
    \[ \vec{\delta}_{T} & = & \vec{\delta}_{S} - \vec{\delta}_{M}. \]
    '''   
    if np.isscalar(kd): kd = torch.tensor([kd]).to(dtype=torch.float32) ;
    output = (4*pi) * (torch.sin(kd)) / torch.maximum(torch.tensor([1e-12]),kd) ;
    return output ;
