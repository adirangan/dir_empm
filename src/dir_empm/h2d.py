from dir_empm.dir_matlab_macros import * ;
from scipy.special import jv

def h2d(kd):
    r'''
    This function provides a formula for the bandlimited integral of a plane-wave in 2d.
    It can also be used to calculate the bandlimited-inner-product of two plane-waves in 2d.

    More specifically, this function calculates the following integral:
    \[ \int_{k=0}^{k=K} \int_{\psi=0}^{\psi=2\pi} \fT(k,\psi) d\psi k\dk \]
    normalized so that the domain has size $(2\pi)^{2}$.
    In this integrand $\fT(k,\psi)$ is a plane-wave:
    \[ \fT(k,\psi) = \exp(+i2\pi k\delta_{T}\cos(\psi - \omega_{T})), \]
    defined by the displacement-vector:
    \[ \vec{\delta}_{T} = \delta_{T}\cdot\transpose{[\cos(\omega_{T}) , \sin(\omega_{T})]}. \]

    The output is:
    \[ (2\pi)^{2} (J_{0}(\kd) + J_{2}(\kd)), \]
    where $J_{v}(\cdot)$ is the standard bessel-function of order-$v$,
    and the intermediate quantity $\kd$ is:
    \[ \kd = 2pi K \delta_{T}, \]
    where $\delta_{T}$ is the magnitude of the displacement $\vec{\delta}_{T}$.

    This function can also be used to calculate the following bandlimited inner-product:
    \[ \int_{k=0}^{k=K} \int_{\psi=0}^{\psi=2\pi} \fS(k,\psi)^{\dagger} \fM(k,\psi) d\psi k\dk \]
    where $\fS(k,\psi)$ and $\fM(k,\psi)$ are plane-waves defined via:
    \[ \fS(k,\psi) = \exp(+i2\pi k\delta_{S}\cos(\psi - \omega_{S})), \]
    \[ \fM(k,\psi) = \exp(+i2\pi k\delta_{M}\cos(\psi - \omega_{M})), \]
    each defined by their displacement-vector:
    \[ \vec{\delta}_{S} = \delta_{S}\cdot\transpose{[\cos(\omega_{S}) , \sin(\omega_{S})]}, \]
    \[ \vec{\delta}_{M} = \delta_{M}\cdot\transpose{[\cos(\omega_{M}) , \sin(\omega_{M})]}. \]

    Here we simply multiply the two plane-waves together to get $\fT$ defined via:
    \begin{eqnarray}
    \vec{\delta}_{T} & = & \vec{\delta}_{S} - \vec{\delta}_{M} \\
    & = & \delta_{T}\cdot\transpose{[\cos(\omega_{T}) , \sin(\omega_{T})]}.
    \end{eqnarray}
    '''   
    if np.isscalar(kd): kd = torch.tensor([kd]).to(dtype=torch.float32) ;
    output = 4*pi**2 * (jv(0,kd) + jv(2,kd)) ;
    return output ;
