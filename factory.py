# -- Import some built-in modules
from functools import cached_property
from typing import NamedTuple
from typing import Optional

# -- Import some base classes
from recan.factory import Ansatz
from recan.factory import Kernel

# -- Import some third-party modules
import torch

# -- Generate the namedtuple for the output
SPFitem = NamedTuple('SPFitem', 
    [
        ('C', torch.Tensor), ('R', torch.Tensor), ('L', torch.Tensor)
    ]
)

class SPFactory:
    """ Spectral function factory generator. """

    def __init__(self, ansatzs: list[Ansatz], kernel: Kernel):

        # Save a reference to the ansatzs and the kernel 
        self.ansatzs, self.kernel = ansatzs, kernel

        # Get the tensor used to unbound parameters
        self.scale = torch.cat([a.scale for a in self.ansatzs])
        self.trans = torch.cat([a.trans for a in self.ansatzs])

    # -- Magic methods of the class {{{
    def __len__(self) -> int:
        if not hasattr(self, 'L'):
            raise AttributeError('Data not found, please, generate it.')
        return self.L.shape[0]

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        return self.C[idx, :], self.L[idx, :]
    # -- }}}

    # -- Property methods of the class {{{
    @property
    def num_parameters(self) -> int:
        return sum(a.num_parameters for a in self.ansatzs)

    @property
    def Ns(self) -> int:
        return self.L.shape[1]

    @cached_property
    def rec_function(self):
        """ Reconstruction function that takes a tensor of parameters. """

        # Obtain the inmutable function for each ansatz
        functions = [a.function for a in self.ansatzs]

        # Generate the slice of a parameter object that will take each function
        limits, slope = [], 0

        # Fill the list with slices
        for a in self.ansatzs:
            limits += [slice(slope, slope + len(a))]
            slope  += len(a)

        # Generate the actual function that will do the reconstruction
        def actual_function(params: torch.Tensor):
            """ Function that returns the actual reconstructed R from a set of parameters. """
            return sum(
                f(self.kernel.omega, params[s]) for f, s in zip(functions, limits)
            )

        return actual_function
    # -- }}}

    def generate_data(self, Nb: int, Ns: int):

        # Generate some random parameters that define the spectral functions
        params = self.scale * torch.rand((Nb, self.num_parameters)) + self.trans

        # Reconstruct the spectral functions using the parameters
        R_buffer = torch.stack([self.rec_function(param) for param in params])

        # Normalise the spectral functions
        R_buffer = (6.0 / (R_buffer * self.kernel.dw).sum(dim=1).view(params.shape[0], 1)) * R_buffer

        # Compute the decomposition of the spectral functions
        U_buffer = torch.linalg.svd(R_buffer, full_matrices=False).Vh[:Ns, :].cpu()

        # Compute the correlation functions
        C_buffer = ((R_buffer @ self.kernel.kernel.T) * self.kernel.dw)

        # Save the data in memory
        self.L, self.C, self.U = R_buffer @ U_buffer.T, C_buffer, U_buffer

    def reconstruct(self, coeffs: torch.Tensor) -> SPFitem:
        """ Reconstruct the spectral function from the coefficients. """

        # Compute the spectral function by composing on the basis set
        R_buffer = coeffs @ self.U.to(coeffs.device)

        # Normalise the spectral functions
        R_buffer = (6.0 / (R_buffer * self.kernel.dw).sum(dim=1).view(coeffs.shape[0], 1)) * R_buffer

        # Compute the correlation functions
        C_buffer = ((R_buffer @ self.kernel.kernel.T) * self.kernel.dw)

        return SPFitem(C=C_buffer, R=R_buffer, L=coeffs)

def plot_robustness(L: torch.Tensor, noise: float, ex: torch.Tensor, S: float = 1.0, T: float = 0.0):
    """ Plot the robustness of the data in a given scale. """

    # Get several parameters to test robustness on natural scale
    L_data = S * L + T
    L_norm = S * L + T + noise * torch.randn(L.shape)
    L_unif = S * L + T + noise * torch.rand(L.shape)

    # Reconstruct the spectral functions from the data
    R_data = dataset.reconstruct(L_data)
    R_norm = dataset.reconstruct(L_norm)
    R_unif = dataset.reconstruct(L_unif)

    # Generate a figure to plot the data
    fig = plt.figure(figsize=(6, 4))

    # Add an axis to the figure
    axis = fig.add_subplot(1, 1, 1)

    # Add some information to the axis
    axis.set_xlabel(r'$\omega$')
    axis.set_ylabel(r'$\rho(\omega)$')
    axis.grid('#555555', alpha=0.6)

    # Plot the data -- Kernel is assumed accesible
    axis.plot(kernel.omega, R_data[ex, :].flatten(), label='Original')
    axis.plot(kernel.omega, R_norm[ex, :].flatten(), label='Normal')
    axis.plot(kernel.omega, R_unif[ex, :].flatten(), label='Uniform')

    # Add some properties to the axis
    axis.legend()

    # Add the title to the data
    fig.suptitle(f'noise={noise}, S={S:.4f}, T={T:.4f}')

    return fig

if __name__ == '__main__':
    pass

