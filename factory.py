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
SPFitem = NamedTuple('SPFitem', [('C', torch.Tensor), ('R', torch.Tensor), ('L', torch.Tensor)])

class SPFactory:
    """ Spectral function factory generator. """

    def __init__(self, ansatzs: list[Ansatz], kernel: Kernel, norm: float = 1.0):

        # Save a reference to the ansatzs and the kernel 
        self.ansatzs, self.kernel = ansatzs, kernel

        # Get the tensor used to unbound parameters
        self.scale = torch.cat([a.scale for a in self.ansatzs])
        self.trans = torch.cat([a.trans for a in self.ansatzs])

        # Save the normalisation number
        self.norm = norm

    # -- Magic methods of the class {{{
    def __len__(self) -> int:
        if not hasattr(self, 'L'):
            raise AttributeError('Data not found, please, generate it.')
        return self.L.shape[0]

    def __getitem__(self, idx) -> torch.Tensor:
        return self.reconstruct(self.L[idx, :])
    # -- }}}

    def generate_data(self, Nb: int, Ns: int) -> SPFitem:
        """ Generate some random spectral functions and their corresponding correlation functions. """

        # Generate some random parameters that define the spectral functions
        params = self.scale * torch.rand((Nb, self.num_parameters)) + self.trans

        # Reconstruct the spectral functions using the parameters
        R_buffer = torch.stack([self.rec_function(param) for param in params]) + 1e-6

        # Normalise the spectral functions
        R_buffer = (self.norm / (R_buffer * self.kernel.dw).sum(dim=1).view(params.shape[0], 1)) * R_buffer

        # Compute the correlation functions
        C_buffer = ((R_buffer @ self.kernel.kernel.T) * self.kernel.dw)

        # Compute the decomposition of the data
        U_buffer = torch.linalg.svd(R_buffer, full_matrices=False).Vh[:Ns, :].cpu()

        # Save the data in memory, we only need L and U
        self.L, self.U = (R_buffer @ U_buffer.T), U_buffer

        return SPFitem(C=C_buffer, R=R_buffer, L=self.L)

    def reconstruct(self, coeffs: torch.Tensor) -> SPFitem:
        """ Reconstruct the spectral function from the coefficients. """

        # Transform the coefficients to a 2d shape if needed
        coeffs = coeffs if coeffs.ndim == 2 else coeffs.view(1, self.Ns)

        # Compute the spectral function by composing on the basis set
        R_buffer = coeffs @ self.U.to(coeffs.device)

        # Normalise the spectral functions
        R_buffer = (self.norm / (R_buffer * self.kernel.dw).sum(dim=1).view(coeffs.shape[0], 1)) * R_buffer

        # Compute the correlation functions
        C_buffer = ((R_buffer @ self.kernel.kernel.to(coeffs.device).T) * self.kernel.dw)

        return SPFitem(C=C_buffer, R=R_buffer, L=coeffs)

    # -- Property methods of the class {{{
    @property
    def num_parameters(self) -> int:
        return sum(a.num_parameters for a in self.ansatzs)

    @property
    def Nb(self) -> int:
        return self.L.shape[0]

    @property
    def Ns(self) -> int:
        return self.L.shape[1]

    @property
    def Nt(self) -> int:
        return self.kernel.Nt

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

if __name__ == '__main__':
    pass
