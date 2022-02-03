# -- Import some built-in modules
from functools import cached_property
from typing import NamedTuple
from typing import Optional

# -- Import some base classes
from .ansatz import Ansatz
from .kernel import Kernel

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

    # -- Methods of the class {{{
    def generate_data(self, Nb: int, Ns: int, use_GPU: bool = True) -> SPFitem:
        r""" 
        Generate a collection of Nb spectral functions using the ansatzs contained in the
        object. After generating the spectral functions, compute the SVD decomposition of
        the spectral function dataset to decompose the dataset into an expansion in terms of
        some coefficients [L] and some basis functions [U]. For each spectral function [R], there
        will be a collection of Ns coefficients that holds:

                    R(\omega) = \sum_{s=0}^{Ns} L_s U_s(\omega)self.

        The correlation function of each spectral function is computed using the spectral
        decomposition on the kernel [K] defined in the object:

                    C(\tau) = \int K(\tau, \omega) R(\omega) d\omega

        The method stores the data in memory under the fields: L, U and C. The spectral functions
        are not stored to avoid memory consumption.

        -- Parameters:
        Nb: int
            Number of random spectral functions to generate. It will dictate the number of
            coefficients and correlation functions in the object.
        Ns: int
            Number of basis functions/coefficients used in the decomposition of R.
        use_GPU: bool
            Use the GPU to compute the SVD decomposition of R to speed up the computation.

        -- Returns:
        SPFitem: NamedTuple[torch.Tensor, torch.Tensor, torch.Tensor]
            NamedTuple containing a reference to the coefficients [L], the spectral functions [R]
            and the correlation functions [C].
        """

        # Generate some random parameters that define the spectral functions
        params = self.scale * torch.rand((Nb, self.num_parameters)) + self.trans

        # Reconstruct the spectral functions using the parameters
        R_buffer = torch.stack([self.__get_reconstruct(param) for param in params])

        # Normalise the spectral functions
        R_buffer = (self.norm / (R_buffer * self.kernel.dw).sum(dim=1).view(params.shape[0], 1)) * R_buffer

        # Compute the correlation functions
        C_buffer = ((R_buffer @ self.kernel.kernel.T) * self.kernel.dw)

        # Compute the decomposition of the data
        U_buffer = torch.linalg.svd(R_buffer.cuda() if use_GPU else R_buffer, full_matrices=False).Vh[:Ns, :].cpu()

        # Save the needed data in memory: L, C and U
        self.L, self.C, self.U = (R_buffer @ U_buffer.T), C_buffer, U_buffer

        return SPFitem(C=C_buffer, R=R_buffer, L=self.L)

    def reconstruct(self, coeffs: torch.Tensor) -> SPFitem:
        """
        Reconstruct a set of spectral functions given some coefficients on the expansion
        in terms of the basis functions [U]. The tensor coeffs can be a collection on Nb 
        examples or just one example.

        -- Parameters:
        coeffs: torch.Tensor
            Coefficients from which we will compute their corresponding spectral and correlation
            functions.

        -- Returns:
        SPFitem: NamedTuple[torch.Tensor, torch.Tensor, torch.Tensor]
            NamedTuple containing a reference to the coefficients [L], the spectral functions [R]
            and the correlation functions [C].
        """

        # Transform the coefficients to a 2d shape if needed
        coeffs = coeffs if coeffs.ndim == 2 else coeffs.view(1, self.Ns)

        # Compute the spectral function by composing on the basis set
        R_buffer = coeffs @ self.U.to(coeffs.device)

        # Normalise the spectral functions
        R_buffer = (self.norm / (R_buffer * self.kernel.dw).sum(dim=1).view(coeffs.shape[0], 1)) * R_buffer

        # Compute the correlation functions
        C_buffer = ((R_buffer @ self.kernel.kernel.to(coeffs.device).T) * self.kernel.dw)

        return SPFitem(C=C_buffer, R=R_buffer, L=coeffs)
    # -- }}}

    # -- Magic methods of the class {{{
    def __len__(self) -> int:
        """ The length of the class is defined by the number of examples in the dataset. """
        if not hasattr(self, 'L'):
            raise AttributeError('Data not found, please, generate it.')
        return self.L.shape[0]

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        """ Retrieving an item from the class is the same as obtaining some L and C examples. """
        return self.C[idx, :], self.L[idx, :]
    # -- }}}

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
    def __get_reconstruct(self):
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
            return sum(f(self.kernel.omega, params[s]) for f, s in zip(functions, limits))

        return actual_function
    # -- }}}

if __name__ == '__main__':
    pass
