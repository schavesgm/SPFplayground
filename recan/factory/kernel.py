# -- Import built-in modules
from abc import ABCMeta
from abc import abstractmethod
from functools import lru_cache

# -- Import third party modules
import torch

class Kernel(metaclass = ABCMeta):
    def __init__(self, Nt: int, Nw: int, w_range: list[float]):
        # The energy range should be a two dimensional object
        assert len(w_range) == 2, f'{w_range =} must be a 2-dimensional object.'

        # Save the inverse of the temperature, the size of the energy space and the energy range
        self.Nt, self.Nw, self.w_range = Nt, Nw, sorted(w_range)

    # -- Private methods of the class {{{
    @lru_cache(maxsize=1)
    def __calculate_omega(self, Nw: int, w_min: float, w_max: float) -> torch.Tensor:
        """ Calculate the tensor containing all needed energies. """
        return torch.linspace(w_min, w_max, Nw)

    @lru_cache(maxsize=1)
    def __calculate_tau(self, Nt: int) -> torch.Tensor:
        """ Calculate the tensor containing all needed times. """
        return torch.arange(0, Nt)
    # -- }}}

    # -- Magic methods of the class {{{
    def __str__(self) -> str:
        """ String representation of the object. """
        return f'<{type(self).__name__}: Nt={self.Nt}, Nw={self.Nw}, w_range={self.w_range}>'

    def __repr__(self) -> str:
        """ String representation of the object. """
        return self.__str__()
    # -- }}}

    # -- Abstract methods of the base class {{{
    @abstractmethod
    def _calculate_kernel(self, Nt: int, Nw: int, w_min: float, w_max: float) -> torch.Tensor:
        """ Return the kernel as a torch tensor. """
        pass
    # -- }}}

    # -- Property methods of the class {{{
    @property
    def omega(self) -> torch.Tensor:
        """ Return a tensor of possible energies as a torch array. """
        return self.__calculate_omega(self.Nw, min(self.w_range), max(self.w_range))

    @property
    def tau(self) -> torch.Tensor:
        """ Return a tensor of possible times as a torch array. """
        return self.__calculate_tau(self.Nt)

    @property
    def kernel(self) -> torch.Tensor:
        """ Return a tensor containing the evaluated kernel. """
        return self._calculate_kernel(self.Nt, self.Nw, min(self.w_range), max(self.w_range))

    @property
    def dw(self) -> float:
        """ Resolution of the energy space. """
        return (max(self.w_range) - min(self.w_range)) / self.Nw

    @property
    def identifier(self) -> str:
        """ String identifier of the kernel. """
        return type(self).__name__ + f'_t{self.Nt}_w{self.Nw}_wr{min(self.w_range):.2f}:{max(self.w_range):.2f}'
    # -- }}}

class NRQCDKernel(Kernel):
    """ Non-relativistic QCD kernel definition. """
    @lru_cache(maxsize=1)
    def _calculate_kernel(self, Nt: int, Nw: int, w_min: float, w_max: float) -> torch.Tensor:
        """ Definition of the NRQCD kernel. Some parameters might not be used. """
        return (- self.tau.view(Nt, 1) * self.omega.view(1, Nw)).exp()

if __name__ == '__main__':
    pass
