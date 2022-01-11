# Import some built-in modules
from copy import deepcopy
from abc  import ABCMeta
from abc  import abstractproperty
from abc  import abstractstaticmethod

# Import some third party classes
import torch

class Parameter:
    """ Interface that defines a Parameter to be used in an ansatz. """
    def __init__(self, name: str, min: float, max: float):
        self.name, self.min, self.max = name, min, max

    def __str__(self) -> str:
        return f'<Parameter {self.name}: min={self.min}, max={self.max}>'

    def __repr__(self) -> str:
        return self.__str__()

class Ansatz(metaclass=ABCMeta):
    """ Abstract base class implementing an ansatz. """

    def __init__(self, *args):

        # Assert all args are parameters
        assert all(isinstance(p, Parameter) for p in args), \
            'All arguments must be instances of Parameter.'

        # Assert all param keys are contained in the names
        assert all(k in [p.name for p in args] for k in self.param_keys), \
            f'{self.param_keys} must be present in arguments: {[p.name for p in args]}'

        # List containing a deep copy of the arguments in the correct order
        deep_copied_args = []

        # Make a deep copy of the arguments in the correct order
        for key in self.param_keys:
            for p in args:
                if p.name == key:
                    deep_copied_args += [deepcopy(p)]

        # Save all relevant parameters
        self.params = deep_copied_args

        # Save a reference to be parameter as an independent field
        for p in self.params: setattr(self, p.name, p)

        # Get all the matrices used to transform from unbounded to bounded
        self.scale = torch.tensor([p.max - p.min for p in self.params])
        self.trans = torch.tensor([p.min for p in self.params])

    def sample_values(self) -> torch.Tensor:
        """ Sample some random parameters from the ansatz taking into accounts links. """
        return torch.rand((self.num_parameters,), dtype=torch.float32)

    def bound_values(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x + self.trans

    def reconstruct(self, omega: torch.Tensor, params: torch.Tensor):
        """ Get the energy representation of the ansatz at given parameters. """
        return self.function(omega, params)

    # -- Magic methods of the class {{{
    def __str__(self) -> str:
        return f'<{type(self).__name__}: {[p.name for p in self.params]}>'

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self) -> int:
        return self.num_parameters
    # -- }}}

    # -- Abstract methods of the class to be defined by childs {{{
    @abstractproperty
    def param_keys(self) -> tuple[str]:
        """ Param keys should be a global class property. Define it in header. """
        pass

    @abstractstaticmethod
    def function(omega: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """ Transform the parameters into the energy representation of the ansatz. """
        pass
    # -- }}}

    # -- Property methods of the class {{{
    @property
    def num_parameters(self) -> int:
        """ Return the number of parameters defining this ansatz """
        return len(self.param_keys)
    # -- }}}

# -- Gaussian Ansatz specification {{{
class GaussianAnsatz(Ansatz):
    """ Definition of the gaussian ansatz. """

    # Set the parameter keys used in this ansatz
    param_keys: tuple = ('A', 'M', 'W')

    @staticmethod
    def function(omega: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """ Function used to instantiate the ansatz in the energy space. """
        return params[0] * (-0.25 * (omega - params[1]).pow(2) / (params[2] ** 2)).exp()
# -- }}}

if __name__ == '__main__':
    pass
