# Import built-in modules
import os

# Import third-party modules
import torch
import torch.nn as nn

class BaseModel(nn.Module):
    """ Base class for a network module, contains some predefined methods. """

    def __init__(self, name: str):
        # Call the parents module
        super().__init__()

        # Set the name of the network to default if the name is empty string
        self.name = name if name else type(self).__name__ + '_default'

    # -- Forward pass of a regular network {{{
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Basic forward pass; calls all modules in the order of definition. """
        for module in self.children():
            x = module.forward(x)
        return x
    # -- }}}

    # -- Utility methods of the class {{{
    def save_parameters(self, path: str):
        """ Save the network parameters in the correct place. """
        # Output path where the data will be stored
        output_path = os.path.join(path, type(self).__name__, self.name)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        torch.save(self.state_dict(), os.path.join(output_path, 'weights.pt'))
        print(f'  Parameters saved correctly at {output_path}', flush=True)

    def load_parameters(self, path: str):
        """ Load the network parameters from path/ClassName/name """
        # Output path where the data should be stored
        output_path = os.path.join(path, type(self).__name__, self.name, 'weights.pt')
        if not os.path.exists(output_path):
            print(f'  Parameters not found at path: {output_path}. New instance.', flush=True)
            return
        self.load_state_dict(torch.load(output_path))
        print(f'  Parameters loaded correctly from {output_path}', flush=True)
    # -- }}}

    # -- Property methods of the class {{{
    @property
    def identifier(self) -> str:
        """ String identifier of the object. """
        return type(self).__name__ + '_' + self.name

    @property
    def num_params(self) -> int:
        """ Get the number of parameters in the network. """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def device(self) -> torch.device:
        """ Get the device in which the model parameters are stored. """
        return next(self.parameters()).device
    # -- }}}
    
if __name__ == '__main__':
    pass
