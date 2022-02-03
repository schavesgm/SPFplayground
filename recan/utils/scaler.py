import torch

class StandardScaler:
    """ StandardScaler class that scales and descales tensors using standard normal. """

    def __init__(self, data: torch.Tensor):
        # Compute the mean and the standard deviation of the data provided
        self.mu, self.sigma = data.mean(), data.std()

    def scale(self, data: torch.Tensor) -> torch.Tensor:
        """ Transform data: X -> ZX = (X - mu) / sigma. """
        return (data - self.mu) / self.sigma

    def descale(self, data: torch.Tensor) -> torch.Tensor:
        """ Transform data: ZX -> X = (ZX * sigma) + mu. """
        return data * self.sigma + self.mu

if __name__ == "__main__":
    pass
