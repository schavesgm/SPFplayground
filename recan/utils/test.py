# -- Import built-in modules
from dataclasses import dataclass, field

# -- Import third-party modules
import torch

# -- Import user-defined modules
from ..models import BaseModel

class Dataset:
    """ Dataset wrapper over pairs input/labels to be used in conjunction with torch DataLoader """
    
    def __init__(self, input: torch.Tensor, label: torch.Tensor) -> None:
        """ 
        The input and label data must be, at least, two dimensional tensors with the same
        size in the first dimension, which represents the number of examples.
        """

        # Assert that the number of examples are the same and the dimensions are correct
        assert input.ndim > 1, f'{input.ndim} must be at least 2'
        assert label.ndim > 1, f'{label.ndim} must be at least 2'
        assert input.shape[0] == label.shape[0], f'{input.shape} must have the same examples as {label.shape}'

        # Save a reference to the data
        self.input, self.label = input, label

    def __len__(self) -> int:
        """ The length of the dataset is just the number of examples in it. """
        return self.input.shape[0]

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        """ Accessing the dataset retrieves a tuple of input/label data """
        return self.input[idx, :], self.label[idx, :]

@dataclass
class TestResults:
    """ Container for the test results as a dataclass. """
    
    # Loss function over the test set
    losses: list[float] = field(default_factory=lambda: [])

    # Examples used in the plotting
    plot_examples: torch.Tensor = torch.tensor([])

    # Input, label and prediction data to be plotted
    input_plotting: torch.Tensor = torch.tensor([])
    label_plotting: torch.Tensor = torch.tensor([])
    preds_plotting: torch.Tensor = torch.tensor([])

def test_model(model: BaseModel, input: torch.Tensor, label: torch.Tensor, plot_examples: torch.Tensor) -> TestResults:
    """ Test a model on a set of inputs and labels. """

    # Set the model in evaluation mode
    model.eval()

    # Generate a DataLoader on input and label
    loader = torch.utils.data.DataLoader(Dataset(input, label), batch_size=256, shuffle=False)

    # Generate a TestResults object to save the data
    test_results = TestResults(plot_examples=plot_examples)

    # Buffers that will hold the plotting examples
    input_plotting, label_plotting, preds_plotting = [], [], []

    # Iterate through all datas
    for mb, (input_mb, label_mb) in enumerate(loader):

        # Check if any of the examples are in the region
        ex_present = (mb * loader.batch_size <= plot_examples) * (plot_examples < (mb + 1) * loader.batch_size)

        # Get all examples that are inside bounds and bound them to batch_size
        inside_bounds = plot_examples[ex_present] - mb * loader.batch_size

        # Compute the total prediction of the network
        preds_mb = model(input_mb.to(model.device))

        # Compute the loss function in this minibatch -> MSError for now
        loss = (label_mb.to(model.device) - preds_mb).pow(2).mean()

        # Append the loss to the test_results
        test_results.losses += [loss.item()]

        # Get the examples in the plotting collection
        if len(inside_bounds) > 0:
            input_plotting += [input_mb[inside_bounds, :].cpu()]
            label_plotting += [label_mb[inside_bounds, :].cpu()]
            preds_plotting += [preds_mb[inside_bounds, :].cpu()]

    # Save the examples in the test results
    test_results.input_plotting = torch.vstack(input_plotting)
    test_results.label_plotting = torch.vstack(label_plotting)
    test_results.preds_plotting = torch.vstack(preds_plotting)

    # Set the model in training mode
    model.train()

    return test_results

if __name__ == "__main__":
    pass
